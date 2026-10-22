//go:build linux

package main

import (
	"bytes"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

var cudaSupportPersistenceRetained []error

type cudaSupportPersistenceOwner struct {
	state     *cudaRequestState
	device    *cuda.Device
	file      *os.File
	arenas    []*cudamem.Arena
	snapshots []io.Closer
}

func (o *cudaSupportPersistenceOwner) Close() error {
	for len(o.snapshots) != 0 {
		if err := o.snapshots[0].Close(); err != nil {
			return err
		}
		o.snapshots = o.snapshots[1:]
	}
	if o.state != nil && o.state.dspark != nil {
		if err := o.state.dspark.Close(); err != nil {
			return err
		}
	}
	for len(o.arenas) != 0 {
		if err := o.arenas[0].Free(); err != nil {
			return err
		}
		o.arenas = o.arenas[1:]
	}
	if o.file != nil {
		if err := o.file.Close(); err != nil {
			return err
		}
		o.file = nil
	}
	if o.device != nil {
		if err := o.device.Close(); err != nil {
			return err
		}
		o.device = nil
	}
	return nil
}

// Real native backing and adapter transactions, with a synthetic empty target
// core. This fixture does not evaluate a target model or a support prediction.
func cudaSupportPersistenceFixture(t *testing.T, position int) (*cudaRequestState, *cudaSupportPersistenceOwner) {
	t.Helper()
	o := new(cudaSupportPersistenceOwner)
	t.Cleanup(func() {
		if err := finishBackendCleanup(nil, o); err != nil {
			cudaSupportPersistenceRetained = append(cudaSupportPersistenceRetained, err)
			t.Errorf("retain native support owner: %v", err)
		}
	})
	d, err := cuda.NewDevice(0)
	o.device = d
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.CreateTemp(t.TempDir(), "identity")
	o.file = file
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString("synthetic state model"); err != nil {
		t.Fatal(err)
	}
	r := &cudaRuntime{cudaFoundation: &cudaFoundation{device: d, file: file, gguf: &gguf.File{}, stateIdentity: new(modelStateIdentityCache)}, ctx: 3, pos: position}
	s := &cudaRequestState{runtime: r, generation: 8}
	o.state = s
	a := cudamem.NewArena(d)
	o.arenas = append(o.arenas, a)
	rows, values, err := dsparkFeatureJournalShape(r.ctx, 1, 2, Flash.Embed)
	if err != nil {
		t.Fatal(err)
	}
	j, err := newDSparkFeatureJournal(rows, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < position; i++ {
		if err := j.append(i, i+1, 1); err != nil {
			t.Fatal(err)
		}
	}
	f := &cudaDSparkFeatures{r: r, request: s, modelSHA: strings.Repeat("a", 64), block: 1, targets: []int{0, 1}, captured: make([]bool, 2), position: -1, token: -1}
	s.dspark = f
	f.features = cudamem.AllocIn[float32](a, "support-features", 2*Flash.Embed)
	f.status = cudamem.AllocIn[uint32](a, "support-status", 1)
	f.history = &cudaDSparkHistory{owner: f, values: cudamem.AllocIn[float32](a, "support-history", values), journal: j, width: 2 * Flash.Embed}
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	history := make([]float32, values)
	for i := range history {
		history[i] = float32(position*7 + i%13)
	}
	for _, err := range []error{f.features.CopyFromHost(make([]float32, f.features.Len())), f.status.CopyFromHost([]uint32{0}), f.history.values.CopyFromHost(history)} {
		if err != nil {
			t.Fatal(err)
		}
	}
	return s, o
}

func TestCUDASupportAdapterRoundTripAndFaults(t *testing.T) {
	if os.Getenv("DS4_SUPPORT_CODEC_NATIVE") != "1" {
		t.Skip("requires separately authorized native support codec gate")
	}
	source, sourceOwner := cudaSupportPersistenceFixture(t, 2)
	var encoded bytes.Buffer
	if err := source.Save(&encoded); err != nil {
		t.Fatal(err)
	}
	destination, _ := cudaSupportPersistenceFixture(t, 0)
	if err := destination.Load(bytes.NewReader(encoded.Bytes())); err != nil {
		t.Fatal(err)
	}
	var reencoded bytes.Buffer
	if err := destination.Save(&reencoded); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded.Bytes(), reencoded.Bytes()) || !reflect.DeepEqual(source.dspark.history.journal, destination.dspark.history.journal) {
		t.Fatal("actual adapter lost support history or frontier")
	}
	for _, name := range []string{"stage", "backup", "after-core", "rollback", "backup-cleanup", "success-cleanup"} {
		t.Run(name, func(t *testing.T) {
			dst, owner := cudaSupportPersistenceFixture(t, 0)
			var before bytes.Buffer
			if err := dst.Save(&before); err != nil {
				t.Fatal(err)
			}
			fault := errors.New("injected " + name)
			hooks := &cudaSupportLoadHooks{}
			calls := 0
			hooks.snapshot = func() (dsv4.Snapshot, error) {
				calls++
				if name == "backup" && calls == 2 {
					return nil, fault
				}
				v, err := dst.Snapshot()
				if v != nil {
					owner.snapshots = append(owner.snapshots, v)
				}
				if err == nil && (name == "stage" || ((name == "backup-cleanup" || name == "success-cleanup") && calls == 2)) {
					closed := false
					v.(*cudaStateSnapshot).closeArena = func(a *cudamem.Arena) error {
						if !closed {
							closed = true
							return fault
						}
						return a.Free()
					}
				}
				return v, err
			}
			if name == "stage" {
				hooks.afterStageTransfer = func() error { return fault }
			} else if name == "after-core" || name == "rollback" || name == "backup-cleanup" {
				hooks.afterCore = func() error {
					if dst.runtime.pos != 2 {
						t.Fatal("fault did not follow actual core replacement")
					}
					return fault
				}
			}
			if name == "rollback" {
				hooks.beforeRollback = func() error { return fault }
			}
			err := dst.loadSupportWithHooks(bytes.NewReader(encoded.Bytes()), hooks)
			if !errors.Is(err, fault) {
				t.Fatalf("fault not retained: %v", err)
			}
			if name == "rollback" {
				if !errors.Is(err, dsv4.ErrDesynchronized) || !errors.Is(dst.runtime.failed, dsv4.ErrDesynchronized) {
					t.Fatal("failed rollback did not poison destination")
				}
				return
			}
			if name == "stage" || name == "backup-cleanup" || name == "success-cleanup" {
				var closer io.Closer
				if !errors.As(err, &closer) || len(owner.snapshots) == 0 || owner.snapshots[0].(*cudaStateSnapshot).arena == nil {
					t.Fatal("failed staging cleanup lost its actual owner")
				}
				for _, v := range owner.snapshots {
					if v.(*cudaStateSnapshot).arena == nil {
						t.Fatal("failed backup cleanup released a dependent owner early")
					}
				}
				if err := closer.Close(); err != nil {
					t.Fatal(err)
				}
				for _, v := range owner.snapshots {
					if v.(*cudaStateSnapshot).arena != nil {
						t.Fatal("cleanup retry did not release staging and backup arenas")
					}
				}
			}
			var after bytes.Buffer
			if err := dst.Save(&after); err != nil {
				t.Fatal(err)
			}
			want := before.Bytes()
			if name == "success-cleanup" {
				want = encoded.Bytes()
			}
			if !bytes.Equal(want, after.Bytes()) {
				t.Fatal("failed transaction changed destination")
			}
		})
	}
	v, err := source.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	sourceOwner.snapshots = append(sourceOwner.snapshots, v)
	sizer, ok := v.(dsv4.SnapshotSizer)
	if !ok {
		t.Fatal("support snapshot size prerequisite is missing")
	}
	if n, err := sizer.SnapshotBytes(); err != nil || n != int64(encoded.Len()) {
		t.Fatalf("snapshot count versus actual Save: %d: %v", n, err)
	}
	f := source.dspark
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	f.modelSHA = strings.Repeat("b", 64)
	if err := sourceOwner.file.Close(); err != nil {
		t.Fatal(err)
	}
	sourceOwner.file = nil
	var closedOwner bytes.Buffer
	if err := writeCUDAStateSnapshot(&closedOwner, v.(*cudaStateSnapshot), [32]byte{}, [32]byte{}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded.Bytes(), closedOwner.Bytes()) {
		t.Fatal("snapshot serialization depends on closed support identity owner")
	}
	if n, err := sizer.SnapshotBytes(); err != nil || n != int64(closedOwner.Len()) {
		t.Fatalf("closed-owner snapshot count: %d: %v", n, err)
	}
}
