//go:build darwin && arm64

package main

import (
	"bytes"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

var supportPersistenceRetained []error

type metalSupportPersistenceOwner struct {
	state     *metalRequestState
	dev       *libmetal.Device
	ops       *metalPinnedOps
	arenas    []*metalmem.Arena
	snapshots []io.Closer
}

func (o *metalSupportPersistenceOwner) Close() error {
	for len(o.snapshots) != 0 {
		if err := o.snapshots[0].Close(); err != nil {
			return err
		}
		o.snapshots = o.snapshots[1:]
	}
	if o.state != nil {
		if err := o.state.runtime.synchronizeState(); err != nil {
			return err
		}
		if f := o.state.runtime.dsparkFeatures; f != nil {
			if err := f.Close(); err != nil {
				return err
			}
		}
		o.state = nil
	}
	// These two APIs return no error. Drain above must succeed before their
	// owners can be released; never manufacture an error result for them.
	for _, a := range o.arenas {
		a.Free()
	}
	o.arenas = nil
	if o.ops != nil {
		o.ops.Close()
		o.ops = nil
	}
	if o.dev != nil {
		if err := o.dev.Close(); err != nil {
			return err
		}
		o.dev = nil
	}
	return nil
}

// This fixture exercises actual adapter Save/Load and native support backing,
// not a target forward or a model parity claim.
func metalSupportPersistenceFixture(t *testing.T, position int) (*metalRequestState, *metalSupportPersistenceOwner) {
	t.Helper()
	owner := new(metalSupportPersistenceOwner)
	t.Cleanup(func() {
		if err := finishBackendCleanup(nil, owner); err != nil {
			supportPersistenceRetained = append(supportPersistenceRetained, err)
			t.Errorf("retain native support owner: %v", err)
		}
	})
	dev, err := libmetal.NewDevice()
	owner.dev = dev
	if err != nil {
		t.Fatal(err)
	}
	ops, err := newMetalPinnedOps(dev)
	owner.ops = ops
	if err != nil {
		t.Fatal(err)
	}
	r := &metalRuntime{dev: dev, ops: ops, ctx: 3, pos: position, arena: metalmem.NewArena(dev)}
	owner.arenas = append(owner.arenas, r.arena)
	metalStateFixtureIdentity(t, r)
	s := &metalRequestState{runtime: r, owner: &metalStateAdapter{}}
	owner.state = s
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
	a := metalmem.NewArena(dev)
	owner.arenas = append(owner.arenas, a)
	f := &metalDSparkFeatures{request: s, weights: &metalDSparkWeights{device: dev, model: &dsparkModel{sha256: strings.Repeat("a", 64), block: 1}, leases: 1}, arena: a, targets: []int{0, 1}, captured: make([]bool, 2), journal: j, epoch: 1, position: -1, token: -1}
	f.data = metalmem.AllocIn[float32](a, "support-persistence-history", values)
	f.status = metalmem.AllocIn[uint32](a, "support-persistence-status", 1)
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	for i := range f.data.HostSlice() {
		f.data.HostSlice()[i] = float32(position*7 + i%13)
	}
	r.dsparkFeatures = f
	return s, owner
}

func TestMetalSupportAdapterRoundTrip(t *testing.T) {
	if os.Getenv("DS4_SUPPORT_CODEC_NATIVE") != "1" {
		t.Skip("requires separately authorized native support codec gate")
	}
	source, sourceOwner := metalSupportPersistenceFixture(t, 2)
	destination, _ := metalSupportPersistenceFixture(t, 0)
	var encoded bytes.Buffer
	if err := source.Save(&encoded); err != nil {
		t.Fatal(err)
	}
	if err := destination.Load(bytes.NewReader(encoded.Bytes())); err != nil {
		t.Fatal(err)
	}
	if destination.runtime.pos != 2 || !reflect.DeepEqual(source.runtime.dsparkFeatures.journal, destination.runtime.dsparkFeatures.journal) || !reflect.DeepEqual(source.runtime.dsparkFeatures.data.HostSlice(), destination.runtime.dsparkFeatures.data.HostSlice()) {
		t.Fatal("actual adapter lost support history or frontier")
	}
	for _, name := range []string{"model", "platform", "layer-order"} {
		t.Run(name, func(t *testing.T) {
			var before bytes.Buffer
			if err := destination.Save(&before); err != nil {
				t.Fatal(err)
			}
			data := append([]byte(nil), encoded.Bytes()...)
			_, support, _, err := readDSparkSupportContainer(bytes.NewReader(data), false, uint64(metalStateMaxBytes))
			if err != nil {
				t.Fatal(err)
			}
			start := len(data) - len(support)
			switch name {
			case "model":
				data[start+9] ^= 1
			case "platform":
				data[8] = 1
			case "layer-order":
				first := append([]byte(nil), support[53:57]...)
				copy(data[start+53:start+57], support[57:61])
				copy(data[start+57:start+61], first)
			}
			if err := destination.Load(bytes.NewReader(data)); err == nil {
				t.Fatal("foreign support accepted")
			}
			var after bytes.Buffer
			if err := destination.Save(&after); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(before.Bytes(), after.Bytes()) {
				t.Fatal("validation failure changed destination")
			}
		})
	}
	// Actual feature Close releases the original native arena and detaches the
	// request. The snapshot's host payload and copied identity remain usable.
	snapshot, err := source.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	sourceOwner.snapshots = append(sourceOwner.snapshots, snapshot)
	sizer, ok := snapshot.(dsv4.SnapshotSizer)
	if !ok {
		t.Fatal("support snapshot size prerequisite is missing")
	}
	if n, err := sizer.SnapshotBytes(); err != nil || n != int64(encoded.Len()) {
		t.Fatalf("snapshot count versus actual Save: %d: %v", n, err)
	}
	f := source.runtime.dsparkFeatures
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	f.weights.model.sha256 = strings.Repeat("b", 64)
	var closedOwner bytes.Buffer
	if err := writeMetalStateSnapshot(&closedOwner, snapshot.(*metalSnapshot)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded.Bytes(), closedOwner.Bytes()) {
		t.Fatal("snapshot serialization depends on closed support owner")
	}
	if n, err := sizer.SnapshotBytes(); err != nil || n != int64(closedOwner.Len()) {
		t.Fatalf("closed-owner snapshot count: %d: %v", n, err)
	}
}
