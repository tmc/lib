package nvtelemetry_test

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"slices"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func withProcState(stat, state string) string {
	i := strings.LastIndexByte(stat, ')')
	fields := strings.Fields(stat[i+1:])
	fields[0] = state
	return stat[:i+1] + " " + strings.Join(fields, " ") + "\n"
}

func TestProcEntityNonLive(t *testing.T) {
	stat := string(refreshFixtureFS(t)["proc/1234/stat"].Data)
	for _, state := range []string{"Z", "X", "x"} {
		t.Run(state, func(t *testing.T) {
			got, err := nvtelemetry.ReadProcEntity(strings.NewReader(withProcState(stat, state)), strings.NewReader(""), strings.NewReader(""), "boot", "node", "pidns")
			if !errors.Is(err, nvtelemetry.ErrProcEntityNotLive) {
				t.Fatalf("state %s: error = %v, want ErrProcEntityNotLive", state, err)
			}
			if got.Process.Process.PID != 0 || got.StartTimeTicks != 0 || len(got.Cgroups) != 0 {
				t.Fatalf("non-live process returned facts: %+v", got)
			}
		})
	}
	for _, state := range []string{"R", "S", "D", "T", "t", "I"} {
		t.Run(state, func(t *testing.T) {
			got, err := nvtelemetry.ReadProcEntity(strings.NewReader(withProcState(stat, state)), strings.NewReader(""), strings.NewReader(""), "boot", "node", "pidns")
			if err != nil || got.Process.Process.PID != 1234 || got.StartTimeTicks != 1000 {
				t.Fatalf("state %s: facts = %+v, error = %v", state, got, err)
			}
		})
	}
	_, err := nvtelemetry.ReadProcEntity(strings.NewReader(withProcState(stat, "ZZ")), strings.NewReader(""), strings.NewReader(""), "boot", "node", "pidns")
	if err == nil || errors.Is(err, nvtelemetry.ErrProcEntityNotLive) {
		t.Fatalf("malformed state: error = %v, want parse error", err)
	}
}

type procReadErrorFS struct {
	fs.FS
	name string
	err  error
}

func (f procReadErrorFS) Open(name string) (fs.File, error) {
	if name == f.name {
		return nil, &fs.PathError{Op: "open", Path: name, Err: f.err}
	}
	return f.FS.Open(name)
}

func TestProcSnapshotWithholdsUnavailable(t *testing.T) {
	for _, name := range []string{"stat", "status", "cgroup"} {
		t.Run("permission/"+name, func(t *testing.T) {
			m := refreshFixtureFS(t)
			for _, file := range []string{"stat", "status", "cgroup"} {
				data := strings.ReplaceAll(string(m["proc/1234/"+file].Data), "1234", "5678")
				m["proc/5678/"+file] = &fstest.MapFile{Data: []byte(data)}
			}
			src := refreshSource()
			src.PIDs = []uint32{1234, 5678}
			got, err := nvtelemetry.ReadSlurmAttributionSnapshot(procReadErrorFS{m, "proc/1234/" + name, fs.ErrPermission}, src)
			if err != nil || !slices.Equal(got.Withheld, []uint32{1234}) || len(got.Snapshot.Processes) != 1 || got.Snapshot.Processes[0].Process.Process.PID != 5678 {
				t.Fatalf("read = %+v, error = %v; want one withheld and healthy pid 5678", got, err)
			}
		})
	}
	for _, state := range []string{"Z", "X", "x"} {
		t.Run("state/"+state, func(t *testing.T) {
			m := refreshFixtureFS(t)
			m["proc/1234/stat"] = &fstest.MapFile{Data: []byte(withProcState(string(m["proc/1234/stat"].Data), state))}
			got, err := nvtelemetry.ReadSlurmAttributionSnapshot(m, refreshSource())
			if err != nil || !slices.Equal(got.Withheld, []uint32{1234}) || len(got.Snapshot.Processes) != 0 {
				t.Fatalf("read = %+v, error = %v; want withheld non-live pid", got, err)
			}
		})
	}
}

func TestProcSnapshotKeepsSourceErrors(t *testing.T) {
	for _, name := range []string{"squeue.json", "mappings.json", "executions.json"} {
		t.Run(name, func(t *testing.T) {
			_, err := nvtelemetry.ReadSlurmAttributionSnapshot(procReadErrorFS{refreshFixtureFS(t), name, fs.ErrPermission}, refreshSource())
			if !errors.Is(err, fs.ErrPermission) {
				t.Fatalf("shared file error = %v, want permission error", err)
			}
		})
	}
	readErr := errors.New("proc read failed")
	_, err := nvtelemetry.ReadSlurmAttributionSnapshot(procReadErrorFS{refreshFixtureFS(t), "proc/1234/stat", readErr}, refreshSource())
	if !errors.Is(err, readErr) {
		t.Fatalf("process I/O error = %v, want original error", err)
	}
	m := refreshFixtureFS(t)
	m["proc/1234/stat"] = &fstest.MapFile{Data: []byte("malformed stat")}
	if _, err := nvtelemetry.ReadSlurmAttributionSnapshot(m, refreshSource()); err == nil {
		t.Fatal("malformed process stat was accepted")
	}
}

func TestProcRefreshRemovesUnavailableAssignment(t *testing.T) {
	for _, kind := range []string{"permission", "zombie"} {
		t.Run(kind, func(t *testing.T) {
			h := newAttributionHarness(t, refreshTestConfig)
			if err := h.r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if got := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); got.Assignment.State != "assigned" {
				t.Fatalf("initial attribution = %+v", got)
			}
			m := refreshFixtureFS(t)
			var fsys fs.FS = procReadErrorFS{m, "proc/1234/stat", fs.ErrPermission}
			if kind == "zombie" {
				m["proc/1234/stat"] = &fstest.MapFile{Data: []byte(withProcState(string(m["proc/1234/stat"].Data), "Z"))}
				fsys = m
			}
			h.set(fsys, nil)
			if err := h.r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if got := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); got.Assignment.State != "unknown" {
				t.Fatalf("unavailable process retained attribution: %+v", got)
			}
			if st := h.r.Status(); st.Failures != 0 || st.Withheld != 1 || st.Records != 0 {
				t.Fatalf("refresh status = %+v", st)
			}
		})
	}
}

func ExampleReadProcEntity_nonLive() {
	stat := "12 (worker) Z" + strings.Repeat(" 0", 18) + " 100\n"
	_, err := nvtelemetry.ReadProcEntity(strings.NewReader(stat), strings.NewReader(""), strings.NewReader(""), "boot", "node", "pidns")
	fmt.Println(errors.Is(err, nvtelemetry.ErrProcEntityNotLive))
	// Output: true
}
