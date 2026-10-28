package nvtelemetry_test

import (
	"context"
	"errors"
	"io/fs"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// permissionFS denies one configured snapshot path. It models an access error;
// it does not exercise procfs, Slurm, credentials or a platform permission policy.
type permissionFS struct {
	base fs.FS
	path string
	err  error
	mu   sync.Mutex
	seen []string
}

func (p *permissionFS) Open(name string) (fs.File, error) {
	p.mu.Lock()
	p.seen = append(p.seen, name)
	p.mu.Unlock()
	if name == p.path {
		return nil, &fs.PathError{Op: "open", Path: name, Err: p.err}
	}
	return p.base.Open(name)
}

func (p *permissionFS) names() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.seen)
}

func TestPermissionOptionalProcess(t *testing.T) {
	for _, name := range []string{"stat", "status", "cgroup"} {
		t.Run(name, func(t *testing.T) {
			m := refreshFixtureFS(t)
			for _, part := range []string{"stat", "status", "cgroup"} {
				b := *m["proc/1234/"+part]
				b.Data = []byte(strings.ReplaceAll(string(b.Data), "1234", "5678"))
				m["proc/5678/"+part] = &b
			}
			src := refreshSource()
			src.PIDs = []uint32{1234, 5678}
			p := &permissionFS{base: m, path: "proc/5678/" + name, err: fs.ErrPermission}
			got, err := nvtelemetry.ReadSlurmAttributionSnapshot(p, src)
			if err != nil {
				t.Fatalf("optional process denial failed the read: %v", err)
			}
			if !slices.Equal(got.Withheld, []uint32{5678}) || len(got.Snapshot.Processes) != 1 || got.Snapshot.Processes[0].Process.Process.PID != 1234 {
				t.Fatalf("denied process was not withheld independently: %+v", got)
			}
			allowed := map[string]bool{"squeue.json": true, "mappings.json": true, "executions.json": true}
			for _, pid := range []string{"1234", "5678"} {
				for _, part := range []string{"stat", "status", "cgroup"} {
					allowed["proc/"+pid+"/"+part] = true
				}
			}
			for _, opened := range p.names() {
				if !allowed[opened] {
					t.Errorf("unexpected snapshot access %q", opened)
				}
			}
			if !slices.Contains(p.names(), p.path) {
				t.Fatal("denied path was not reached")
			}
		})
	}
}

func TestPermissionRequiredSource(t *testing.T) {
	for _, name := range []string{"squeue.json", "mappings.json", "executions.json"} {
		t.Run(name, func(t *testing.T) {
			p := &permissionFS{base: refreshFixtureFS(t), path: name, err: fs.ErrPermission}
			got, err := nvtelemetry.ReadSlurmAttributionSnapshot(p, refreshSource())
			if !errors.Is(err, fs.ErrPermission) {
				t.Fatalf("required source denial = %v, want permission error", err)
			}
			if !reflect.DeepEqual(got, nvtelemetry.AttributionRead{}) {
				t.Fatalf("required source denial published partial evidence: %+v", got)
			}
			if !slices.Contains(p.names(), name) {
				t.Fatal("required source was not reached")
			}
		})
	}
}

func TestPermissionOtherProcessError(t *testing.T) {
	other := errors.New("snapshot read failed")
	for _, name := range []string{"stat", "status", "cgroup"} {
		t.Run(name, func(t *testing.T) {
			p := &permissionFS{base: refreshFixtureFS(t), path: "proc/1234/" + name, err: other}
			got, err := nvtelemetry.ReadSlurmAttributionSnapshot(p, refreshSource())
			if !errors.Is(err, other) {
				t.Fatalf("other process error = %v, want original error", err)
			}
			if !reflect.DeepEqual(got, nvtelemetry.AttributionRead{}) {
				t.Fatalf("other process error became withheld evidence: %+v", got)
			}
		})
	}
}

func TestPermissionRefresherRequiredSource(t *testing.T) {
	for _, name := range []string{"squeue.json", "mappings.json", "executions.json"} {
		t.Run(name, func(t *testing.T) {
			h := newAttributionHarness(t, refreshTestConfig)
			deny := &permissionFS{base: refreshFixtureFS(t), path: name, err: fs.ErrPermission}
			h.set(deny, nil)
			if err := h.r.Step(context.Background()); !errors.Is(err, fs.ErrPermission) {
				t.Fatalf("initial denied step = %v, want permission error", err)
			}
			if h.r.Ready() {
				t.Fatal("denied first read made source ready")
			}
			if got := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); got.Assignment.State != "unknown" || got.Reason != "source_stale" {
				t.Fatalf("denied first read created an assignment: %+v", got)
			}
			h.set(refreshFixtureFS(t), nil)
			if err := h.r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			if got := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); got.Assignment.State != "assigned" || got.Assignment.Owner.JobID != "42" {
				t.Fatalf("allowed source did not assign known job: %+v", got)
			}
			entries, last, fails := h.table.Stats()
			h.set(deny, nil)
			h.clock.Advance(time.Second)
			if err := h.r.Step(context.Background()); !errors.Is(err, fs.ErrPermission) {
				t.Fatalf("replacement denied step = %v, want permission error", err)
			}
			if e, l, f := h.table.Stats(); e != entries || l != last || f != fails {
				t.Fatalf("denied read changed table: (%d,%d,%d), want (%d,%d,%d)", e, l, f, entries, last, fails)
			}
			if !h.r.Ready() {
				t.Fatal("previous success was discarded before MaxAge")
			}
			h.clock.Advance(5 * time.Second)
			if h.r.Ready() {
				t.Fatal("denied source stayed ready past MaxAge")
			}
			if got := h.r.Resolve(refreshQueryProcess(1000), refreshDeviceUUID, 200); got.Assignment.State != "unknown" || got.Reason != "source_stale" {
				t.Fatalf("denied stale source returned assignment: %+v", got)
			}
		})
	}
}
