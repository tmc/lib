package cudahotswap

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

// Until Push is installed, tests publish a second generation explicitly. This
// is not a fake Push path or a source-to-device qualification claim.
func addSelectionVersion(t *testing.T, s *Service) (*generation, *serviceKernel) {
	t.Helper()
	s.mutations.Lock()
	defer s.mutations.Unlock()
	p, _, err := s.table.pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	k := &serviceKernel{raw: 2}
	g := &generation{digest: "second-source", flags: []string{"--fmad=false"}, signature: p.gen.signature, function: k.raw, owner: k}
	stamp, err := s.table.publish("kernel", g, p.gen.version)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.release(); err != nil {
		t.Fatal(err)
	}
	s.mu.Lock()
	state := s.states["kernel"]
	state.Generation, state.Version = stamp.revision, stamp.version
	state.SourceDigest, state.CompilerKey = g.digest, "second-key"
	s.states["kernel"] = state
	s.versionKeys[g] = "second-key"
	s.mu.Unlock()
	return g, k
}

func TestServiceSelectRevertMetadataAndRetainedOwner(t *testing.T) {
	original := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: original})
	registerService(t, s)
	originalState, _ := s.Status("kernel")
	g, second := addSelectionVersion(t, s)
	if _, err := s.SetCompileFlags("kernel", nil, "flags"); err != nil {
		t.Fatal(err)
	}
	state, err := s.Revert(context.Background(), "kernel", "original")
	if err != nil || state.Generation != 3 || state.Version != originalState.Version || state.ConfigRevision != 1 || state.SourceDigest != originalState.SourceDigest || state.CompilerKey != originalState.CompilerKey || state.Label != "original" {
		t.Fatalf("revert: %+v, %v", state, err)
	}
	state, err = s.Select(context.Background(), "kernel", g.version, "second")
	if err != nil || state.Generation != 4 || state.Version != g.version || state.ConfigRevision != 1 || state.SourceDigest != "second-source" || state.CompilerKey != "second-key" || state.Label != "second" {
		t.Fatalf("select: %+v, %v", state, err)
	}
	versions, err := s.Versions("kernel")
	if err != nil || len(versions) != 2 || !versions[0].Original || versions[0].Current || versions[1].Original || !versions[1].Current || versions[1].CompilerKey != "second-key" {
		t.Fatalf("versions: %+v, %v", versions, err)
	}
	versions[1].Flags[0] = "mutated"
	if g.flags[0] != "--fmad=false" || original.calls != 0 || second.calls != 0 || len(*leases) != 3 {
		t.Fatal("metadata aliases or selection unloads retained code")
	}
	for _, lease := range *leases {
		if lease.calls != 1 {
			t.Fatal("selection retains idle GPU lease")
		}
	}
	if err := s.Close(context.Background()); err != nil || original.calls != 1 || second.calls != 1 {
		t.Fatalf("close owners: %v, %d/%d", err, original.calls, second.calls)
	}
}

func TestServiceSelectPatchesAndRollsBackGraph(t *testing.T) {
	s, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	g, k := addSelectionVersion(t, s)
	e, n, stream := new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream)
	var installed driver.Function
	var fail driver.Function
	s.graphs.ops.patch = func(_ *cudagraph.Exec, _ *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		installed = fn // A failing patch may still have changed the node.
		if fn == fail {
			return errors.New("partially applied patch")
		}
		return nil
	}
	if _, err := s.graphs.bind("kernel", e, n, stream, driver.Dims(1, 1, 1), nil); err != nil {
		t.Fatal(err)
	}
	if state, err := s.Revert(context.Background(), "kernel", "patch original"); err != nil || state.Version != 1 || installed != 1 {
		t.Fatalf("graph revert: %+v, %v, installed=%d", state, err, installed)
	}
	before, _ := s.Status("kernel")
	fail = 2
	if state, err := s.Select(context.Background(), "kernel", g.version, "failed patch"); state != (State{}) || err == nil || installed != 1 || k.calls != 0 {
		t.Fatalf("graph rollback: %+v, %v, installed=%d, owner closes=%d", state, err, installed, k.calls)
	}
	if after, _ := s.Status("kernel"); after != before || g.pins.Load() != 0 {
		t.Fatal("failed patch changed selection or leaked temporary retained pin")
	}
	fail = 0
	if state, err := s.Select(context.Background(), "kernel", g.version, "retry"); err != nil || state.Version != g.version || installed != 2 {
		t.Fatalf("retry: %+v, %v, installed=%d", state, err, installed)
	}
	if err := s.Close(context.Background()); err != nil || k.calls != 1 {
		t.Fatalf("close graph pins before module: %v, %d", err, k.calls)
	}
}

func TestServiceSelectRefusalAndFailedPublication(t *testing.T) {
	s, leases := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	for _, tc := range []struct {
		ctx     context.Context
		name    string
		version uint64
	}{{nil, "kernel", 1}, {context.Background(), "missing", 1}, {context.Background(), "kernel", 0}, {context.Background(), "kernel", 99}} {
		if _, err := s.Select(tc.ctx, tc.name, tc.version, "bad"); err == nil {
			t.Fatal("invalid selection admitted")
		}
	}
	if len(*leases) != 1 {
		t.Fatal("invalid selections acquire GPU lease")
	}
	s.table.mu.Lock()
	s.table.next = math.MaxUint64
	s.table.mu.Unlock()
	before, _ := s.Status("kernel")
	if state, err := s.Select(context.Background(), "kernel", 1, "overflow"); state != (State{}) || err == nil {
		t.Fatalf("overflow admitted: %+v, %v", state, err)
	}
	if after, _ := s.Status("kernel"); after != before || s.graphs.patch != nil || (*leases)[1].calls != 1 {
		t.Fatal("failed table selection changes state or retains transaction/lease")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Versions("kernel"); err == nil {
		t.Fatal("closed versions admitted")
	}
}
