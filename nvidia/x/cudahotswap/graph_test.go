package cudahotswap

import (
	"context"
	"errors"
	"math"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func graphFixture(t *testing.T, count int) (*graphRegistry, *generation, []*cudagraph.Node, *cudagraph.Exec, *cuda.Stream) {
	t.Helper()
	table := new(generationTable)
	g, _ := testGeneration(t, 1, "")
	if _, err := table.publish("kernel", g, 0); err != nil {
		t.Fatal(err)
	}
	r := newGraphRegistry(table)
	r.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error { return nil }
	r.ops.launch = func(*cudagraph.Exec, *cuda.Stream) error { return nil }
	r.ops.wait = func(context.Context, *cuda.Stream) error { return nil }
	e, s := new(cudagraph.Exec), new(cuda.Stream)
	var nodes []*cudagraph.Node
	for i := 0; i < count; i++ {
		n := new(cudagraph.Node)
		if _, err := r.bind("kernel", e, n, s, driver.Dims(1, 1, 1), nil); err != nil {
			t.Fatal(err)
		}
		nodes = append(nodes, n)
	}
	t.Cleanup(func() {
		if err := r.close(); err != nil {
			t.Error(err)
		}
		if err := table.close(); err != nil {
			t.Error(err)
		}
	})
	return r, g, nodes, e, s
}

func TestGraphPatchPublicationAndPins(t *testing.T) {
	r, old, _, e, _ := graphFixture(t, 2)
	candidate, owner := testGeneration(t, 2, "")
	patched := 0
	r.ops.patch = func(_ *cudagraph.Exec, _ *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		if fn != candidate.function {
			t.Fatal("unexpected candidate", fn)
		}
		patched++
		return nil
	}
	tx, err := r.prepare("kernel", candidate)
	if err != nil || patched != 2 || old.pins.Load() != 2 || candidate.pins.Load() != 0 {
		t.Fatal(tx, err, patched, old.pins.Load(), candidate.pins.Load())
	}
	if p, err := r.launch(e); err == nil || p != nil {
		t.Fatal("launch allowed before table publication")
	}
	if _, err := r.table.publish("kernel", candidate, old.version); err != nil {
		t.Fatal("prepublication candidate pin violates table contract", err)
	}
	tx.published()
	if err := tx.Close(); err != nil {
		t.Fatal(err)
	}
	if err := tx.Close(); err != nil || owner.count() != 0 || candidate.pins.Load() != 2 || old.pins.Load() != 0 {
		t.Fatal("published candidate unloaded or pins not transferred", err)
	}
}

func TestGraphPatchRollbackRetainsBothOwners(t *testing.T) {
	r, old, nodes, e, _ := graphFixture(t, 2)
	candidate, owner := testGeneration(t, 2, "")
	applyErr, rollbackErr := errors.New("late patch failure"), errors.New("rollback failure")
	state := map[*cudagraph.Node]driver.Function{nodes[0]: old.function, nodes[1]: old.function}
	failedRollback := true
	r.ops.patch = func(_ *cudagraph.Exec, n *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		state[n] = fn // Driver side effect may precede the returned error.
		if fn == candidate.function && n == nodes[1] {
			return applyErr
		}
		if fn == old.function && failedRollback {
			return rollbackErr
		}
		return nil
	}
	tx, err := r.prepare("kernel", candidate)
	if tx == nil || !errors.Is(err, applyErr) {
		t.Fatal("partial transaction owner lost", tx, err)
	}
	if err := tx.Close(); !errors.Is(err, rollbackErr) || owner.count() != 0 || old.pins.Load() != 2 {
		t.Fatal("failed rollback dropped ownership", err)
	}
	if _, err := r.launch(e); err == nil {
		t.Fatal("launch allowed after failed rollback")
	}
	failedRollback = false
	if err := tx.Close(); err != nil {
		t.Fatal(err)
	}
	if state[nodes[0]] != old.function || state[nodes[1]] != old.function || owner.count() != 1 {
		t.Fatal("rollback missed failing node or candidate unload", state, owner.count())
	}
}

func TestGraphAbortUnloadRetry(t *testing.T) {
	r, _, _, _, _ := graphFixture(t, 1)
	g, owner := testGeneration(t, 2, "")
	tx, err := r.prepare("kernel", g)
	if err != nil {
		t.Fatal(err)
	}
	owner.fail = true
	if err := tx.Close(); err == nil || r.patch != tx {
		t.Fatal("failed candidate cleanup lost", err)
	}
	owner.fail = false
	if err := tx.Close(); err != nil || owner.count() != 1 {
		t.Fatal(err)
	}
}

func TestGraphPendingCancellationAndDrainRetry(t *testing.T) {
	r, g, _, e, stream := graphFixture(t, 2)
	launchErr, drainErr := errors.New("enqueue context restore"), errors.New("drain failure")
	r.ops.launch = func(_ *cudagraph.Exec, s *cuda.Stream) error {
		if s != stream {
			t.Fatal("owning stream replaced")
		}
		return launchErr
	}
	failed, waits := true, 0
	r.ops.wait = func(_ context.Context, s *cuda.Stream) error {
		waits++
		if s != stream {
			t.Fatal("different completion stream")
		}
		if failed {
			return drainErr
		}
		return nil
	}
	p, err := r.launch(e)
	if p == nil || !errors.Is(err, launchErr) || g.pins.Load() != 4 {
		t.Fatal("partial enqueue lost pins", err, g.pins.Load())
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.wait(ctx); !errors.Is(err, context.Canceled) || waits != 0 || g.pins.Load() != 4 {
		t.Fatal("observation cancellation released work", err)
	}
	if err := p.wait(context.Background()); !errors.Is(err, drainErr) || g.pins.Load() != 4 {
		t.Fatal("failed drain released work", err)
	}
	if err := r.close(); err == nil {
		t.Fatal("closed pending registry")
	}
	if _, err := r.prepare("kernel", &generation{}); err == nil {
		t.Fatal("patched pending executable")
	}
	failed = false
	if err := p.wait(context.Background()); err != nil || g.pins.Load() != 2 {
		t.Fatal("successful drain failed to release extra pins", err)
	}
	if err := p.wait(context.Background()); err != nil || waits != 2 {
		t.Fatal("completed drain repeated", err, waits)
	}
}

func TestGraphBindCopiesAndFailedPatchRetainsPin(t *testing.T) {
	r, g, _, e, s := graphFixture(t, 0)
	args := []any{uint32(3)}
	patchErr := errors.New("late initial patch failure")
	r.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error { return patchErr }
	b, err := r.bind("kernel", e, new(cudagraph.Node), s, driver.Dims(1, 1, 1), args)
	args[0] = uint32(4)
	if b == nil || !errors.Is(err, patchErr) || b.args[0] != uint32(3) || g.pins.Load() != 1 {
		t.Fatal("initial partial patch owner or arguments lost", b, err)
	}
	if _, err := r.launch(e); err == nil {
		t.Fatal("initial failed binding executed")
	}
}

func TestGraphMutationWaitOrdering(t *testing.T) {
	r, _, _, e, _ := graphFixture(t, 1)
	entered, release := make(chan struct{}), make(chan struct{})
	var completed atomic.Bool
	r.ops.wait = func(context.Context, *cuda.Stream) error {
		close(entered)
		<-release
		completed.Store(true)
		return nil
	}
	p, err := r.launch(e)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := p.wait(context.Background()); err != nil {
			t.Error(err)
		}
	}()
	<-entered
	// Actual work remains pinned throughout the blocked completion callback.
	if p.pins[0].gen.pins.Load() != 2 {
		t.Fatal("late completion does not retain generation")
	}
	r.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error {
		if !completed.Load() {
			t.Error("mutation overlaps actual completion")
		}
		return nil
	}
	candidate, _ := testGeneration(t, 2, "")
	started := make(chan struct{})
	transaction := make(chan *graphPatch, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		close(started)
		tx, err := r.prepare("kernel", candidate)
		if err != nil {
			t.Error(err)
		}
		transaction <- tx
	}()
	<-started
	close(release)
	wg.Wait()
	if tx := <-transaction; tx == nil {
		t.Fatal("mutation lost owner")
	} else if err := tx.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestGraphOldWaitCannotForgetNewLaunch(t *testing.T) {
	r, _, _, e, _ := graphFixture(t, 1)
	first, err := r.launch(e)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	second, err := r.launch(e)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if r.pending[e] != second {
		t.Fatal("old observation removed new pending owner")
	}
	if err := second.wait(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestGraphCommitCleanupRetryCannotUnloadPublishedCandidate(t *testing.T) {
	r, old, _, _, _ := graphFixture(t, 1)
	candidate, candidateOwner := testGeneration(t, 2, "")
	tx, err := r.prepare("kernel", candidate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.table.publish("kernel", candidate, old.version); err != nil {
		t.Fatal(err)
	}
	tx.published()
	oldOwner := old.owner.(*generationOwner)
	r.table.mu.Lock()
	old.retired = true
	oldOwner.fail = true
	r.table.mu.Unlock()
	if err := tx.Close(); err == nil || candidateOwner.count() != 0 || candidate.pins.Load() != 1 || r.patch != tx {
		t.Fatal("publication owner lost on cleanup error", err)
	}
	oldOwner.fail = false
	if err := tx.Close(); err != nil || oldOwner.count() != 1 || candidateOwner.count() != 0 || candidate.pins.Load() != 1 {
		t.Fatal("retry repeated transfer or lost ownership", err)
	}
}

func TestGraphNilTableRefusesBeforeAdoption(t *testing.T) {
	r := newGraphRegistry(nil)
	g, owner := testGeneration(t, 2, "")
	for _, prepare := range []func(string, *generation) (*graphPatch, error){r.prepare, r.prepareRetained} {
		if tx, err := prepare("kernel", g); err == nil || tx != nil || r.patch != nil || owner.count() != 0 {
			t.Fatal("invalid registry adopted owner", tx, err)
		}
	}
	if p, err := r.launch(new(cudagraph.Exec)); err == nil || p != nil {
		t.Fatal("invalid registry admitted launch", p, err)
	}
}

func TestGraphRetainedSelectAndRevert(t *testing.T) {
	r, a, nodes, e, _ := graphFixture(t, 1)
	external, _, err := r.table.pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	b, bOwner := testGeneration(t, 2, "")
	actual := map[*cudagraph.Node]driver.Function{nodes[0]: a.function}
	r.ops.patch = func(_ *cudagraph.Exec, n *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		actual[n] = fn
		return nil
	}
	tx, err := r.prepare("kernel", b)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.table.publish("kernel", b, a.version); err != nil {
		t.Fatal(err)
	}
	tx.published()
	if err := tx.Close(); err != nil {
		t.Fatal(err)
	}
	for i, target := range []*generation{a, b, a} {
		tx, err = r.prepareRetained("kernel", target)
		if err != nil {
			t.Fatal("existing pins or retained target rejected", err)
		}
		if i == 2 {
			_, err = r.table.revert("kernel")
		} else {
			_, err = r.table.selectVersion("kernel", target.version)
		}
		if err != nil {
			t.Fatal(err)
		}
		tx.published()
		if err := tx.Close(); err != nil {
			t.Fatal(err)
		}
		if err := tx.Close(); err != nil {
			t.Fatal("idempotent transfer", err)
		}
		if actual[nodes[0]] != target.function || r.bindings[0].pin.gen != target || bOwner.count() != 0 {
			t.Fatal("graph/table selection differs or retained owner closed")
		}
		p, err := r.launch(e)
		if err != nil {
			t.Fatal(err)
		}
		if p.pins[0].gen != target {
			t.Fatal("launch pins stale selection")
		}
		if err := p.wait(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if a.pins.Load() != 2 || b.pins.Load() != 0 {
		t.Fatal("retained transaction pin leaked or released twice", a.pins.Load(), b.pins.Load())
	}
	if err := external.release(); err != nil {
		t.Fatal(err)
	}
}

func TestGraphRetainedSelectionFailureRollbackRetry(t *testing.T) {
	r, a, nodes, e, _ := graphFixture(t, 1)
	b, bOwner := testGeneration(t, 2, "")
	if _, err := r.table.publish("kernel", b, a.version); err != nil {
		t.Fatal(err)
	}
	if _, err := r.table.selectVersion("kernel", a.version); err != nil {
		t.Fatal(err)
	}
	// Graph still selects A: prepare the retained B as a Select transaction.
	actual := a.function
	rollbackFailure := false
	want := errors.New("rollback failed")
	r.ops.patch = func(_ *cudagraph.Exec, n *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		if n != nodes[0] {
			t.Fatal("wrong graph node")
		}
		actual = fn
		if fn == a.function && rollbackFailure {
			return want
		}
		return nil
	}
	tx, err := r.prepareRetained("kernel", b)
	if err != nil {
		t.Fatal(err)
	}
	r.table.mu.Lock()
	r.table.next = math.MaxUint64
	r.table.mu.Unlock()
	if _, err := r.table.selectVersion("kernel", b.version); err == nil {
		t.Fatal("exhausted table selected target")
	}
	rollbackFailure = true
	if err := tx.Close(); !errors.Is(err, want) || bOwner.count() != 0 || b.pins.Load() != 1 || a.pins.Load() != 1 {
		t.Fatal("failed rollback lost retained owners", err)
	}
	if _, err := r.launch(e); err == nil {
		t.Fatal("failed rollback allowed launch")
	}
	rollbackFailure = false
	if err := tx.Close(); err != nil {
		t.Fatal(err)
	}
	if actual != a.function || bOwner.count() != 0 || b.pins.Load() != 0 || a.pins.Load() != 1 {
		t.Fatal("retained abort unloads module or leaks pin", actual, b.pins.Load(), a.pins.Load())
	}
}

func TestGraphRetainedIdentityRefusals(t *testing.T) {
	r, a, _, _, _ := graphFixture(t, 1)
	copy := &generation{version: a.version, digest: a.digest, signature: a.signature, function: a.function, owner: a.owner}
	for _, tt := range []struct {
		name   string
		target *generation
	}{
		{"kernel", copy}, {"other", a}, {"kernel", nil},
	} {
		if tx, err := r.prepareRetained(tt.name, tt.target); err == nil || tx != nil {
			t.Fatal("unowned retained identity adopted", tx, err)
		}
	}
	r.table.mu.Lock()
	a.retired = true
	r.table.mu.Unlock()
	if tx, err := r.prepareRetained("kernel", a); err == nil || tx != nil {
		t.Fatal("retired identity adopted", tx, err)
	}
	r.table.mu.Lock()
	a.retired = false
	r.table.mu.Unlock()
}

func TestGraphClosedTableRefusesBeforeWork(t *testing.T) {
	r, a, _, e, _ := graphFixture(t, 1)
	if err := r.table.close(); !errors.Is(err, errGenerationPinned) {
		t.Fatal("expected retained binding", err)
	}
	g, owner := testGeneration(t, 2, "")
	if tx, err := r.prepare("kernel", g); err == nil || tx != nil || owner.count() != 0 {
		t.Fatal("closed table adopted candidate", tx, err)
	}
	if tx, err := r.prepareRetained("kernel", a); err == nil || tx != nil {
		t.Fatal("closed table borrowed target", tx, err)
	}
	if p, err := r.launch(e); err == nil || p != nil {
		t.Fatal("closed table launched", p, err)
	}
}
