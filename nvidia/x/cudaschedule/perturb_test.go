package cudaschedule_test

import (
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

const PERTURB_PTX = `.version 8.7
.target sm_90a
.address_size 64

.visible .entry race(
    .param .u64 data
)
{
    .reg .pred %p0;
    .reg .b32 %r0;
    .reg .b64 %rd1;
L_loop:
    atom.global.add.u32 %r0, [%rd1], 1;
    red.global.add.u32 [%rd1], %r0;
    ld.global.acquire.u32 %r0, [%rd1];
    st.shared.release.u32 [%rd1], %r0;
    bar.sync 0;
    membar.gl;
    shfl.sync.idx.b32 %r0, %r0, 0, 31;
    @%p0 bra L_loop;
    ret;
}
`

func TestPerturb_SeedDeterministic(t *testing.T) {
	opts := cudaschedule.Options{
		Seed:       42,
		Intensity:  1.0,
		MaxSleepNs: 64000,
	}

	baseline, err := cudaschedule.Perturb(PERTURB_PTX, opts)
	if err != nil {
		t.Fatalf("initial Perturb failed: %v", err)
	}
	if !strings.Contains(baseline.PTX, "nanosleep.u32") {
		t.Fatalf("expected injected nanosleep.u32, got none")
	}

	// Test across varying GOMAXPROCS and concurrent executions
	for _, procs := range []int{1, 2, 4, 8} {
		prev := runtime.GOMAXPROCS(procs)
		defer runtime.GOMAXPROCS(prev)

		const workers = 8
		var wg sync.WaitGroup
		results := make([]cudaschedule.Rewrite, workers)
		errors := make([]error, workers)

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				res, err := cudaschedule.Perturb(PERTURB_PTX, opts)
				results[idx] = res
				errors[idx] = err
			}(w)
		}
		wg.Wait()

		for w := 0; w < workers; w++ {
			if errors[w] != nil {
				t.Fatalf("procs=%d, worker=%d failed: %v", procs, w, errors[w])
			}
			if results[w].PTX != baseline.PTX {
				t.Fatalf("procs=%d, worker=%d PTX output differs from baseline", procs, w)
			}
			if results[w].Report.SitesInjected != baseline.Report.SitesInjected {
				t.Fatalf("procs=%d, worker=%d sites injected mismatch", procs, w)
			}
		}
	}
}

func TestPerturb_EmptyDrawReported(t *testing.T) {
	opts := cudaschedule.Options{
		Seed:      12345,
		Intensity: 0.0, // Guaranteed 0 sites selected
	}

	rewrite, err := cudaschedule.Perturb(PERTURB_PTX, opts)
	if err != nil {
		t.Fatalf("Perturb failed: %v", err)
	}

	if rewrite.Report.SitesInjected != 0 {
		t.Fatalf("SitesInjected = %d, want 0", rewrite.Report.SitesInjected)
	}
	if rewrite.PTX != PERTURB_PTX {
		t.Fatalf("empty draw did not return exact original PTX text")
	}
	if strings.Contains(rewrite.PTX, "nanosleep.u32") {
		t.Fatalf("empty draw injected nanosleep.u32 into output")
	}
}

func TestControl_SubstrateSleepNotCounted(t *testing.T) {
	// PTX emitted by cudapipeline.EmitWaitLoop containing nanosleep in spin-wait
	const pipelinePTX = `.version 8.7
.target sm_90
.address_size 64

.visible .entry spin_wait()
{
    .reg .pred %p;
    .reg .b64 %rd1;
wait_loop:
    mbarrier.try_wait.parity.shared::cta.b64 %p, [%rd1], 0;
    @!%p nanosleep.u32 20;
    @!%p bra wait_loop;
    ret;
}
`
	sites, err := cudaschedule.Analyze(pipelinePTX)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	for _, s := range sites {
		if strings.Contains(s.Head, "nanosleep") || strings.Contains(s.Text, "nanosleep") {
			t.Fatalf("analyzer classified pre-existing substrate nanosleep as a schedule site: %+v", s)
		}
	}

	// Exactly 2 sites expected: mbarrier (Barrier) and bra back-edge (Backedge)
	if len(sites) != 2 {
		t.Fatalf("Analyze found %d sites, want 2 (Barrier and Backedge)", len(sites))
	}
	if sites[0].Kind != cudaschedule.SiteBarrier {
		t.Errorf("site 0 kind = %v, want Barrier", sites[0].Kind)
	}
	if sites[1].Kind != cudaschedule.SiteBackedge {
		t.Errorf("site 1 kind = %v, want Backedge", sites[1].Kind)
	}
}

func TestControl_DeterminismFalsifiable(t *testing.T) {
	optsA := cudaschedule.Options{
		Seed:       100,
		Intensity:  1.0,
		MaxSleepNs: 64000,
	}
	optsB := cudaschedule.Options{
		Seed:       101, // Perturbed by one
		Intensity:  1.0,
		MaxSleepNs: 64000,
	}

	rewriteA, err := cudaschedule.Perturb(PERTURB_PTX, optsA)
	if err != nil {
		t.Fatalf("Perturb A failed: %v", err)
	}

	rewriteB, err := cudaschedule.Perturb(PERTURB_PTX, optsB)
	if err != nil {
		t.Fatalf("Perturb B failed: %v", err)
	}

	bodyA := rewriteA.PTX[strings.Index(rewriteA.PTX, "\n")+1:]
	bodyB := rewriteB.PTX[strings.Index(rewriteB.PTX, "\n")+1:]
	if bodyA == bodyB {
		t.Fatalf("perturbation with different seeds (100 vs 101) produced identical injected code; seed was ignored")
	}
}

func TestPerturb_FocusMatching(t *testing.T) {
	opts := cudaschedule.Options{
		Seed:       7,
		Intensity:  1.0,
		MaxSleepNs: 64000,
		Focus:      "atom.global",
	}

	rewrite, err := cudaschedule.Perturb(PERTURB_PTX, opts)
	if err != nil {
		t.Fatalf("Perturb with focus failed: %v", err)
	}

	foundFocused := false
	for _, dec := range rewrite.Report.Decisions {
		if strings.Contains(dec.Site.Head, "atom.global") {
			foundFocused = true
			if dec.BeforeNs == 0 && dec.AfterNs == 0 {
				t.Errorf("focused site %s was not injected", dec.Site.Head)
			}
		}
	}
	if !foundFocused {
		t.Fatalf("focused site atom.global not found in decisions")
	}
}
