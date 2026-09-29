package cudaschedule_test

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
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
	// Wrap the real emitter output in a parseable module.
	pipelinePTX := `.version 8.7
.target sm_90
.address_size 64

.visible .entry spin_wait()
{
    .reg .pred %p;
    .reg .b64 %rd1;
` + cudapipeline.EmitWaitLoop("%rd1", 0, "wait") + `
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

	// The emitter contributes the mbarrier and branch sites, but not nanosleep.
	if len(sites) != 2 {
		t.Fatalf("Analyze found %d sites, want 2 (Barrier and Backedge)", len(sites))
	}
	if sites[0].Kind != cudaschedule.SiteBarrier {
		t.Errorf("site 0 kind = %v, want Barrier", sites[0].Kind)
	}
	if sites[1].Kind != cudaschedule.SiteBackedge {
		t.Errorf("site 1 kind = %v, want Backedge", sites[1].Kind)
	}

	rewrite, err := cudaschedule.Perturb(pipelinePTX, cudaschedule.Options{Seed: 77, Intensity: 1})
	if err != nil {
		t.Fatal(err)
	}
	marked := 0
	unmarked := 0
	for _, line := range strings.Split(rewrite.PTX, "\n") {
		if !strings.Contains(line, "nanosleep.u32") {
			continue
		}
		if strings.Contains(line, "// ptx_schedule") {
			marked++
		} else {
			unmarked++
		}
	}
	if marked != rewrite.Report.SitesInjected || unmarked != 1 {
		t.Fatalf("emitted PTX has %d marked and %d unmarked sleeps, report injects %d; want one substrate sleep\n%s", marked, unmarked, rewrite.Report.SitesInjected, rewrite.PTX)
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

func TestPerturb_TouchRatesAndSleepBounds(t *testing.T) {
	if cudaschedule.DefaultMaxSleepNs != 64000 {
		t.Fatalf("DefaultMaxSleepNs = %d, want 64000", cudaschedule.DefaultMaxSleepNs)
	}
	const twoSites = `.version 8.7
.target sm_90
.address_size 64
.visible .entry rates(){
    atom.global.add.u32 %r, [%rd], 1;
    bar.sync 0;
    ret;
}
`
	const draws = 2000
	within := func(got, want float64) {
		t.Helper()
		if delta := got - want; delta < -0.04 || delta > 0.04 {
			t.Fatalf("touch rate %.3f, want about %.3f", got, want)
		}
	}
	var plain, focused, unfocused int
	for seed := uint64(0); seed < draws; seed++ {
		for mode := 0; mode < 2; mode++ {
			opts := cudaschedule.Options{Seed: seed, Intensity: 0.4, MaxSleepNs: 50000}
			if mode == 1 {
				opts.Focus = "atom.global"
			}
			r, err := cudaschedule.Perturb(twoSites, opts)
			if err != nil {
				t.Fatal(err)
			}
			for _, decision := range r.Report.Decisions {
				injected := decision.BeforeNs > 0 || decision.AfterNs > 0
				if injected && max(decision.BeforeNs, decision.AfterNs) > 20000 {
					t.Fatalf("sleep exceeds scaled bound: %+v", decision)
				}
				if mode == 0 && injected {
					plain++
				}
				if mode == 1 {
					if strings.Contains(decision.Site.Head, "atom.global") && injected {
						focused++
					}
					if !strings.Contains(decision.Site.Head, "atom.global") && injected {
						unfocused++
					}
				}
			}
		}
	}
	within(float64(plain)/float64(draws*2), 0.75*0.4)
	within(float64(focused)/draws, 0.95*0.4)
	within(float64(unfocused)/draws, 0.15*0.4)

	var sawLargeDefault bool
	for seed := uint64(0); seed < 100; seed++ {
		r, err := cudaschedule.Perturb(twoSites, cudaschedule.Options{Seed: seed, Intensity: 1})
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range r.Report.Decisions {
			for _, ns := range []uint32{d.BeforeNs, d.AfterNs} {
				if ns > 64000 {
					t.Fatalf("default sleep %d exceeds 64000", ns)
				}
				if ns > 2000 {
					sawLargeDefault = true
				}
			}
		}
	}
	if !sawLargeDefault {
		t.Fatal(fmt.Sprintf("zero MaxSleepNs did not use the %d ns default", cudaschedule.DefaultMaxSleepNs))
	}
}

func TestPerturb_NonTrivialEmptyDrawIsExactNoOp(t *testing.T) {
	const oneSite = `.version 8.7
.target sm_90
.address_size 64
.visible .entry only(){ atom.global.add.u32 %r, [%rd], 1; ret; }
`
	for seed := uint64(0); seed < 10000; seed++ {
		r, err := cudaschedule.Perturb(oneSite, cudaschedule.Options{Seed: seed, Intensity: 0.5})
		if err != nil {
			t.Fatal(err)
		}
		if r.Report.SitesInjected == 0 {
			if r.PTX != oneSite {
				t.Fatalf("empty draw changed PTX for seed %d", seed)
			}
			return
		}
	}
	t.Fatal("no non-trivial empty draw found in 10000 seeds")
}

func TestPerturb_InlineAsmInjectionStaysOutsideMarkers(t *testing.T) {
	const inline = `.version 8.7
.target sm_90
.address_size 64
.visible .entry inline(){
// begin inline asm
    atom.global.add.u32 %r, [%rd], 1;
    atom.global.add.u32 %r, [%rd], 1;
// end inline asm
    ret;
}
`
	for seed := uint64(0); seed < 256; seed++ {
		r, err := cudaschedule.Perturb(inline, cudaschedule.Options{Seed: seed, Intensity: 2})
		if err != nil {
			t.Fatal(err)
		}
		if r.Report.SitesTotal != 2 || r.Report.SitesInjected != 2 {
			t.Fatalf("seed %d sites = %d/%d, want 2/2", seed, r.Report.SitesInjected, r.Report.SitesTotal)
		}
		begin := strings.Index(r.PTX, "// begin inline asm")
		end := strings.Index(r.PTX, "// end inline asm")
		for offset := 0; ; {
			i := strings.Index(r.PTX[offset:], "nanosleep.u32")
			if i < 0 {
				break
			}
			i += offset
			if i > begin && i < end {
				t.Fatalf("seed %d injected nanosleep inside inline asm markers:\n%s", seed, r.PTX)
			}
			offset = i + len("nanosleep.u32")
		}
	}
}
