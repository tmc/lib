// Package cudaschedule implements deterministic schedule-perturbation campaigns
// for PTX assembly.
//
// A kernel that races only under one interleaving passes until it does not.
// This package finds points in a PTX module where another thread's progress is
// observable, injects seeded nanosleep.u32 delays at a drawn subset of them,
// and sweeps seeds to classify execution outcomes into an eight-way verdict taxonomy.
//
// # Site Taxonomy
//
// [Analyze] scans a PTX module and classifies schedule-sensitive sites into ten kinds:
//   - [SiteAtomic]: atomic read-modify-write instructions (atom.*)
//   - [SiteReduction]: memory reductions with no returned value (red.*)
//   - [SiteBarrier]: CTA, warp, and mbarrier synchronization (bar.*, barrier.*, mbarrier.*)
//   - [SiteFence]: memory ordering and fences (fence.*, membar.*)
//   - [SiteOrderedMemory]: memory loads and stores with explicit ordering qualifiers
//     (.volatile, .acquire, .release, .relaxed, .acq_rel, .mmio), excluding non-coherent loads
//   - [SiteWarpCollective]: warp-wide communication (shfl.*, vote.*, match.*, elect.*, activemask, redux.*)
//   - [SiteAsyncProxy]: asynchronous proxy pipeline operations (cp.async.*, cp.reduce.async.*,
//     st.async.*, tensormap.cp_fenceproxy.*, wgmma.*, tcgen05.*, clusterlaunchcontrol.*)
//   - [SiteGridDependency]: programmatic dependent launch boundaries (griddepcontrol.launch_dependents, griddepcontrol.wait)
//   - [SiteTensorMapMutation]: generic-proxy tensor map descriptor mutations (tensormap.replace.*)
//   - [SiteBackedge]: intraprocedural branches (bra, brx) to the same or an earlier basic block
//
// The site taxonomy is a closed set: unsupported ordering-sensitive families such as multimem.*
// return an error, never an unclassified site.
//
// # Verdict Taxonomy
//
// [Classify] maps execution outcomes into eight verdicts:
//   - [VerdictPass]: the perturbed run behaved like the baseline
//   - [VerdictSkipped]: the target declined execution via a recognized skip marker
//   - [VerdictHang]: the watchdog timer fired
//   - [VerdictCrash]: the process exited with a non-zero status or signal
//   - [VerdictMismatch]: the output contained an explicit correctness failure marker
//   - [VerdictOutputChanged]: stdout differed from baseline without an explicit error marker (opt-in)
//   - [VerdictGpuWedged]: a timeout occurred and a pristine re-run also failed
//   - [VerdictHarnessError]: the harness failed to perturb or execute the variant
//
// Exactly five verdicts are findings: [VerdictMismatch], [VerdictOutputChanged],
// [VerdictHang], [VerdictCrash], and [VerdictGpuWedged].
package cudaschedule
