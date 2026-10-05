// Package cudaschedule implements deterministic schedule-perturbation campaigns
// for PTX assembly.
//
// A kernel that races only under one interleaving passes until it does not.
// This package finds points in a PTX module where another thread's progress is
// observable, injects seeded nanosleep.u32 delays at a drawn subset of them,
// and sweeps seeds to classify execution outcomes into a nine-value verdict taxonomy.
//
// # Site Taxonomy
//
// [Analyze] scans a PTX module and classifies schedule-sensitive sites into ten kinds:
//   - [SiteAtomic]: atomic read-modify-write instructions (atom.*)
//   - [SiteReduction]: memory reductions with no returned value (red.*),
//     including red.async, which also completes an mbarrier transaction
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
// [Replay] applies an explicit subset of recorded injection decisions to
// pristine PTX. Each complete site record must match [Analyze], including
// source spans and backedge block indices. Duplicate or changed sites,
// delays above the original options' ceiling and sleeps after a backedge
// are errors. Replay retains decision order and derives injection counts;
// it does not redraw decisions from the seed. The full recorded list with
// the original options reproduces [Perturb] byte for byte, while an empty
// subset returns the pristine source.
//
// # Verdict Taxonomy
//
// [Classify] maps execution outcomes into nine verdicts:
//   - [VerdictPass]: the perturbed run behaved like the baseline
//   - [VerdictSkipped]: the target declined execution via a recognized skip marker
//   - [VerdictHang]: the watchdog timer fired
//   - [VerdictCrash]: the process exited with a non-zero status or signal
//   - [VerdictMismatch]: the output contained an explicit correctness failure marker
//   - [VerdictOutputChanged]: stdout differed from baseline without an explicit error marker (opt-in)
//   - [VerdictGpuWedged]: a timeout occurred and the pristine health check did
//     not pass; this includes a skipped pristine run
//   - [VerdictHarnessError]: the harness failed to perturb or execute the variant
//   - [VerdictEmptyDraw]: no site was selected, so no variant run occurred
//
// Exactly five verdicts are findings: [VerdictMismatch], [VerdictOutputChanged],
// [VerdictHang], [VerdictCrash], and [VerdictGpuWedged].
//
// [RunCampaign] records a SHA-256 digest for each supplied PTX input and requires
// a matching runner-reported loaded-PTX digest before classifying the result.
// Executables print CUDASCHEDULE_LOADED_PTX_SHA256: followed by the digest
// after loading the artifact; RunVariant reports RunRecord.PTXDigest directly.
// A missing receipt is HarnessError, not evidence that the variant ran.
// When an oracle is attached, its PID, patched-module count and instrumented
// access count must be nonzero; the default runner also checks the PID.
// Seed
// records retain the pristine source digest, launch geometry, input reference,
// sleep ceiling, and focus string. Optional [CampaignArms] repeat the sweep
// against a planted-bug input and a clean control; an undetected planted bug
// reports the number of seeds drawn. [CampaignReport.CalibrationValidated]
// stays false until a confirmed planted finding and a complete passing clean
// control establish calibration. Site counts are not device coverage.
// The default process runner may set [CampaignOptions.ArtifactDir] and
// [CampaignOptions.KeepArtifacts] to retain PTX inputs, process output, and a
// replay script for each run.
// GpuWedged findings are not confirmed with more variants because the pristine
// health check already failed on that device.
package cudaschedule
