// Package cudatrial runs interleaved, provenance-bearing multi-arm evaluation
// trials across rounds.
//
// In hardware benchmarking and kernel optimization, comparing sequential blocks
// (e.g. 100 runs of Arm A followed by 100 runs of Arm B) is vulnerable to thermal
// throttling, power-state transitions, and memory fragmentation drift. Interleaving
// arms in ABBA order balances background conditions across all candidates.
//
// cudatrial also enforces non-additive latency decomposition (WallNs, DeviceNs,
// QueueNs, and GpuIntervalUnionNs) and explicitly bars naive duration summation
// across distinct timing channels. Optional hardware lock verification ensures
// trials have exclusive GPU access.
//
// Basic usage:
//
//	arms := []cudatrial.Arm{
//		{
//			Name: "baseline",
//			Run: func(ctx context.Context) (cudatrial.Receipt, error) {
//				// execute baseline kernel and hash outputs
//				return cudatrial.Receipt{Complete: true, Digest: "a1b2"}, nil
//			},
//		},
//		{
//			Name: "candidate",
//			Run: func(ctx context.Context) (cudatrial.Receipt, error) {
//				// execute candidate kernel and hash outputs
//				return cudatrial.Receipt{Complete: true, Digest: "a1b2"}, nil
//			},
//		},
//	}
//
//	receipts, err := cudatrial.Interleave(ctx, arms, 5)
//
// Each receipt also carries the conditions the arm ran under, observed at the
// arm boundaries into Receipt.Before and Receipt.After. Conditions records host
// available memory (MemAvailable, not MemFree), host page cache, retained
// artifact bytes, and a device memory pool readout, distinguishing a measured
// zero from a value that could not be read the same way Timings does.
// SameArmDisagreements pairs each same-arm digest disagreement with the
// conditions that differed between the two rounds, so that a result produced
// under a neighbouring arm's memory footprint is not charged to the code:
//
//	for _, d := range cudatrial.SameArmDisagreements(receipts, 1<<30) {
//		if d.Confounded() {
//			log.Printf("not attributable to code: %v", d)
//		}
//	}
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// RaceCorpus evaluates cuHadron race cases with per-case access models. It
// reports unsupported hardware as pending and requires a passing lane 52
// exec-control result for the same configuration before computing a rate.
// Cases derive from NVlabs/cuHadron commit 0a6c6d5 (Apache-2.0).
package cudatrial
