// Package cudatrial runs interleaved, provenance-bearing multi-arm evaluation
// trials across rounds.
//
// In hardware benchmarking and kernel optimization, comparing sequential blocks
// (e.g. 100 runs of Arm A followed by 100 runs of Arm B) is vulnerable to thermal
// throttling, power-state transitions, and memory fragmentation drift. Interleaving
// arms in round-robin order balances background conditions across all candidates.
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
package cudatrial
