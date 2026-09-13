// Package cudapipeline provides static verification and transaction-barrier analysis for asynchronous tensor copy pipelines.
//
// Asynchronous copy pipelines on NVIDIA Hopper and Blackwell architectures coordinate
// hardware Tensor Memory Accelerator (TMA) bulk transfers (cp.async.bulk.tensor)
// using transaction barriers (mbarrier).
//
// ValidatePTX audits PTX modules to statically verify pipeline protocols:
//   - Barrier initialization: ensures mbarriers are initialized before arrival or transfer.
//   - Transaction balance: verifies transaction byte counts expected via mbarrier.expect_tx
//     and mbarrier.arrive.expect_tx match bytes completed by cp.async.bulk.tensor copies.
//   - Arrival tokens: verifies thread arrivals precede wait operations.
//   - Phase parity: verifies alternating phase parity (0 and 1) across pipeline stages in
//     mbarrier.try_wait.parity and mbarrier.test_wait.parity.
//
// # Hardware Grounding
//
// Grounded in sm_121a (Blackwell GB10) where TMA (cp.async.bulk.tensor.2d),
// mbarrier, and multimem assemble and execute. WGMMA (sm_90) and tcgen05 (sm_100) paths
// are deferred to testbeds with corresponding hardware support.
package cudapipeline
