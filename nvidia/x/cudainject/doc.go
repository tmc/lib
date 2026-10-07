// Package cudainject builds the CUPTI injection agent and reads its artifacts.
// Building requires Linux, a C11 compiler, and a CUDA toolkit with CUPTI 12.8
// or newer. Reading artifacts does not require CUDA or a GPU.
//
// WireVersion describes injection records, not the separate capture manifest
// version checked by ReadCapture. Decoder.Stats reports missing sequence
// records and cumulative producer drops; an empty read is not evidence of loss
// free execution. Record identities separate sessions, processes and agent
// generations. Correlation IDs are meaningful only within that identity.
//
// The existing agent enables its periodic flush worker when the injected
// process sets CUDAPROF_FLUSH_INTERVAL_MS to a positive integer. The default is
// exit draining. Periodic flushing does not guarantee that every provider
// produces data or that a killed process leaves a complete journal.
package cudainject
