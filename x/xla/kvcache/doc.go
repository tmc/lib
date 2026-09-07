// Package kvcache manages device-resident key/value history for causal attention.
//
// A cache borrows its engine and input buffers and owns its history and compiled
// programs. Close the cache before its engine and close every returned attention
// buffer. Calls, including Close, must be externally synchronized. Inputs must
// belong to the same engine and to the device reported by the initialized cache
// buffers. All batch members share one length and append the same chunk size.
//
// History updates are functional PJRT outputs, not donated or guaranteed in-place
// mutations. The cache retains a decode program and at most one prefill program.
// Reset discards logical history without downloading or clearing device arrays.
package kvcache
