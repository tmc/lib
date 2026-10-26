// Package cudaready holds ready.cuh, epoch-tagged dataflow counters for
// persistent and cooperative CUDA kernels: a stage signals a counter when
// its writes are visible, and a later stage waits for the count it needs
// instead of a grid-wide barrier.
//
// A counter is a uint64 word, epoch<<32 | count, zeroed once and never
// reset: a word from an earlier epoch counts as zero. ready::signal fences,
// then raises the word to epoch<<32 with atomicMax and adds its count with
// atomicAdd; ready::wait polls with backoff, then fences to acquire.
//
// The counters are safe only among co-resident blocks, as in a cooperative
// launch or a grid no larger than the device holds at once: a waiting block
// spins, so a producer that never becomes resident deadlocks it.
//
// Memory ordering is verified only on hardware. The host race test checks
// the counting with threads standing in for blocks, and it passes with
// either fence removed. TestDeviceLitmus, a message-passing test in which
// consumers must read every word producers wrote before signaling, needs
// nvcc and a CUDA device and skips without them.
//
// Include the header from a source tree, or write [Header] beside a
// generated kernel.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaready

import _ "embed"

// Header is the text of ready.cuh.
//
//go:embed ready.cuh
var Header string
