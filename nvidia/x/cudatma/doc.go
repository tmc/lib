// Package cudatma describes checked tensor-map transfers and device-local
// multicast participation. It does not pack opaque vendor descriptors.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// AsyncCopyPipeline tracks commit groups, transaction completion, barrier
// parity, and access to buffers while transfers remain in flight.
package cudatma
