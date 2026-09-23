// Package cudacluster expresses thread-block-cluster distributed shared memory.
//
// A cluster CTA may read and write another CTA's shared memory through a
// mapped window.  This package emits the PTX for that mapping, for the cluster
// barrier that must bracket it, and for the cluster geometry registers that
// address it.  It plans nothing and schedules nothing: cluster dimensions are
// an input.
//
// Two facts are kept apart throughout, because collapsing them is the mistake
// this package exists to make hard.
//
// The first is whether an assembler accepts a construct for a target.  That is
// a toolchain fact, recorded per exact target in
// [github.com/tmc/lib/nvidia/x/cudadevicetable], and it is what [Check]
// answers.  The second is whether a device executes it correctly, which only a
// run on that device answers.  [Check] returning nil is permission to attempt,
// never evidence of a working remote window.
//
// Absence is not a negative.  A target no instrument has looked at yields
// [ErrHardwarePending], distinct from [ErrUnsupported], which is an assembler
// that looked and refused.
//
// The emitters are pure text and are not gated: a PTX string is not a device
// access.  The gate is on the typed surface that stands for a real window.
package cudacluster
