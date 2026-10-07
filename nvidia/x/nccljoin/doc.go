// Package nccljoin aligns decoded NCCL Inspector collectives within an explicit
// caller-selected job and communicator epoch. Missing ranks are unobserved in
// that capture, not proof of absent execution, a stall or a dropped operation.
// Dump timestamps use the producer's host clock; durations are not start/end
// times. The package does not correct clocks or make health judgments.
//
// Point-to-point records are outside this version. Partial capture and drop
// facts are retained separately: neither identifies an undecoded collective.
// Communicator ID and sequence alone do not establish an epoch. Gathering,
// selecting epochs and verifying external membership belong to the caller.
package nccljoin
