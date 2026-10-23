// Package gpucampaign runs the phases of a bounded GPU measurement campaign
// as subprocesses and records a receipt for each, so that a campaign that
// loses its GPU, overruns its time or crashes still says what ran.
//
// A [Runner] owns one deadline. A phase starts only if its whole cap fits in
// the time left, runs in its own process group, and is killed with the group
// when its cap expires, so compilers and other descendants die with it. When
// a GPU phase fails, the runner calls its health check (see
// [github.com/tmc/lib/nvidia/x/gpuhealth]); if that fails too, every later
// GPU phase is NOT_RUN with the health failure as its reason, instead of
// failing one by one with misleading errors.
//
// [WriteJSON] writes receipts atomically, so a reader never sees half a
// receipt from a killed campaign.
//
// The package is the in-process part of the remote campaign layout used for
// the H200 kernel challenge: pinned image, preflight, phases ordered timing
// before profiling, receipts after every phase. Rental, SSH and container
// setup stay outside it.
package gpucampaign
