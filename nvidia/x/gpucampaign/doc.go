// Package gpucampaign runs the phases of a bounded GPU measurement campaign
// as subprocesses and records a receipt for each, so that a campaign that
// loses its GPU, overruns its time or crashes still says what ran.
//
// A [Runner] owns one monotonic deadline. A phase starts only if its whole cap
// fits in the time left, runs in its own process group, and is killed with the
// group when its cap expires, so compilers and other descendants die with it.
// Before trusting a zero exit code or parsing JSON output, the runner inspects
// the phase's output with [DetectTerminalCUDAError]; if a GPU phase logs a
// terminal CUDA fault (illegal memory access, device-side assert, Xid, or lost
// device), or if its post-failure health check fails (see
// [github.com/tmc/lib/nvidia/x/gpuhealth]), the runner latches the GPU as lost
// and marks every later GPU phase NOT_RUN with that reason.
//
// [WriteJSON] writes receipts atomically, so a reader never sees half a
// receipt from a killed campaign. [Runner.WriteCheckpoint] additionally
// verifies the runner's monotonic deadline before and after syncing the
// temporary file, refusing to commit (and rolling back) if the deadline
// expired during the operation or checkpoint I/O.
package gpucampaign
