// Package cudaprofagent lets a Go program finalize an attached cudaprof CUPTI
// agent before it exits.
//
// A Go program cannot finalize the agent by exiting. The agent's other three
// routes all run through libc: the ELF destructor and atexit are driven by
// exit(3), and the SIGTERM terminator needs a signal nobody sends -- and the Go
// runtime installs its own SIGTERM disposition after the agent's constructor
// has run, which takes that route away as well. runtime.exit issues the
// exit_group syscall directly, so none of the three fire. CUPTI's buffers are
// flushed at finalize, so they die with the process: the target exits 0 and
// reports its own work done while the capture holds no records at all.
//
// A target under capture therefore has to ask:
//
//	func main() {
//		defer cudaprofagent.Finalize()
//		...
//	}
//
// Finalize is a no-op when no agent is attached, so the call is safe to leave
// in a program that is usually run without one.
//
// On linux, capturing with cudaprof capture --finalize=exit-group makes the
// agent finalize on the exit_group syscall instead, which needs no change to
// the target.
package cudaprofagent
