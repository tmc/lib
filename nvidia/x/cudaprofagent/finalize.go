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
package cudaprofagent

import "sync"

// Finalize flushes and closes the attached agent's journal, and reports
// whether an agent was there to finalize.
//
// The boolean is the point of the signature. A capture that collects nothing
// looks the same whether the agent was absent or present and never asked to
// finalize, and those need different fixes: the first is a launch problem, the
// second is this call being missing. Returning the distinction lets a target
// say which one it was instead of leaving the reader to guess from an empty
// journal.
//
// Calling it more than once is harmless. The agent runs the ordered flush
// once and every other caller waits for the finalized record, so a deferred
// call followed by a normal exit finalizes exactly once.
func Finalize() bool {
	fn := agentFinalize()
	if fn == nil {
		return false
	}
	fn()
	return true
}

var agentFinalize = sync.OnceValue(lookupFinalize)
