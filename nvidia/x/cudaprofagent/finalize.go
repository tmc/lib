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
