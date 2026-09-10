//go:build darwin && arm64

package main

// metalSubmissions counts command buffer traffic across a whole run.
//
// metalBatchStats describes one batch and the runtime keeps only the most
// recent one per region, so nothing in the process could report how many
// command buffers a prefill actually issued. That number is the open question
// about prefill throughput: a gputrace sidecar measured ds4-min issuing 178,183
// command buffers to prefill 2048 tokens where ds4.c issues 5, and holding the
// GPU busy 49.5% of the prefill wall against ds4.c's 96.4%.
//
// A change that claims to reduce submissions has to show this count falling.
// Reordering the loops around evalLayer, for instance, does not: the same
// number of batches is created either way, and only a counter says so.
//
// Counting is not synchronization: these fields are incremented from the
// goroutine that drives the runtime, which encodes and submits serially. They
// are diagnostics, never a control input, and nothing reads them to decide what
// to encode.
type metalSubmissions struct {
	Buffers int // command buffers created
	Commits int // Commit calls; a committed buffer may still be in flight
	Waits   int // Wait calls, each one a synchronization point with the GPU
	Encoded int // dispatches encoded across every buffer
}

// sub returns the difference between two readings, so a caller can bracket a
// phase rather than reporting a whole session's total.
func (s metalSubmissions) sub(start metalSubmissions) metalSubmissions {
	return metalSubmissions{
		Buffers: s.Buffers - start.Buffers,
		Commits: s.Commits - start.Commits,
		Waits:   s.Waits - start.Waits,
		Encoded: s.Encoded - start.Encoded,
	}
}

// add returns the sum of two readings, so a caller measuring across several
// bracketed phases can accumulate them.
func (s metalSubmissions) add(more metalSubmissions) metalSubmissions {
	return metalSubmissions{
		Buffers: s.Buffers + more.Buffers,
		Commits: s.Commits + more.Commits,
		Waits:   s.Waits + more.Waits,
		Encoded: s.Encoded + more.Encoded,
	}
}

// perToken reports the counts divided by a token count, which is the form the
// ds4.c comparison takes: two command buffers per layer per token against a
// whole-chunk submission. It returns zeroes for a non-positive token count
// rather than dividing.
func (s metalSubmissions) perToken(tokens int) (buffers, commits, waits, encoded float64) {
	if tokens <= 0 {
		return 0, 0, 0, 0
	}
	n := float64(tokens)
	return float64(s.Buffers) / n, float64(s.Commits) / n, float64(s.Waits) / n, float64(s.Encoded) / n
}

// Submissions reports the runtime's command buffer counts so far.
func (r *metalRuntime) Submissions() metalSubmissions {
	if r == nil || r.ops == nil {
		return metalSubmissions{}
	}
	return r.ops.submissions
}
