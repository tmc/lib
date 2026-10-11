package cudaplan

import "context"

// Completion tracks execution of an asynchronous plan invocation and owns
// the borrowed input/output buffers and workspace leases until GPU execution completes.
type Completion struct {
	state *completionState
	steps []ExecutedStep
}

// ExecutedStep identifies a successfully submitted dispatch. Submission does
// not establish numerical correctness or completion; use Completion.Wait to
// observe completion. A graph replay reports Graph for every captured step.
type ExecutedStep struct {
	Index    int
	NodeName string
	Backend  TargetBackend
}

// ExecutedSteps returns the actual successfully submitted backends, in plan
// order. On a partial submission failure, unsubmitted steps are absent.
func (c *Completion) ExecutedSteps() []ExecutedStep {
	if c == nil {
		return nil
	}
	return append([]ExecutedStep(nil), c.steps...)
}

// Done reports whether a successful native drain has been observed. It does
// not poll the device; work may have finished before Wait observes it.
func (c *Completion) Done() bool {
	if c == nil || c.state == nil {
		return true
	}
	return c.state.done.Load()
}

// Wait waits for the GPU work to complete and releases borrowed buffer leases.
//
// Invariant: Cancellation stops only the host wait; it does NOT free in-flight
// or queued GPU buffers or cancel GPU execution. Native wait errors similarly retain buffers.
// A successful native drain releases borrowed buffer leases. A counted worker
// outcome error is then retained and returned by every Wait; it does not keep
// storage borrowed after the GPU has stopped using it.
func (c *Completion) Wait(ctx context.Context) error {
	if c == nil || c.state == nil {
		return nil
	}
	return c.state.Wait(ctx)
}

// Close waits without cancellation and releases the completion event.
// Safe to call multiple times or concurrently.
func (c *Completion) Close() error {
	return c.Wait(nil)
}
