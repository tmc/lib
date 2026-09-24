// Package cudaagreement executes one plan through more than one CUDA backend
// and reports, per step, the backend that actually ran it.
//
// The record is made by the executor after submission, from the path it took.
// It is not the backend a planner asked for: an executor that cannot admit a
// step to its backend falls back, runs the step another way, and records the
// fallback. Agreement that holds only over steps an arm declined says nothing
// about that arm, so [Verify] fails on a declined step even when the outputs
// match.
//
// Outputs are compared bitwise. That is the whole comparison: an integer path
// must agree exactly, and a float path needs a tolerance this package does not
// supply.
//
// [LaunchExecutor], [GraphExecutor] and [MegaExecutor] are the arms. The mega
// arm runs each chain of admissible steps as one launch of a [cudamega]
// persistent worker, and records [cudaplan.BackendMega] for a step only from
// the worker's own account of the run: a zero abort word and a start-order
// buffer showing that exactly the CTAs the program dealt the task took it.
//
// [Account] partitions a region's device span, from CUPTI kernel records
// ([KernelsFromActivity]), into time when only chain kernels ran, only other
// kernels ran, both ran, or none did. A region with no kernels has a span of
// zero, not the time it was observed, and [SpanAccount.Fractions] refuses to
// divide by it.
//
// A run needs a device, a stream, and one [Modules] to own the loaded kernel
// modules:
//
//	mods := cudaagreement.NewModules(dev)
//	defer mods.Close()
//	launch := &cudaagreement.LaunchExecutor{Modules: mods}
//	graph := &cudaagreement.GraphExecutor{Device: dev, Modules: mods}
//	defer graph.Close()
//
//	var arms []cudaagreement.Arm
//	for _, e := range []cudaagreement.Executor{launch, graph} {
//	    rec, err := e.Execute(stream, plan, buffers)
//	    // ... synchronize, read the output buffer ...
//	    arms = append(arms, cudaagreement.Arm{Record: rec, Output: out})
//	}
//	err := cudaagreement.Verify(arms)
package cudaagreement
