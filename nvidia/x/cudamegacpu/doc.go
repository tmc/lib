// Package cudamegacpu runs a [cudamega.Program] on operating-system threads.
//
// It exists to make the primitive boundary in [cudamega] a claim rather than a
// hope. An interface with one implementation has not been shown to abstract
// anything, and the cheapest second implementation is not a second vendor: it
// is the same persistent-worker model on N OS threads, with no device, no
// lock and no booking, running in ordinary tests on every change.
//
// The forward-progress guarantee this backend declares is
// [cudamega.IndependentThreads], and it is true rather than asserted: each
// worker thread calls [runtime.LockOSThread], so a worker that spins on a
// predecessor's completion word cannot prevent that predecessor from running.
// That is exactly the property a cross-worker spin needs and exactly the
// property Vulkan and Metal do not provide.
//
// This backend is a portability falsifier and not an equivalence oracle. It
// shares a scheduler and a driver with the thing under test, so a
// dependency-ordering defect reproduces identically on both arms and their
// agreement says nothing about it. The oracle is [Reference], which computes
// the same result sequentially from the graph and does not use the driver,
// the primitives or any thread.
package cudamegacpu
