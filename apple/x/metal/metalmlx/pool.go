package metalmlx

import (
	"runtime"

	"github.com/tmc/apple/foundation"
)

// WithPool runs fn inside an Objective-C autorelease pool.
//
// Metal hands back autoreleased objects, and the bindings make an
// autoreleased NSString for every string argument. Go has no per-thread pool,
// so without one those objects are never freed. The goroutine is pinned for
// the call because a pool has to be drained on the thread that made it.
// Anything that must outlive fn has to be retained inside it.
func WithPool(fn func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	pool := foundation.NewNSAutoreleasePool()
	defer pool.Drain()
	fn()
}
