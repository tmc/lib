package unlockdestroy

import (
	"runtime"

	"github.com/tmc/lib/nvidia/driver"
)

func deferOrder(dev driver.Device) error {
	runtime.LockOSThread()
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		runtime.UnlockOSThread()
		return err
	}
	defer ctx.Destroy()
	defer runtime.UnlockOSThread() // want `this deferred UnlockOSThread runs before the deferred Destroy above it; defer it first`
	return driver.CtxSynchronize()
}

func unlockFirst(dev driver.Device) error {
	runtime.LockOSThread()
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		return err
	}
	runtime.UnlockOSThread()
	return ctx.Destroy() // want `Destroy runs after runtime.UnlockOSThread, on whatever thread the goroutine is on`
}

func elsewhere(ctx driver.Context) {
	go func() {
		ctx.Destroy() // want `ctx.Destroy runs in a new goroutine, on a different thread from the one ctx is current on`
	}()
}

func ordered(dev driver.Device) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		return err
	}
	defer ctx.Destroy()
	return driver.CtxSynchronize()
}

func ownGoroutine(dev driver.Device) {
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		ctx, err := driver.NewContext(dev, 0)
		if err != nil {
			return
		}
		defer ctx.Destroy()
	}()
}
