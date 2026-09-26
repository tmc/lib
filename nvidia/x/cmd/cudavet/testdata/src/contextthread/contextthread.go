package contextthread

import (
	"runtime"

	"github.com/tmc/lib/nvidia/driver"
)

func unlocked(dev driver.Device) error {
	ctx, err := driver.NewContext(dev, 0) // want `NewContext makes a context current on a thread the goroutine may leave; call runtime.LockOSThread first`
	if err != nil {
		return err
	}
	defer ctx.Destroy()
	return ctx.SetCurrent()
}

func handedOff(dev driver.Device) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		return err
	}
	defer ctx.Destroy()
	done := make(chan error)
	go func() { // want `goroutine calls CtxSynchronize without the context made current before it started; it runs on another thread, so make the context current inside it`
		done <- driver.CtxSynchronize()
	}()
	return <-done
}

func locked(dev driver.Device) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		return err
	}
	defer ctx.Destroy()
	done := make(chan error)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		if err := ctx.SetCurrent(); err != nil {
			done <- err
			return
		}
		done <- driver.CtxSynchronize()
	}()
	return <-done
}

// enumerate starts goroutines before any context is current; they need none.
func enumerate() {
	go func() { driver.DeviceGet(0) }()
}

// setup makes ctx current; its only caller has locked the thread.
func setup(ctx driver.Context) error {
	return ctx.SetCurrent()
}

func lockedCaller(ctx driver.Context) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return setup(ctx)
}

// deferred runs its closure on the locked thread of the function around it.
func deferred(ctx driver.Context) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer func() { ctx.PushCurrent() }()
}

// helper is fine, but its caller is not.
func helper(ctx driver.Context) error {
	return ctx.SetCurrent()
}

func Caller(ctx driver.Context) error {
	return helper(ctx) // want `helper makes a context current on a thread the goroutine may leave; call runtime.LockOSThread first`
}
