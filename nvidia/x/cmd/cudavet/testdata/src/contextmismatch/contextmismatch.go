package contextmismatch

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

var alpha = float32(1)

func setCurrent(h cublas.Handle, a, b driver.Context, y driver.DevicePtr) error {
	if err := a.SetCurrent(); err != nil {
		return err
	}
	x, err := driver.MemAllocBytes(64)
	if err != nil {
		return err
	}
	if err := b.SetCurrent(); err != nil {
		return err
	}
	return h.Saxpy(16, &alpha, x, 1, y, 1) // want `x was allocated while a was current, but Saxpy uses it while b is current; device memory belongs to the context it was allocated in`
}

func pushPop(f driver.Function, a, b driver.Context) error {
	if err := a.PushCurrent(); err != nil {
		return err
	}
	p, _, err := driver.MemAllocPitch(64, 4, 4)
	if err != nil {
		return err
	}
	driver.CtxPopCurrent()
	if err := b.PushCurrent(); err != nil {
		return err
	}
	defer driver.CtxPopCurrent()
	args := []unsafe.Pointer{unsafe.Pointer(&p)}
	return f.LaunchKernel(driver.Dim3{X: 1}, driver.Dim3{X: 32}, 0, 0, args) // want `p was allocated while a was current, but LaunchKernel uses it`
}

func popRestores(f driver.Function, a, b driver.Context) error {
	if err := b.SetCurrent(); err != nil {
		return err
	}
	if err := a.PushCurrent(); err != nil {
		return err
	}
	p, err := driver.MemAlloc[float32](16)
	if err != nil {
		return err
	}
	driver.CtxPopCurrent()
	return f.LaunchKernel(driver.Dim3{X: 1}, driver.Dim3{X: 32}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}) // want `p was allocated while a was current, but LaunchKernel uses it while b is current`
}

func devices(h cublas.Handle) error {
	d0, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	d1, err := cuda.NewDevice(1)
	if err != nil {
		return err
	}
	var x driver.DevicePtr
	if err := d0.WithContext(func() error {
		var err error
		x, err = driver.MemAllocBytes(64)
		return err
	}); err != nil {
		return err
	}
	return d1.WithContext(func() error {
		return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while the primary context of device 0 was current, but Saxpy uses it while the primary context of device 1 is current`
	})
}

func contextRun(h cublas.Handle, d0, d1 *cuda.Device) error {
	c0, err := d0.PrimaryContext()
	if err != nil {
		return err
	}
	c1, err := d1.PrimaryContext()
	if err != nil {
		return err
	}
	var x driver.DevicePtr
	c0.Run(func() error {
		var err error
		x, err = driver.MemAllocBytes(64)
		return err
	})
	return c1.Run(func() error {
		return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while the primary context of d0 was current, but Saxpy uses it while the primary context of d1 is current`
	})
}

func created(h cublas.Handle, dev driver.Device) error {
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		return err
	}
	defer ctx.Close()
	x, err := driver.MemAllocBytes(64)
	if err != nil {
		return err
	}
	primary, err := dev.PrimaryCtxRetain()
	if err != nil {
		return err
	}
	primary.SetCurrent()
	y := x + 32
	return h.Saxpy(8, &alpha, y, 1, y, 1) // want `y was allocated while ctx was current, but Saxpy uses it while the primary context of dev is current`
}

func bothBranches(h cublas.Handle, a, b driver.Context, ok bool) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	x += driver.DevicePtr(len("pad"))
	if ok {
		b.SetCurrent()
	} else {
		b.SetCurrent()
	}
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
}

func returnInBranch(h cublas.Handle, a, b driver.Context, ok bool) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	if !ok {
		a.SetCurrent()
		return nil
	}
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
}

func sameContext(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	a.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func samePrimary(h cublas.Handle, dev driver.Device) error {
	c1, _ := dev.PrimaryCtxRetain()
	c2, _ := dev.PrimaryCtxRetain()
	c1.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	c2.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func sameOrdinal(h cublas.Handle) error {
	dev, _ := driver.DeviceGet(0)
	c1, _ := dev.PrimaryCtxRetain()
	c2, _ := driver.Device(0).PrimaryCtxRetain()
	c1.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	c2.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func oneBranch(h cublas.Handle, a, b driver.Context, ok bool) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	if ok {
		b.SetCurrent()
	}
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func peer(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	if err := a.EnablePeerAccess(0); err != nil {
		return err
	}
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func enablePeers(a, b driver.Context) error {
	b.SetCurrent()
	return a.EnablePeerAccess(0)
}

func peerHelper(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	enablePeers(a, b)
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func copies(a, b driver.Context, host []float32) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	if err := x.CopyToHost(host); err != nil {
		return err
	}
	return x.Free()
}

func switchTo(c driver.Context) { c.SetCurrent() }

func helperSwitches(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	switchTo(a)
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func closureSwitches(h cublas.Handle, a, b driver.Context) error {
	back := func() { a.SetCurrent() }
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	back()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func closureKeeps(h cublas.Handle, a, b driver.Context) error {
	check := func(err error) {}
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	check(b.SetCurrent())
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
}

func perContext(h cublas.Handle, ctxs []driver.Context) {
	for _, c := range ctxs {
		c.SetCurrent()
		x, _ := driver.MemAllocBytes(64)
		h.Saxpy(16, &alpha, x, 1, x, 1)
	}
}

func loopSwitches(h cublas.Handle, a, b driver.Context, n int) {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	for i := 0; i < n; i++ {
		h.Saxpy(16, &alpha, x, 1, x, 1)
		b.SetCurrent()
	}
}

func loopBreaks(h cublas.Handle, a, b driver.Context, n int) {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	for i := 0; i < n; i++ {
		if i == 1 {
			a.SetCurrent()
			break
		}
		b.SetCurrent()
	}
	h.Saxpy(16, &alpha, x, 1, x, 1)
}

func loopReports(h cublas.Handle, a, b driver.Context, n int) {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	for i := 0; i < n; i++ {
		h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
	}
}

func managed(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocManaged[float32](16, 1)
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func restored(h cublas.Handle, a, b driver.Context) error {
	prev, _ := driver.CtxGetCurrent()
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	prev.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func mixedKinds(h cublas.Handle, c *cuda.Context, d *cuda.Device) error {
	var x driver.DevicePtr
	c.Run(func() error {
		var err error
		x, err = driver.MemAllocBytes(64)
		return err
	})
	return d.WithContext(func() error {
		return h.Saxpy(16, &alpha, x, 1, x, 1)
	})
}

func reassigned(h cublas.Handle, a, b driver.Context, y driver.DevicePtr) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	x = y
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func fill(p *driver.DevicePtr) {}

func addressTaken(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	fill(&x)
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func goroutine(h cublas.Handle, a, b driver.Context) {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	go func() {
		b.SetCurrent()
		h.Saxpy(16, &alpha, x, 1, x, 1)
	}()
}

func unknownDevices(h cublas.Handle, i, j int) error {
	c1, _ := driver.Device(i).PrimaryCtxRetain()
	c2, _ := driver.Device(j).PrimaryCtxRetain()
	c1.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	c2.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func withGoto(h cublas.Handle, a, b driver.Context, ok bool) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	if ok {
		a.SetCurrent()
		goto use
	}
use:
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func assignedLater(h cublas.Handle, a, b driver.Context, y driver.DevicePtr) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	defer func() { x = y }()
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func offset(p driver.DevicePtr) driver.DevicePtr { return p }

func fromCall(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	y := x + offset(0)
	return h.Saxpy(16, &alpha, y, 1, y, 1)
}

func funcParam(h cublas.Handle, a, b driver.Context, fn func()) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	fn()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func popUnknown(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	driver.CtxPopCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func onA(a driver.Context) {
	prev, _ := driver.CtxGetCurrent()
	defer prev.SetCurrent()
	a.SetCurrent()
}

func helperRestores(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	onA(a)
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
}

func viaRun(c *cuda.Context, a driver.Context) {
	c.Run(func() error { return a.SetCurrent() })
}

func helperRuns(h cublas.Handle, c *cuda.Context, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	viaRun(c, a)
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
}

func destroyed(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	b.Close()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func eitherDevice(h cublas.Handle, d0, d1 driver.Device, a driver.Context, ok bool) error {
	c, _ := d0.PrimaryCtxRetain()
	if ok {
		c, _ = d1.PrimaryCtxRetain()
	}
	c.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	d0ctx, _ := d0.PrimaryCtxRetain()
	d0ctx.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func typesDiffer(h cublas.Handle, c *cuda.Context, a driver.Context) error {
	var x driver.DevicePtr
	c.Run(func() error {
		x, _ = driver.MemAllocBytes(64)
		return nil
	})
	a.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func ordinalOrParam(h cublas.Handle, d driver.Device) error {
	c0, _ := driver.Device(0).PrimaryCtxRetain()
	cd, _ := d.PrimaryCtxRetain()
	c0.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	cd.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func localDevices(h cublas.Handle, i, j int) error {
	di := driver.Device(i)
	dj := driver.Device(j)
	ci, _ := di.PrimaryCtxRetain()
	cj, _ := dj.PrimaryCtxRetain()
	ci.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	cj.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func switchNoDefault(h cublas.Handle, a, b driver.Context, n int) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	switch n {
	case 1:
		b.SetCurrent()
	}
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func switchBreaks(h cublas.Handle, a, b driver.Context, n int, ok bool) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	switch n {
	case 1:
		a.SetCurrent()
		if ok {
			break
		}
		b.SetCurrent()
	default:
	}
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func selectAll(h cublas.Handle, a, b driver.Context, c1, c2 chan int) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	select {
	case <-c1:
		b.SetCurrent()
	case <-c2:
		b.SetCurrent()
	}
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
}

func runPushes(h cublas.Handle, c *cuda.Context, a, b driver.Context) error {
	b.SetCurrent()
	a.PushCurrent()
	x, _ := driver.MemAllocBytes(64)
	c.Run(func() error { return a.PushCurrent() })
	driver.CtxPopCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func eitherOrdinal(h cublas.Handle, i int, ok bool) error {
	dev, _ := driver.DeviceGet(0)
	if ok {
		dev, _ = driver.DeviceGet(1)
	}
	other, _ := driver.DeviceGet(0)
	if ok {
		other = driver.Device(i)
	}
	c, _ := dev.PrimaryCtxRetain()
	co, _ := other.PrimaryCtxRetain()
	c1, _ := driver.Device(1).PrimaryCtxRetain()
	c.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	co.SetCurrent()
	y, _ := driver.MemAllocBytes(64)
	c1.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, y, 1)
}

func pickDevice(i int) driver.Device { return driver.Device(i % 2) }

func pickedDevices(h cublas.Handle) error {
	d0 := pickDevice(0)
	d2 := pickDevice(2)
	c0, _ := d0.PrimaryCtxRetain()
	c2, _ := d2.PrimaryCtxRetain()
	c0.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	c2.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func twoPointers(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	w, _ := driver.MemAllocBytes(64)
	d := (x - x) + w
	return h.Saxpy(16, &alpha, d, 1, d, 1)
}

func getFunc() func() { return nil }

func funcFromCall(h cublas.Handle, a, b driver.Context) error {
	f := getFunc()
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	f()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func continueSkips(h cublas.Handle, a, b driver.Context, n int) {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	for i := 0; i < n; i++ {
		b.SetCurrent()
		if i == 0 {
			a.SetCurrent()
			continue
		}
		h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
	}
}

func loopCarried(h cublas.Handle, a, b driver.Context, z driver.DevicePtr, n int) {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	y := x
	for i := 0; i < n; i++ {
		y = x
		x = z
	}
	b.SetCurrent()
	h.Saxpy(16, &alpha, y, 1, y, 1)
}

func branchAllocs(h cublas.Handle, a, b driver.Context, ok bool) error {
	var x driver.DevicePtr
	if ok {
		a.SetCurrent()
		x, _ = driver.MemAllocBytes(64)
	} else {
		b.SetCurrent()
		x, _ = driver.MemAllocBytes(64)
	}
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func rangeAssigns(h cublas.Handle, a, b driver.Context, ctxs []driver.Context) error {
	b.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	c := a
	for _, c = range ctxs {
	}
	c.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func ctxAddressTaken(h cublas.Handle, a, b driver.Context, set func(*driver.Context)) error {
	b.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	c := a
	set(&c)
	c.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func aliased(h cublas.Handle, a, b driver.Context) error {
	c := a
	c.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current`
}

func cycle(h cublas.Handle, a, b driver.Context) error {
	c, d := a, b
	c = d
	d = c
	c.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func zeroContext(h cublas.Handle, b driver.Context) error {
	var z driver.Context
	z.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func rangeLater(h cublas.Handle, a, b driver.Context, ptrs []driver.DevicePtr) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	defer func() {
		for _, x = range ptrs {
		}
	}()
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func unreachable(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	return nil
	h.Saxpy(16, &alpha, x, 1, x, 1)
	return nil
}

func switchVia(c driver.Context) { switchTo(c) }

func helperChain(h cublas.Handle, a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	switchVia(a)
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func stackDisagree(h cublas.Handle, a, b driver.Context, ok bool) error {
	if ok {
		b.SetCurrent()
		a.PushCurrent()
	} else {
		a.PushCurrent()
	}
	x, _ := driver.MemAllocBytes(64)
	driver.CtxPopCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func stackAgrees(h cublas.Handle, a, b driver.Context, ok bool) error {
	b.SetCurrent()
	if ok {
		a.PushCurrent()
	} else {
		a.PushCurrent()
	}
	x, _ := driver.MemAllocBytes(64)
	driver.CtxPopCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current, but Saxpy uses it while b is current`
}

func deviceArith(h cublas.Handle) error {
	d := driver.Device(1)
	d -= 1
	c, _ := d.PrimaryCtxRetain()
	c0, _ := driver.Device(0).PrimaryCtxRetain()
	c.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	c0.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func newAndParam(h cublas.Handle, a driver.Context, dev driver.Device) error {
	c, _ := driver.NewContext(dev, 0)
	x, _ := driver.MemAllocBytes(64)
	_ = c
	a.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func keep(p driver.DevicePtr) error { return nil }

func nonBinding(a, b driver.Context) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	return keep(x)
}

func builtinCall(h cublas.Handle, a, b driver.Context, ptrs []driver.DevicePtr) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	b.SetCurrent()
	ptrs = append(ptrs, x)
	return h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current, but Saxpy uses it while b is current`
}

func loopBreakAssigns(h cublas.Handle, a, b driver.Context, n int) error {
	a.SetCurrent()
	x, _ := driver.MemAllocBytes(64)
	for i := 0; i < n; i++ {
		b.SetCurrent()
		x, _ = driver.MemAllocBytes(64)
		break
	}
	b.SetCurrent()
	return h.Saxpy(16, &alpha, x, 1, x, 1)
}

func insideGoroutine(h cublas.Handle, a, b driver.Context) {
	go func() {
		a.SetCurrent()
		x, _ := driver.MemAllocBytes(64)
		b.SetCurrent()
		h.Saxpy(16, &alpha, x, 1, x, 1) // want `x was allocated while a was current, but Saxpy uses it while b is current`
	}()
}
