package memallocptr

import "github.com/tmc/lib/nvidia/driver"

type node struct {
	value float32
	next  *node
}

type particle struct {
	pos, vel [3]float32
	mass     float32
}

func alloc(n int) error {
	a, err := driver.MemAlloc[*float32](n) // want `MemAlloc\[\*float32\]: \*float32 holds a pointer, which the garbage collector cannot follow into device memory`
	if err != nil {
		return err
	}
	defer a.Free()
	b, err := driver.MemAlloc[node](n) // want `MemAlloc\[node\]: node holds a pointer in field next, which the garbage collector cannot follow into device memory`
	if err != nil {
		return err
	}
	defer b.Free()
	c, err := driver.MemAlloc[particle](n)
	if err != nil {
		return err
	}
	defer c.Free()
	return nil
}

func copies(d driver.DevicePtr, names []string, xs []float64) error {
	if err := d.CopyFromHost(names); err != nil { // want `CopyFromHost\[string\]: string holds a string, which the garbage collector cannot follow into device memory`
		return err
	}
	return d.CopyFromHost(xs)
}
