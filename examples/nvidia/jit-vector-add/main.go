// Jit-vector-add compiles and launches a CUDA vector-add kernel using the
// high-level CUDA helper packages.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const source = `
extern "C" __global__ void add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}`

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	defer dev.Close()
	compiled, err := nvrtcx.Compile(source, nvrtcx.WithName("add.cu"), nvrtcx.WithDeviceArch(dev))
	if err != nil {
		return err
	}

	const n = 1024
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(2 * i)
	}
	c := make([]float32, n)

	return dev.WithContext(func() error {
		mod, err := compiled.Load(dev)
		if err != nil {
			return err
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("add")
		if err != nil {
			return err
		}
		da, err := cudamem.Alloc[float32](dev, n)
		if err != nil {
			return err
		}
		defer da.Free()
		db, err := cudamem.Alloc[float32](dev, n)
		if err != nil {
			return err
		}
		defer db.Free()
		dc, err := cudamem.Alloc[float32](dev, n)
		if err != nil {
			return err
		}
		defer dc.Free()
		if err := da.CopyFromHost(a); err != nil {
			return err
		}
		if err := db.CopyFromHost(b); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(fn).Grid((n+255)/256).Block(256).Launch(da, db, dc, int32(n)); err != nil {
			return err
		}
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
		if err := dc.CopyToHost(c); err != nil {
			return err
		}
		for i := range c {
			if want := a[i] + b[i]; c[i] != want {
				return fmt.Errorf("c[%d] = %g, want %g", i, c[i], want)
			}
		}
		bytesCount := len(compiled.CUBIN())
		if bytesCount == 0 {
			bytesCount = len(compiled.PTX())
		}
		fmt.Printf("compiled %d bytes and added %d floats\n", bytesCount, n)
		return nil
	})
}
