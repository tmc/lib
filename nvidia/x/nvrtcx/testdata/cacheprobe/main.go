// Cacheprobe compiles, loads, and executes a small result-checked CUDA kernel.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func main() {
	variant := flag.Int("variant", 1, "kernel multiplier")
	clear := flag.Bool("clear", false, "clear the cache before compiling")
	physical := flag.Bool("physical", false, "compile for the physical architecture")
	flag.Parse()
	if err := run(*variant, *clear, *physical); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(variant int, clear, physical bool) error {
	if variant < 1 {
		return fmt.Errorf("invalid variant %d", variant)
	}
	if clear {
		if err := nvrtcx.ClearCache(); err != nil {
			return err
		}
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return err
	}
	source := fmt.Sprintf(`
extern "C" __global__ void probe(int *value) {
  *value = *value * %d + 7;
}`, variant)
	arch := nvrtcx.WithArch(major, minor)
	if physical {
		arch = nvrtcx.WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor))
	}
	compiled, err := nvrtcx.Compile(source, nvrtcx.WithName("probe.cu"), arch)
	if err != nil {
		return err
	}
	const input = int32(11)
	want := input*int32(variant) + 7
	var got []int32
	err = dev.WithContext(func() error {
		mod, err := compiled.Load(dev)
		if err != nil {
			return err
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("probe")
		if err != nil {
			return err
		}
		value, err := cudamem.Alloc[int32](dev, 1)
		if err != nil {
			return err
		}
		defer value.Free()
		if err := value.CopyFromHost([]int32{input}); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(fn).Grid(1).Block(1).Launch(value); err != nil {
			return err
		}
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
		got = make([]int32, 1)
		return value.CopyToHost(got)
	})
	if err != nil {
		return err
	}
	if got[0] != want {
		return fmt.Errorf("kernel result = %d, want %d", got[0], want)
	}
	fmt.Printf("pid=%d key=%s variant=%d ptx=%d cubin=%d result=%d\n", os.Getpid(), compiled.Key(), variant, len(compiled.PTX()), len(compiled.CUBIN()), got[0])
	return nil
}
