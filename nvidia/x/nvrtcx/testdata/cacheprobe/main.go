// Cacheprobe compiles, loads, and executes a small result-checked CUDA kernel.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
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
	cleanup := probeCleanup{
		withContext: dev.WithContext,
		drain:       driver.CtxSynchronize,
		device:      dev.Close,
	}
	const input = int32(11)
	want := input*int32(variant) + 7
	var compiled *nvrtcx.CompiledModule
	var got []int32
	err = cleanup.run(func() error {
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
		compiled, err = nvrtcx.Compile(source, nvrtcx.WithName("probe.cu"), arch)
		if err != nil {
			cleanup.program = retryCleanup(err)
			return err
		}
		if err := dev.WithContext(func() error {
			mod, err := compiled.Load(dev)
			if err != nil {
				cleanup.module = retryCleanup(err)
				return err
			}
			cleanup.module = mod.Close
			fn, err := mod.Function("probe")
			if err != nil {
				return err
			}
			value, err := cudamem.Alloc[int32](dev, 1)
			if err != nil {
				return err
			}
			cleanup.free = value.Free
			if err := value.CopyFromHost([]int32{input}); err != nil {
				return err
			}
			// A failed launch may have admitted work. Only a successful drain
			// permits cleanup of its allocation and module.
			cleanup.pending = true
			if err := cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(1, 1, 1).Launch(value); err != nil {
				return err
			}
			if err := driver.CtxSynchronize(); err != nil {
				return err
			}
			cleanup.pending = false
			got = make([]int32, 1)
			return value.CopyToHost(got)
		}); err != nil {
			return err
		}
		if got[0] != want {
			return fmt.Errorf("kernel result = %d, want %d", got[0], want)
		}
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Printf("pid=%d key=%s variant=%d ptx=%d cubin=%d result=%d\n", os.Getpid(), compiled.Key(), variant, len(compiled.PTX()), len(compiled.CUBIN()), got[0])
	return nil
}

// A callback remains owned until it succeeds. The NVRTC program does not depend
// on the device context; allocations and modules do.
type probeCleanup struct {
	withContext func(func() error) error
	drain       func() error
	program     func() error
	free        func() error
	module      func() error
	device      func() error
	pending     bool
}

func (c *probeCleanup) run(work func() error) (err error) {
	defer func() { err = c.finish(err) }()
	return work()
}

func (c *probeCleanup) finish(err error) error {
	if c.program != nil {
		if cleanupErr := c.program(); cleanupErr != nil {
			err = errors.Join(err, fmt.Errorf("cacheprobe: retry program cleanup: %w", cleanupErr))
		} else {
			c.program = nil
		}
	}
	if c.pending || c.free != nil || c.module != nil {
		cleanupErr := c.withContext(func() error {
			if c.pending {
				if err := c.drain(); err != nil {
					return fmt.Errorf("cacheprobe: drain before cleanup: %w", err)
				}
				c.pending = false
			}
			var err error
			if c.free != nil {
				if cleanupErr := c.free(); cleanupErr != nil {
					err = errors.Join(err, fmt.Errorf("cacheprobe: free allocation: %w", cleanupErr))
				} else {
					c.free = nil
				}
			}
			if c.module != nil {
				if cleanupErr := c.module(); cleanupErr != nil {
					err = errors.Join(err, fmt.Errorf("cacheprobe: unload module: %w", cleanupErr))
				} else {
					c.module = nil
				}
			}
			return err
		})
		if cleanupErr != nil {
			err = errors.Join(err, fmt.Errorf("cacheprobe: clean device resources: %w", cleanupErr))
		}
	}
	if c.pending || c.free != nil || c.module != nil {
		return errors.Join(err, errors.New("cacheprobe: incomplete device cleanup retains resources"))
	}
	if c.device != nil {
		if cleanupErr := c.device(); cleanupErr != nil {
			err = errors.Join(err, fmt.Errorf("cacheprobe: release device context: %w", cleanupErr))
		} else {
			c.device = nil
		}
	}
	return err
}

// Compile and Load may retain a native object in a cleanup error. The fixture
// retries it once, without repeating compilation or loading.
func retryCleanup(err error) func() error {
	var closer io.Closer
	if errors.As(err, &closer) {
		return closer.Close
	}
	return nil
}
