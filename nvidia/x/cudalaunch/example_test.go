package cudalaunch_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// kernels names the device functions a forward pass needs. Fields without a
// "cuda" tag are looked up under the field name.
type kernels struct {
	Embed driver.Function `cuda:"EmbeddingLookup"`
	Norm  driver.Function `cuda:"RMSNorm"`
	GEMV  driver.Function `cuda:"LinearGEMV"`
}

// This illustrative fragment requires a loaded module and allocated device buffers.
func ExampleBind() {
	// mod is a module already loaded onto a device, for example by
	// nvrtcx.CompiledModule.Load.
	var mod driver.Module

	var k kernels
	if err := cudalaunch.Bind(mod, &k); err != nil {
		fmt.Println(err)
		return
	}

	var x, weight, out cudamem.Slice[float32]
	err := cudalaunch.Kernel(k.Norm).Grid(1).Block(256).
		Launch(x, weight, out, int32(x.Len()), float32(1e-5))
	if err != nil {
		fmt.Println(err)
	}
}

// A zero Slice packs as a null device pointer, so an optional kernel argument
// such as a missing bias vector needs no sentinel of its own.
// This illustrative fragment requires a real kernel and allocated input/output buffers.
func ExampleLauncher_Launch_optionalArgument() {
	var fn driver.Function
	var in, weight, out cudamem.Slice[float32]
	var bias cudamem.Slice[float32] // absent: packed as a null pointer

	err := cudalaunch.Kernel(fn).Grid(uint32(out.Len())).Block(128).
		Launch(in, weight, bias, out, int32(in.Len()), int32(out.Len()))
	if err != nil {
		fmt.Println(err)
	}
}
