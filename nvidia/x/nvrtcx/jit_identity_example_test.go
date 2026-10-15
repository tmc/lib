package nvrtcx_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func ExampleCompiledModule_CompilerOptions() {
	// A zero module has not observed an invocation. No compiler is called.
	var module nvrtcx.CompiledModule
	options, known := module.CompilerOptions()
	fmt.Println(len(options), known)
	// Output: 0 false
}

func ExampleStripOfflineFlags() {
	flags, jit := nvrtcx.StripOfflineFlags([]string{"-O0"})
	opts, values := jit.DriverOptions()
	fmt.Println(len(flags), len(opts), values)
	// Output: 0 1 [0]
}
