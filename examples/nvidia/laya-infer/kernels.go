package main

import (
	_ "embed"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// forwardCUDA is the embedded CUDA source for the encoder and decision head.
//
//go:embed forward.cu
var forwardCUDA string

// kernels holds the compiled device functions for one loaded module.
type kernels struct {
	mod driver.Module

	layerNorm   driver.Function
	embed       driver.Function
	splitQKV    driver.Function
	attnSoftmax driver.Function
	flashAttn   driver.Function
	addResid    driver.Function
	addBias     driver.Function
	addTypeEmb  driver.Function
	geglu       driver.Function
	relu        driver.Function
	gelu        driver.Function
	gatherRows  driver.Function
}

// forwardUnit is the name of the translation unit forward.cu compiles as,
// and forwardName the file name it compiles under. The line tables in the
// PTX and the SASS record forwardName, so it is the name a source-attributed
// profile reports and a reader has to be able to find on disk.
// cudalineinfo derives the second from the first, so they are written that
// way here too.
const (
	forwardUnit = "forward"
	forwardName = forwardUnit + ".cu"
)

// An entry is an entry point of forward.cu and the field of kernels holding
// its resolved handle.
type entry struct {
	name string
	dst  *driver.Function
}

// entries lists every entry point in forward.cu, in resolution order. It is
// the one list of them: loadKernels resolves it and forwardKernels names it,
// so what runs and what a profile attributes cannot drift apart.
func (k *kernels) entries() []entry {
	return []entry{
		{"LayerNorm", &k.layerNorm},
		{"Embed", &k.embed},
		{"SplitQKVRope", &k.splitQKV},
		{"AttnSoftmax", &k.attnSoftmax},
		{"FlashAttention", &k.flashAttn},
		{"AddResidual", &k.addResid},
		{"AddBias", &k.addBias},
		{"AddTypeEmb", &k.addTypeEmb},
		{"GeGLU", &k.geglu},
		{"ReLU", &k.relu},
		{"GELU", &k.gelu},
		{"GatherRows", &k.gatherRows},
	}
}

// forwardKernels names every entry point in forward.cu. A unit that declares
// them attributes the whole module; a unit naming one kernel would file the
// other eleven under the remainder as subroutines.
func forwardKernels() []string {
	var k kernels
	es := k.entries()
	names := make([]string, len(es))
	for i, e := range es {
		names[i] = e.name
	}
	return names
}

// compileKernels JIT-compiles forward.cu for dev and resolves every entry point.
func compileKernels(dev *cuda.Device) (*kernels, error) {
	mod, err := nvrtcx.Compile(forwardCUDA,
		nvrtcx.WithName(forwardName),
		nvrtcx.WithDeviceArch(dev),
	)
	if err != nil {
		return nil, fmt.Errorf("compile forward kernels: %w", err)
	}
	return loadKernels(mod, dev)
}

// loadKernels loads a compiled forward.cu onto dev and resolves every entry
// point. The --lines path compiles the same source through cudalineinfo and
// loads the result here, so the module that runs is the module that was
// disassembled.
func loadKernels(mod *nvrtcx.CompiledModule, dev *cuda.Device) (*kernels, error) {
	loaded, err := mod.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", err)
	}

	k := &kernels{mod: loaded}
	for _, e := range k.entries() {
		fn, err := loaded.GetFunction(e.name)
		if err != nil {
			return nil, fmt.Errorf("resolve kernel %s: %w", e.name, err)
		}
		*e.dst = fn
	}
	return k, nil
}

// Close unloads the compiled module.
func (k *kernels) Close() error {
	if k.mod != 0 {
		return k.mod.Unload()
	}
	return nil
}
