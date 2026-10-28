package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// deviceProfile returns the profile of dev: its product's datasheet
// profile, or its target's when the product is not listed, with the SM
// count and shared memory limits read from dev.
func deviceProfile(dev *cuda.Device) (cudadevicetable.Profile, error) {
	name, err := dev.Name()
	if err != nil {
		return cudadevicetable.Profile{}, fmt.Errorf("device name: %w", err)
	}
	p, err := cudadevicetable.LookupProfile(name)
	if err != nil {
		major, minor, err := dev.ComputeCapability()
		if err != nil {
			return cudadevicetable.Profile{}, fmt.Errorf("device compute capability: %w", err)
		}
		if p, err = cudadevicetable.LookupProfile(fmt.Sprintf("sm_%d%da", major, minor)); err != nil {
			return cudadevicetable.Profile{}, err
		}
		p.Product = name
	}
	for _, a := range []struct {
		v    *cudadevicetable.Value
		attr driver.DeviceAttribute
		name string
	}{
		{&p.SMs, driver.DevAttrMultiprocessorCount, "DevAttrMultiprocessorCount"},
		{&p.SharedPerBlock, driver.DevAttrMaxSharedMemoryPerBlock, "DevAttrMaxSharedMemoryPerBlock"},
		{&p.SharedPerBlockOptin, driver.DevAttrMaxSharedMemoryPerBlockOptin, "DevAttrMaxSharedMemoryPerBlockOptin"},
		{&p.SharedPerSM, driver.DevAttrMaxSharedMemoryPerMultiprocessor, "DevAttrMaxSharedMemoryPerMultiprocessor"},
	} {
		n, err := dev.Attribute(a.attr)
		if err != nil {
			return cudadevicetable.Profile{}, fmt.Errorf("read %s: %w", a.name, err)
		}
		*a.v = cudadevicetable.Value{Value: int64(n), Source: cudadevicetable.Probed, Instrument: a.name}
	}
	return p, nil
}

// emitEgraph writes the e-graph decoder's kernels for the profile named
// target, as built for cfg, to dir/<target>/: llama_egraph.cu, the source
// llama-infer would hand NVRTC, and names.txt, its kernels' name
// expressions, one per line. It needs no GPU; compile the source with
// nvidia/x/cmd/cudacompilecheck to see each kernel's registers, spills
// and shared memory for that target.
func emitEgraph(dir, target string, cfg *ModelConfig, opt egraphOptions) error {
	p, err := cudadevicetable.LookupProfile(target)
	if err != nil {
		return err
	}
	opt.Profile = p
	// Without weights, a Qwen2 config is the only sign of QKV biases.
	bias := cfg.ModelType == "qwen2"
	plans, err := buildEgraphPlans(cfg, bias, opt)
	if err != nil {
		return fmt.Errorf("build e-graph plans for %s: %w", target, err)
	}
	eg := &egraphDecoder{plans: plans, megas: make(map[*hopper.Plan]hopper.Megakernel)}
	if opt.Mega {
		for _, pl := range eg.decodePlans() {
			if eg.megas[pl], err = pl.Megakernel(""); err != nil {
				return err
			}
		}
	}
	src, names, err := eg.source()
	if err != nil {
		return err
	}
	out := filepath.Join(dir, p.Target)
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "llama_egraph.cu"), []byte(src), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "names.txt"), []byte(strings.Join(names, "\n")+"\n"), 0o644); err != nil {
		return err
	}
	fmt.Printf("[E-Graph] %s (%s, %d SMs %s, %d B shared/block): %d kernels -> %s\n",
		p.Target, p.Product, p.SMs.Value, p.SMs.Source, p.SharedPerBlockOptin.Value, len(names), out)
	return nil
}
