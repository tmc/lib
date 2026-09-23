package main

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
)

// TestSamplePCsPhysical runs the first kernel under the hardware PC sampler
// and checks that the samples land on the Go statements the static table
// names. The counts are a measurement and will differ run to run; what must
// hold is that every sample resolves to an instruction of the kernel, and
// that the statement whose instructions include an exponential and a
// division takes more of the time than the loads around it.
func TestSamplePCsPhysical(t *testing.T) {
	if os.Getenv("CUDA_PC_SAMPLING_TEST") != "1" {
		t.Skip("CUDA_PC_SAMPLING_TEST=1 requires CUPTI 13 and exclusive use of the device's profiling hardware")
	}
	if err := cuda.Init(); err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	opts := cudalineinfo.Options{Arch: fmt.Sprintf("sm_%d%d", major, minor), Nvdisasm: os.Getenv("NVDISASM")}
	a, fk, err := attribute(ctx, kernels[0], opts)
	if err != nil {
		t.Fatal(err)
	}
	cuCtx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	r, err := newRun(dev, a, fk, inputs(fk, 1<<20))
	if err != nil {
		t.Fatal(err)
	}
	defer r.close()
	workload := func(between func() error) error {
		for i := 0; i < 20; i++ {
			if err := r.launch(); err != nil {
				return err
			}
			if between == nil {
				continue
			}
			if err := between(); err != nil {
				return err
			}
		}
		return nil
	}
	p, err := samplePCs(ctx, cupti.Context(cuCtx.Raw()), a, 10, "", workload)
	if err != nil {
		t.Fatal(err)
	}
	if p.Total() == 0 {
		t.Fatal("the sampler returned no samples at all")
	}
	if p.Unresolved != 0 || p.Foreign != 0 {
		t.Errorf("%d unresolved and %d foreign of %d samples are at PCs the disassembly does not cover", p.Unresolved, p.Foreign, p.Total())
	}
	var silu, top cudalineinfo.Site
	for site, n := range p.Sites {
		if site.Line == siluLine {
			silu = site
		}
		if n > p.Sites[top] {
			top = site
		}
	}
	if silu == (cudalineinfo.Site{}) {
		t.Fatalf("no sample landed on %s:%d", kernelsFile, siluLine)
	}
	if top != silu {
		t.Errorf("most samples landed on %s (%d), not on %s (%d)", top, p.Sites[top], silu, p.Sites[silu])
	}
	t.Logf("%d samples, %d on %s", p.Total(), p.Sites[silu], silu)
}

// siluLine is the line of kernels.go holding act := x.Silu(), whose
// exponential and division are the kernel's arithmetic.
const (
	kernelsFile = "kernels.go"
	siluLine    = 14
)
