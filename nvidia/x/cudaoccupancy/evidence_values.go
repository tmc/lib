package cudaoccupancy

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

// OccupancyValue derives a theoretical occupancy ceiling from compiled
// resources and launch geometry. ResourcesKnown must describe all required
// resource metadata; absent metadata is not an observed zero allocation.
func (k KernelResource) OccupancyValue(source cudaevidence.Source, threads, dynamicShared int, resourcesKnown bool) (cudaevidence.Value, error) {
	if err := source.Identity.Validate(); err != nil {
		return cudaevidence.Value{}, err
	}
	if k.Name != source.Identity.Entry {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: entry mismatch")
	}
	if strings.TrimSpace(source.Producer) == "" || strings.TrimSpace(source.Locator) == "" {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: missing producer or locator")
	}
	if k.Registers < 0 || k.StaticSharedMem < 0 || k.StackFrameBytes < 0 || k.ConstantBankSize < 0 || dynamicShared < 0 || threads <= 0 {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: invalid resources or launch geometry")
	}
	dev, err := LookupDevice(source.Identity.Target)
	if err != nil {
		return cudaevidence.Value{}, err
	}
	if threads > dev.MaxThreadsPerBlock || dev.WarpSize <= 0 || dev.RegAllocationUnit <= 0 || dev.SmemAllocationUnit <= 0 {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: invalid device or excessive block size")
	}
	// CalculateDevice rounds resources before multiplying them. Admit only
	// inputs whose additions, rounding and products fit its int arithmetic.
	if threads > math.MaxInt-(dev.WarpSize-1) || k.StaticSharedMem > math.MaxInt-dynamicShared {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: resource addition overflows")
	}
	shared := k.StaticSharedMem + dynamicShared
	if shared > math.MaxInt-(dev.SmemAllocationUnit-1) || k.Registers > math.MaxInt/dev.WarpSize {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: resource rounding or multiplication overflows")
	}
	registers := k.Registers * dev.WarpSize
	if registers > math.MaxInt-(dev.RegAllocationUnit-1) {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: register rounding overflows")
	}
	warps := (threads + dev.WarpSize - 1) / dev.WarpSize
	registers = roundUp(registers, dev.RegAllocationUnit)
	if registers > math.MaxInt/warps {
		return cudaevidence.Value{}, fmt.Errorf("occupancy evidence: block register count overflows")
	}
	if !resourcesKnown {
		return cudaevidence.New(cudaevidence.Record{Source: source, Reason: "required compiled resource metadata unavailable"})
	}
	report := k.CalculateDevice(dev, threads, dynamicShared)
	number := report.Occupancy
	return cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Derived, Number: &number, Unit: "ratio", Source: source,
		Basis: fmt.Sprintf("Calculate: architecture=%s threads=%d registers=%d static_shared=%d dynamic_shared=%d; theoretical residency ceiling, not runtime occupancy", report.Architecture, threads, k.Registers, k.StaticSharedMem, dynamicShared)})
}
