package cudaheadroom

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudashared"
)

// Budget records measured per-block user and function limits, in bytes.
// Reserved is nil when the optional driver reservation query is unavailable.
// ReservationError retains that query's error. Neither changes Ceiling.
type Budget struct {
	UserLimit, CompiledStatic, FunctionDynamic, Ceiling int64
	Reserved                                            *int64
	ReservationError                                    error
}

// Inspect queries the device and loaded function owned by kernel.
// The caller must keep the module open and serialize configuration changes.
func Inspect(kernel *cuda.Kernel) (Budget, error) {
	p, err := kernelProbes(kernel)
	if err != nil {
		return Budget{}, err
	}
	return inspect(p)
}

// probes is a private test seam. Public entry points bind only owned methods;
// there is no global hook and no alternate public kernel implementation.
type probes struct {
	device   func(driver.DeviceAttribute) (int, error)
	function func(driver.FunctionAttribute) (int, error)
	set      func(driver.FunctionAttribute, int) error
}

func kernelProbes(k *cuda.Kernel) (probes, error) {
	dev := k.Device()
	if dev == nil {
		return probes{}, fmt.Errorf("cudaheadroom: kernel has no device")
	}
	return probes{dev.Attribute, k.Attribute, k.SetAttribute}, nil
}

func inspect(p probes) (Budget, error) {
	if p.device == nil || p.function == nil {
		return Budget{}, fmt.Errorf("cudaheadroom: unavailable attribute probes")
	}
	mode, err := p.function(driver.FuncAttributeSharedMemoryMode)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query shared mode: %w", err)
	}
	if mode != int(driver.SharedMemoryModeDefault) {
		return Budget{}, fmt.Errorf("cudaheadroom: unsupported shared mode %d", mode)
	}
	user, err := p.device(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query user limit: %w", err)
	}
	static, err := p.function(driver.FuncAttributeSharedSizeBytes)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query compiled static size: %w", err)
	}
	dynamic, err := p.function(driver.FuncAttributeMaxDynamicSharedSizeBytes)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query function dynamic limit: %w", err)
	}
	b, err := budget(int64(user), int64(static), int64(dynamic))
	if err != nil {
		return Budget{}, err
	}
	reserved, err := p.device(driver.DevAttrReservedSharedMemoryPerBlock)
	if err != nil {
		b.ReservationError = err
	} else if reserved < 0 {
		b.ReservationError = fmt.Errorf("cudaheadroom: negative reservation %d", reserved)
	} else {
		v := int64(reserved)
		b.Reserved = &v
	}
	return b, nil
}
func budget(user, static, dynamic int64) (Budget, error) {
	if user <= 0 || static < 0 || dynamic < 0 || static > user {
		return Budget{}, fmt.Errorf("cudaheadroom: invalid measured limits user=%d static=%d dynamic=%d", user, static, dynamic)
	}
	ceiling := user - static
	if dynamic < ceiling {
		ceiling = dynamic
	}
	return Budget{UserLimit: user, CompiledStatic: static, FunctionDynamic: dynamic, Ceiling: ceiling}, nil
}

// CheckPlan checks plan against fresh measured limits for kernel.
// otherStatic is the compiled static storage outside the plan. The caller must
// establish that the plan's static regions correspond to this kernel's actual
// compiled storage; byte equality alone does not establish that correspondence.
// Unknown correspondence must not be passed as an estimate. Static storage is
// checked by equality and is not added to the compiled total a second time.
func CheckPlan(kernel *cuda.Kernel, plan *cudashared.Plan, otherStatic int64) (Budget, error) {
	p, err := kernelProbes(kernel)
	if err != nil {
		return Budget{}, err
	}
	return checkMeasuredPlan(p, plan, otherStatic)
}

func checkMeasuredPlan(p probes, plan *cudashared.Plan, otherStatic int64) (Budget, error) {
	b, err := inspect(p)
	if err != nil {
		return Budget{}, err
	}
	if err := checkPlan(b, plan, otherStatic, true); err != nil {
		return b, err
	}
	return b, nil
}
func checkPlan(b Budget, p *cudashared.Plan, other int64, cap bool) error {
	if p == nil {
		return fmt.Errorf("cudaheadroom: nil shared plan")
	}
	static, dynamic := p.StaticBytes(), p.DynamicBytes()
	if static < 0 || dynamic < 0 || other < 0 || other > b.CompiledStatic || static != b.CompiledStatic-other {
		return fmt.Errorf("cudaheadroom: static correspondence mismatch plan=%d other=%d compiled=%d", static, other, b.CompiledStatic)
	}
	ceiling := b.UserLimit - b.CompiledStatic
	if cap {
		ceiling = b.Ceiling
	}
	if dynamic > ceiling {
		return fmt.Errorf("cudaheadroom: requested dynamic shared memory %d bytes exceeds ceiling %d bytes", dynamic, ceiling)
	}
	return nil
}

// PreparePlan validates the device budget and static correspondence, raises the
// function dynamic limit only if needed, and reads it back before checking plan.
// It does not launch. A setter error may leave function configuration changed;
// callers must retain ownership and inspect it before proceeding.
func PreparePlan(kernel *cuda.Kernel, plan *cudashared.Plan, otherStatic int64) (Budget, error) {
	p, err := kernelProbes(kernel)
	if err != nil {
		return Budget{}, err
	}
	return preparePlan(p, plan, otherStatic)
}

func preparePlan(p probes, plan *cudashared.Plan, otherStatic int64) (Budget, error) {
	b, err := inspect(p)
	if err != nil {
		return Budget{}, err
	}
	if err = checkPlan(b, plan, otherStatic, false); err != nil {
		return b, err
	}
	need := plan.DynamicBytes()
	if need > b.FunctionDynamic {
		if need > int64(^uint32(0)>>1) {
			return b, fmt.Errorf("cudaheadroom: dynamic size exceeds function attribute range")
		}
		if p.set == nil {
			return b, fmt.Errorf("cudaheadroom: unavailable dynamic limit setter")
		}
		if err = p.set(driver.FuncAttributeMaxDynamicSharedSizeBytes, int(need)); err != nil {
			return b, fmt.Errorf("cudaheadroom: set dynamic limit: %w", err)
		}
	}
	return checkMeasuredPlan(p, plan, otherStatic)
}
