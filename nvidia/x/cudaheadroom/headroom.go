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
	if kernel == nil || kernel.Device() == nil {
		return Budget{}, fmt.Errorf("cudaheadroom: kernel has no device")
	}
	return inspect(kernel)
}

func inspect(k *cuda.Kernel) (Budget, error) {
	dev := k.Device()
	if dev == nil {
		return Budget{}, fmt.Errorf("cudaheadroom: kernel has no device")
	}
	mode, err := k.Attribute(driver.FuncAttributeSharedMemoryMode)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query shared mode: %w", err)
	}
	if mode != int(driver.SharedMemoryModeDefault) {
		return Budget{}, fmt.Errorf("cudaheadroom: unsupported shared mode %d", mode)
	}
	user, err := dev.Attribute(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query user limit: %w", err)
	}
	static, err := k.Attribute(driver.FuncAttributeSharedSizeBytes)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query compiled static size: %w", err)
	}
	dynamic, err := k.Attribute(driver.FuncAttributeMaxDynamicSharedSizeBytes)
	if err != nil {
		return Budget{}, fmt.Errorf("cudaheadroom: query function dynamic limit: %w", err)
	}
	b, err := budget(int64(user), int64(static), int64(dynamic))
	if err != nil {
		return Budget{}, err
	}
	reserved, err := dev.Attribute(driver.DevAttrReservedSharedMemoryPerBlock)
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
	b, err := Inspect(kernel)
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
	b, err := Inspect(kernel)
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
		if err = kernel.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, int(need)); err != nil {
			return b, fmt.Errorf("cudaheadroom: set dynamic limit: %w", err)
		}
	}
	return CheckPlan(kernel, plan, otherStatic)
}
