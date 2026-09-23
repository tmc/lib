package cudabpfdev

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

type resourceLimits struct {
	registers, sharedBytes uint32
}

type compiledResources struct {
	registers, sharedBytes, localBytes, maxThreads int32
}

func (r compiledResources) check(l resourceLimits) error {
	if r.registers < 0 || r.sharedBytes < 0 || r.localBytes < 0 || r.maxThreads < 0 {
		return fmt.Errorf("cudabpfdev: invalid compiled resource attributes")
	}
	if l.registers == 0 || uint32(r.registers) > l.registers {
		return fmt.Errorf("cudabpfdev: compiled registers %d exceed budget %d", r.registers, l.registers)
	}
	if uint32(r.sharedBytes) > l.sharedBytes {
		return fmt.Errorf("cudabpfdev: compiled shared memory %d exceeds budget %d", r.sharedBytes, l.sharedBytes)
	}
	// This emitter has no local arrays or stack. Local memory, including spills,
	// is not an authorized way to satisfy its register budget.
	if r.localBytes != 0 {
		return fmt.Errorf("cudabpfdev: compiled local memory %d is outside the emitter budget", r.localBytes)
	}
	if r.maxThreads < 32 {
		return fmt.Errorf("cudabpfdev: compiled kernel permits only %d threads per block", r.maxThreads)
	}
	return nil
}

func queryResources(get func(driver.FunctionAttribute) (int32, error)) (compiledResources, error) {
	var r compiledResources
	for _, field := range []struct {
		attribute driver.FunctionAttribute
		value     *int32
	}{
		{driver.FuncAttributeNumRegs, &r.registers},
		{driver.FuncAttributeSharedSizeBytes, &r.sharedBytes},
		{driver.FuncAttributeLocalSizeBytes, &r.localBytes},
		{driver.FuncAttributeMaxThreadsPerBlock, &r.maxThreads},
	} {
		value, err := get(field.attribute)
		if err != nil {
			return compiledResources{}, fmt.Errorf("cudabpfdev: query %s: %w", field.attribute, err)
		}
		*field.value = value
	}
	return r, nil
}
