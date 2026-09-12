//go:build linux

package cuptipcsampling

import (
	"errors"
	"fmt"

	"github.com/ebitengine/purego"
)

// CUPTI's PC-sampling header assigns function-name cleanup to native free.
// Resolve the process allocator, including any interposition, before enabling
// sampling. Do not unload it while a sampling session retains allocations.
func samplingNameAllocator() (func(uintptr), error) {
	ptr, err := purego.Dlsym(purego.RTLD_DEFAULT, "free")
	if err != nil {
		return nil, fmt.Errorf("cuptipcsampling: resolve native free: %w", err)
	}
	if ptr == 0 {
		return nil, errors.New("cuptipcsampling: native free is null")
	}
	var free func(uintptr)
	purego.RegisterFunc(&free, ptr)
	return free, nil
}
