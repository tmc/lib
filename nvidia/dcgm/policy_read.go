package dcgm

import (
	"fmt"
	"runtime"

	"github.com/ebitengine/purego"
)

// ReadPolicies reads count v1 policy slots without setting or triggering policy.
// Count must be between 1 and MaxNumDevices. The native API supplies no output
// count, so the returned extent is requested, not a measured group population.
// The caller owns the group, connection and optional native status handle.
func ReadPolicies(handle Handle, group GpuGrp, count int, status StatusHandle) ([]Policy, error) {
	if handle == 0 || count < 1 || count > MaxNumDevices {
		return nil, fmt.Errorf("dcgm: invalid policy array extent")
	}
	if err := checkLib(); err != nil {
		return nil, err
	}
	var get func(Handle, GpuGrp, int32, *Policy, StatusHandle) Result
	loadMu.Lock()
	ptr, err := lookupLibrarySymbol(lib, "dcgmPolicyGet")
	if err == nil {
		func() {
			defer func() {
				if p := recover(); p != nil {
					err = fmt.Errorf("bind policy get: %v", p)
				}
			}()
			purego.RegisterFunc(&get, ptr)
		}()
	}
	loadMu.Unlock()
	if err != nil {
		return nil, &SymbolError{Symbol: "dcgmPolicyGet", Err: err}
	}
	policies := make([]Policy, count)
	for i := range policies {
		policies[i].Version = 0x01000090
	}
	result := get(handle, group, int32(count), &policies[0], status)
	runtime.KeepAlive(policies)
	if err := toError(result, "dcgmPolicyGet"); err != nil {
		return nil, fmt.Errorf("dcgm: read policies: %w", err)
	}
	for _, p := range policies {
		if p.Version != 0x01000090 {
			return nil, fmt.Errorf("dcgm: policy response layout mismatch")
		}
	}
	return policies, nil
}
