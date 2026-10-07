package dcgm

import (
	"encoding/binary"
	"fmt"
)

// PolicyBool creates a boolean condition parameter using the pinned BOOL tag.
func PolicyBool(value bool) PolicyConditionParams {
	p := PolicyConditionParams{Tag: 0}
	if value {
		var union [8]byte
		binary.NativeEndian.PutUint32(union[:4], 1)
		p.Val[0] = binary.NativeEndian.Uint64(union[:])
	}
	return p
}

// PolicyNumber creates an unsigned threshold using the pinned LLONG tag.
func PolicyNumber(value uint64) PolicyConditionParams {
	return PolicyConditionParams{Tag: 1, Val: [1]uint64{value}}
}

// ReadPolicies reads count policy slots, each initialized with the v1 stamp.
// Count must be between 1 and MaxNumDevices. DCGM accepts an input count and
// supplies no output count; the returned slice therefore contains exactly the
// caller-requested slots, not an independently measured group size.
func ReadPolicies(handle Handle, group GpuGrp, count int, status StatusHandle) ([]Policy, error) {
	if handle == 0 || count < 1 || count > MaxNumDevices {
		return nil, fmt.Errorf("dcgm: invalid policy array extent")
	}
	policies := make([]Policy, count)
	for i := range policies {
		policies[i].Version = PolicyVersion1
	}
	if err := PolicyGet(handle, group, int32(count), &policies[0], status); err != nil {
		return nil, fmt.Errorf("dcgm: read policies: %w", err)
	}
	return policies, nil
}
