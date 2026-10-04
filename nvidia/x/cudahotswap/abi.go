package cudahotswap

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// ABIError reports incompatible kernel parameter lists. Parameter names and
// pointer intent are not part of this check: PTX encodes pointers as integers.
type ABIError struct {
	Baseline  cudalaunch.Signature
	Candidate cudalaunch.Signature
}

func (e *ABIError) Error() string {
	return fmt.Sprintf("cudahotswap: parameter ABI mismatch: baseline %s, candidate %s", e.Baseline, e.Candidate)
}

// CheckABI compares parameter counts, PTX types and byte sizes. Empty parameter
// lists are valid, but a parameter with an unknown type or size is refused.
func CheckABI(baseline, candidate cudalaunch.Signature) error {
	for _, sig := range []cudalaunch.Signature{baseline, candidate} {
		for i, p := range sig.Params {
			known := false
			switch p.Type {
			case "b8", "u8", "s8", "pred", "b16", "u16", "s16", "f16", "bf16", "b32", "u32", "s32", "f32", "tf32", "f16x2", "bf16x2", "b64", "u64", "s64", "f64", "b128":
				known = true
			}
			if !known || p.Size <= 0 {
				return fmt.Errorf("cudahotswap: unknown parameter ABI for %q at index %d", sig.KernelName, i)
			}
		}
	}
	match := len(baseline.Params) == len(candidate.Params)
	if match {
		for i, p := range baseline.Params {
			q := candidate.Params[i]
			if p.Type != q.Type || p.Size != q.Size {
				match = false
				break
			}
		}
	}
	if match {
		return nil
	}
	baseline.Params = append([]cudalaunch.Param(nil), baseline.Params...)
	candidate.Params = append([]cudalaunch.Param(nil), candidate.Params...)
	return &ABIError{Baseline: baseline, Candidate: candidate}
}
