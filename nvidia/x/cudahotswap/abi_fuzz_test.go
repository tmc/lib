package cudahotswap

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// The oracle compares concrete lists without invoking CheckABI or the PTX
// parser. Renaming and pointer-intent changes never alter an accepted list.
func FuzzCheckABI(f *testing.F) {
	f.Add([]byte{0, 1, 2}, []byte{0, 1, 2})
	f.Add([]byte{0, 1}, []byte{1, 0})
	f.Add([]byte{}, []byte{0})
	f.Fuzz(func(t *testing.T, left, right []byte) {
		if len(left) > 32 || len(right) > 32 {
			t.Skip()
		}
		types := []string{"u32", "u64", "f32", "b8"}
		sizes := []int{4, 8, 4, 1}
		build := func(b []byte, pointer bool) cudalaunch.Signature {
			s := cudalaunch.Signature{KernelName: "k"}
			for _, v := range b {
				i := int(v) % len(types)
				s.Params = append(s.Params, cudalaunch.Param{Type: types[i], Size: sizes[i], IsPointer: pointer})
			}
			return s
		}
		wantMismatch := len(left) != len(right)
		if !wantMismatch {
			for i := range left {
				if left[i]%4 != right[i]%4 {
					wantMismatch = true
				}
			}
		}
		err := CheckABI(build(left, true), build(right, false))
		var abi *ABIError
		if errors.As(err, &abi) != wantMismatch || (!wantMismatch && err != nil) {
			t.Fatalf("CheckABI = %v, mismatch want %v", err, wantMismatch)
		}
	})
}
