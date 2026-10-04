package cudahotswap

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestABIUsesPTXSignature(t *testing.T) {
	base, err := cudalaunch.SignatureFromPTX(`.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 x, .param .u32 n) { ret; }`, "k")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, params string
		mismatch     bool
	}{
		{"renamed", ".param .u64 p, .param .u32 count", false},
		{"count", ".param .u64 p", true},
		{"type", ".param .u64 p, .param .f32 count", true},
		{"size", ".param .u64 p, .param .u64 count", true},
		{"array", ".param .u64 p, .param .b8 count[4]", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			candidate, err := cudalaunch.SignatureFromPTX(fmt.Sprintf(".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry other(%s) { ret; }", tt.params), "other")
			if err != nil {
				t.Fatal(err)
			}
			err = CheckABI(base, candidate)
			var abi *ABIError
			if errors.As(err, &abi) != tt.mismatch {
				t.Fatalf("CheckABI = %v, mismatch want %v", err, tt.mismatch)
			}
			if !tt.mismatch && err != nil {
				t.Fatal(err)
			}
			if tt.mismatch {
				outcome, recognized := Classify(fmt.Errorf("push: %w", err))
				if outcome != OutcomeABIMismatch || !recognized {
					t.Fatalf("classification = %s/%v", outcome, recognized)
				}
				old := abi.Candidate.Params[0]
				candidate.Params[0].Name = "changed"
				if abi.Candidate.Params[0] != old {
					t.Fatal("error aliases caller signature")
				}
			}
		})
	}
}

func TestABIEmptyAndUnknown(t *testing.T) {
	if err := CheckABI(cudalaunch.Signature{}, cudalaunch.Signature{}); err != nil {
		t.Fatal(err)
	}
	for _, p := range []cudalaunch.Param{{Type: "u32"}, {Size: 4}, {Type: "u32", Size: -1}, {Type: "unknown", Size: 4}} {
		s := cudalaunch.Signature{KernelName: "k", Params: []cudalaunch.Param{p}}
		if CheckABI(s, s) == nil {
			t.Fatalf("unknown ABI admitted: %+v", p)
		}
	}
	base := cudalaunch.Signature{Params: []cudalaunch.Param{{Type: "u64", Size: 8, IsPointer: true}}}
	candidate := cudalaunch.Signature{Params: []cudalaunch.Param{{Type: "u64", Size: 8}}}
	if err := CheckABI(base, candidate); err != nil {
		t.Fatalf("PTX pointer intent affected ABI: %v", err)
	}
}
