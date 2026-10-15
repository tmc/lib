package cudasass_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestControlDecodeContract(t *testing.T) {
	known := cudasass.Encoding{Lo: 0x0000000607007c35, Hi: 0x041fe2000f8e0000}
	for _, tt := range []struct {
		name    string
		e       cudasass.Encoding
		arch    cudasass.Arch
		want    cudasass.ControlCode
		refused bool
		missing bool
	}{
		{"captured", known, cudasass.SM121a, cudasass.ControlCode{Stall: 1, Yield: true, WriteBarrier: 7, ReadBarrier: 7, WaitMask: 1, Reuse: 1}, false, false},
		{"known zero", cudasass.Encoding{Lo: 1}, cudasass.SM121a, cudasass.ControlCode{}, false, false},
		{"missing", cudasass.Encoding{}, cudasass.SM121a, cudasass.ControlCode{}, true, true},
		{"unsupported", known, "sm_90", cudasass.ControlCode{}, true, false},
		{"architecture first", cudasass.Encoding{}, "sm_90", cudasass.ControlCode{}, true, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cudasass.DecodeControl(tt.e, tt.arch)
			if got != tt.want || (err != nil) != tt.refused {
				t.Fatalf("DecodeControl = %v, %v; want %v, refused %v", got, err, tt.want, tt.refused)
			}
			if err == nil {
				back, err := cudasass.EncodeControl(tt.e, got, tt.arch)
				if err != nil || back != tt.e {
					t.Fatalf("round trip = %v, %v; want %v", back, err, tt.e)
				}
				return
			}
			checkControlDecodeError(t, fmt.Errorf("consumer: %w", err), tt.arch, tt.missing)
		})
	}
	for _, arch := range []cudasass.Arch{"", "sm_121", "sm_120a", "SM_121A", "sm_121a ", "sm_121a\x00", "unknown"} {
		t.Run(string(arch), func(t *testing.T) {
			for _, e := range []cudasass.Encoding{{}, known} {
				got, err := cudasass.DecodeControl(e, arch)
				if got != (cudasass.ControlCode{}) {
					t.Fatalf("refused decode returned %v", got)
				}
				checkControlDecodeError(t, err, arch, false)
			}
		})
	}
}

func checkControlDecodeError(t *testing.T, err error, arch cudasass.Arch, missing bool) {
	t.Helper()
	var refusal *cudasass.ControlDecodeError
	if !errors.As(err, &refusal) || refusal.Arch != arch || refusal.MissingEncoding != missing {
		t.Fatalf("error = %v; want ControlDecodeError{Arch:%q, MissingEncoding:%v}", err, arch, missing)
	}
	if errors.Is(err, cudaerrors.ErrUnsupported) || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("static decode refusal classified as hardware measurement: %v", err)
	}
}

func TestControlDecodeConsumerCauses(t *testing.T) {
	for _, tt := range []struct {
		name string
		arch cudasass.Arch
	}{
		{"missing", cudasass.SM121a},
		{"architecture first", "sm_90"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			missing := tt.arch == cudasass.SM121a
			a := cudasass.Instruction{Opcode: "NOP", Address: 0x20}
			_, err := cudasass.ReEncode(a, tt.arch)
			checkControlDecodeError(t, err, tt.arch, missing)
			_, _, err = cudasass.RecomputeControl(a, a, tt.arch)
			checkControlDecodeError(t, err, tt.arch, missing)
			_, err = cudasass.CheckMutation(&cudasass.Function{Instructions: []cudasass.Instruction{a}}, tt.arch, cudasass.Mutation{Index: 0}, nil)
			checkControlDecodeError(t, err, tt.arch, missing)
			if !missing {
				_, err = cudasass.Dependencies(cudasass.BasicBlock{}, tt.arch)
				checkControlDecodeError(t, err, tt.arch, false)
			}
		})
	}
}

func TestControlDecodeMissingBarrierStillSkipped(t *testing.T) {
	b := cudasass.BasicBlock{Instructions: []cudasass.Instruction{
		{Opcode: "IADD", Operands: "R4, R2, UR6"},
		{Opcode: "IADD", Operands: "R5, R4, UR7"},
	}}
	edges, err := cudasass.Dependencies(b, cudasass.SM121a)
	if err != nil {
		t.Fatal(err)
	}
	if len(edges) != 1 || edges[0].Kind != cudasass.RAW || edges[0].From != 0 || edges[0].To != 1 {
		t.Fatalf("missing encodings changed ordinary dependency decisions: %v", edges)
	}
}
