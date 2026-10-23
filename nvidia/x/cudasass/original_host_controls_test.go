package cudasass_test

import (
	"bytes"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// TestCheckMutationRejectsReencodedByteMismatch exercises the original-byte
// guard with bytes produced by ReEncode. The negative arm injects a corrupt
// captured encoding; it does not claim that a tool emitted those bytes.
func TestCheckMutationRejectsReencodedByteMismatch(t *testing.T) {
	// Transcribed from the retained fixture's S2R at 0x10. The replacement
	// changes SR_TID.X to SR_TID.Y, preserving registers and control fields.
	original := cudasass.Encoding{Lo: 0x0000000000077919, Hi: 0x000e220000002100}
	replacement := cudasass.Encoding{Lo: 0x0000000000077919, Hi: 0x000e220000002200}
	for _, tt := range []struct {
		name    string
		corrupt bool
	}{
		{"unchanged capture", false},
		{"corrupt capture", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fn := roundTripFunction(t)
			index := instructionAt(t, fn, 0x10)
			if got := fn.Instructions[index].Encoding; got != original {
				t.Fatalf("captured encoding = %v, want %v", got, original)
			}
			if tt.corrupt {
				// Hi bit 0 is outside the decoded control fields and is not
				// emitted by the covered S2R form. Leave all text unchanged.
				fn.Instructions[index].Encoding.Hi ^= 1
			}
			wantFn := roundTripFunction(t)
			if tt.corrupt {
				wantFn.Instructions[index].Encoding.Hi ^= 1
			}
			t.Cleanup(func() {
				if !reflect.DeepEqual(fn, wantFn) {
					t.Error("encoder or mutation checker changed the input function")
				}
			})

			captured := fn.Instructions[index].Encoding
			encoded, err := cudasass.ReEncode(fn.Instructions[index], cudasass.SM121a)
			if err != nil {
				t.Fatal(err)
			}
			if encoded != original {
				t.Fatalf("rebuilt encoding = %v, want fixture bytes %v", encoded, original)
			}
			if tt.corrupt {
				if encoded.Lo^captured.Lo != 0 || encoded.Hi^captured.Hi != 1 {
					t.Fatalf("rebuilt encoding %v must differ from capture %v only at Hi bit 0", encoded, captured)
				}
			} else if encoded != captured {
				t.Fatalf("unchanged capture differs from rebuilt encoding: %v vs %v", captured, encoded)
			}

			mutation := cudasass.Mutation{
				Index:       index,
				Replacement: cudasass.Instruction{Opcode: "S2R", Operands: "R7, SR_TID.Y"},
			}
			plan, err := cudasass.CheckMutation(fn, cudasass.SM121a, mutation, nil)
			if tt.corrupt {
				if err == nil || !strings.Contains(err.Error(), "does not round-trip byte-identically") {
					t.Fatalf("CheckMutation error = %v, want original-byte mismatch refusal", err)
				}
				if plan != (cudasass.MutationPlan{}) {
					t.Fatalf("refused mutation returned a plan: %+v", plan)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if plan.Before != original || plan.After != replacement || plan.Before == plan.After {
				t.Fatalf("plan encodings = %v -> %v, want %v -> %v", plan.Before, plan.After, original, replacement)
			}
			beforeControl, err := cudasass.DecodeControl(plan.Before, cudasass.SM121a)
			if err != nil {
				t.Fatal(err)
			}
			afterControl, err := cudasass.DecodeControl(plan.After, cudasass.SM121a)
			if err != nil {
				t.Fatal(err)
			}
			if beforeControl != afterControl || plan.Control != beforeControl {
				t.Fatalf("control fields changed: before %+v, after %+v, plan %+v", beforeControl, afterControl, plan.Control)
			}
		})
	}
}

// TestDependencyModelRejectsRetainedDifferentSMFixture passes the target parsed
// from a retained SM90 listing to Dependencies. It checks the explicit arch
// contract, not ELF admission or a caller supplying a false architecture.
func TestDependencyModelRejectsRetainedDifferentSMFixture(t *testing.T) {
	// This unmodified listing was built from cudalowering/testdata/other.cu.
	// Its producer commands and provenance are in that directory's gen.sh
	// and cudalowering/lowering_test.go. No tool runs while reading it.
	t.Run("retained sm90", func(t *testing.T) {
		data, err := os.ReadFile("../cudalowering/testdata/other.sm90.sass")
		if err != nil {
			t.Fatal(err)
		}
		d, err := cudasass.Parse(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		if d.Target != "sm_90" {
			t.Fatalf("parsed target = %q, want sm_90", d.Target)
		}
		fn := d.Function("_Z9unrelatedPii")
		if fn == nil {
			t.Fatal("retained SM90 kernel is missing")
		}
		blocks := cudasass.BasicBlocks(fn)
		if len(blocks) == 0 || len(blocks[0].Instructions) == 0 {
			t.Fatal("retained SM90 kernel has no first basic block")
		}
		edges, err := cudasass.Dependencies(blocks[0], cudasass.Arch(d.Target))
		var refusal *cudasass.ControlDecodeError
		if !errors.As(err, &refusal) {
			t.Fatalf("Dependencies error = %v, want typed architecture refusal", err)
		}
		if refusal.Arch != "sm_90" || refusal.MissingEncoding {
			t.Fatalf("refusal = %+v, want sm_90 architecture refusal before encoding analysis", refusal)
		}
		if edges != nil {
			t.Fatalf("refused architecture returned edges: %v", edges)
		}
	})

	t.Run("retained sm121a", func(t *testing.T) {
		d := readRoundTripFixture(t)
		if d.Target != string(cudasass.SM121a) {
			t.Fatalf("parsed target = %q, want %s", d.Target, cudasass.SM121a)
		}
		fn := d.Function("intmix")
		if fn == nil {
			t.Fatal("retained SM121a kernel is missing")
		}
		blocks := cudasass.BasicBlocks(fn)
		if len(blocks) == 0 || len(blocks[0].Instructions) == 0 {
			t.Fatal("retained SM121a kernel has no first basic block")
		}
		edges, err := cudasass.Dependencies(blocks[0], cudasass.Arch(d.Target))
		if err != nil {
			t.Fatal(err)
		}
		if len(edges) == 0 {
			t.Fatal("retained SM121a first block produced no dependency edges")
		}
	})
}
