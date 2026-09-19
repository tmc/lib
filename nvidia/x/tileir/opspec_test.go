package tileir

import (
	"slices"
	"testing"
)

// TestOpSpecShapes pins the shapes this package's encoder depends on, so a
// regenerated table that changes them fails here rather than at the assembler.
//
// The flags column is the one that was missing. NVIDIA's writer emits a flags
// varint between the result types and the attributes for every operation with
// an optional field, and a UnitAttr is encoded as nothing but its bit, so a
// table that reported only attribute counts described a stream the assembler
// does not read. wantFlags names the bits in bit order, which is version
// order and not declaration order.
func TestOpSpecShapes(t *testing.T) {
	tests := []struct {
		mnemonic  string
		opcode    uint32
		results   int
		operands  int
		countPfx  bool
		segments  bool
		attrCount int
		wantFlags []string
	}{
		{"cuda_tile.make_token", 0x44, 1, 0, false, false, 0, nil},
		{"cuda_tile.constant", 0x10, 1, 0, false, false, 1, nil},
		{"cuda_tile.return", 0x5C, 0, 1, true, false, 0, nil},
		// addf's flush_to_zero is a UnitAttr: one bit, no payload.
		{"cuda_tile.addf", 0x02, 1, 2, false, false, 2, []string{"flush_to_zero"}},
		// The two view operations carry no list-wide operand count: under
		// AttrSizedOperandSegments each group is written on its own.
		{"cuda_tile.store_view_tko", 0x66, 1, 4, false, true, 4,
			[]string{"memory_scope", "optimization_hints", "token"}},
		{"cuda_tile.load_view_tko", 0x3E, 2, 3, false, true, 4,
			[]string{"memory_scope", "optimization_hints", "token"}},
	}
	for _, tt := range tests {
		t.Run(tt.mnemonic, func(t *testing.T) {
			spec, ok := lookupOpSpec(tt.mnemonic)
			if !ok {
				t.Fatalf("absent from the generated table")
			}
			if spec.Opcode != tt.opcode {
				t.Errorf("Opcode = 0x%02X, want 0x%02X", spec.Opcode, tt.opcode)
			}
			if spec.NumResults != tt.results {
				t.Errorf("NumResults = %d, want %d", spec.NumResults, tt.results)
			}
			if len(spec.Operands) != tt.operands {
				t.Errorf("len(Operands) = %d, want %d", len(spec.Operands), tt.operands)
			}
			if spec.EncodeOperandCount != tt.countPfx {
				t.Errorf("EncodeOperandCount = %v, want %v", spec.EncodeOperandCount, tt.countPfx)
			}
			if spec.AttrSegments != tt.segments {
				t.Errorf("AttrSegments = %v, want %v", spec.AttrSegments, tt.segments)
			}
			if len(spec.Attributes) != tt.attrCount {
				t.Errorf("len(Attributes) = %d, want %d", len(spec.Attributes), tt.attrCount)
			}
			if spec.AttrSegments && spec.EncodeOperandCount {
				t.Error("EncodeOperandCount is set alongside AttrSegments; the writer takes one branch or the other")
			}
			var gotFlags []string
			for _, b := range spec.Flags {
				gotFlags = append(gotFlags, b.Name)
			}
			if !slices.Equal(gotFlags, tt.wantFlags) {
				t.Errorf("Flags = %v, want %v", gotFlags, tt.wantFlags)
			}
			if spec.HasFlags() != (len(tt.wantFlags) > 0) {
				t.Errorf("HasFlags = %v, want %v", spec.HasFlags(), len(tt.wantFlags) > 0)
			}
		})
	}
}

// TestFlagBitsIndexTheirOwnField requires every flags bit to point at a field
// that exists and is optional.
//
// A bit that indexes out of range, or names an attribute the writer always
// emits, would make the encoder write a payload the reader is not expecting
// at a position the reader does not look -- and neither the table nor a
// round trip against our own decoder would say so.
func TestFlagBitsIndexTheirOwnField(t *testing.T) {
	checked := 0
	for name, spec := range opSpecs {
		for _, b := range spec.Flags {
			checked++
			if b.Operand {
				if b.Index < 0 || b.Index >= len(spec.Operands) {
					t.Errorf("%s: flag %q indexes operand %d of %d", name, b.Name, b.Index, len(spec.Operands))
					continue
				}
				if spec.Operands[b.Index] != operandOptional {
					t.Errorf("%s: flag %q names operand %d, which is not optional", name, b.Name, b.Index)
				}
				if !spec.AttrSegments {
					t.Errorf("%s: flag %q is an operand bit without AttrSizedOperandSegments", name, b.Name)
				}
				continue
			}
			if b.Index < 0 || b.Index >= len(spec.Attributes) {
				t.Errorf("%s: flag %q indexes attribute %d of %d", name, b.Name, b.Index, len(spec.Attributes))
				continue
			}
			attr := spec.Attributes[b.Index]
			if attr.Name != b.Name {
				t.Errorf("%s: flag %q indexes attribute %q", name, b.Name, attr.Name)
			}
			if !attr.Optional {
				t.Errorf("%s: flag %q names a required attribute", name, b.Name)
			}
		}
		for i, a := range spec.Attributes {
			if a.Unit && !a.Optional {
				t.Errorf("%s: attribute %q is a unit attribute but not optional", name, a.Name)
			}
			if !a.Optional {
				continue
			}
			found := false
			for _, b := range spec.Flags {
				if !b.Operand && b.Index == i {
					found = true
				}
			}
			if !found {
				t.Errorf("%s: optional attribute %q owns no flags bit", name, a.Name)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no operation in the table has a flags bit; the test is vacuous")
	}
	t.Logf("checked %d flags bits across %d operations", checked, len(opSpecs))
}

// TestOpSpecTableIsPopulated guards against a generator that silently emits an
// empty or truncated table.
func TestOpSpecTableIsPopulated(t *testing.T) {
	if len(opSpecs) < 100 {
		t.Fatalf("generated table holds %d operations, want at least 100", len(opSpecs))
	}
}

// TestAuditNamesExist requires every operation named by an audit rule to be a
// real Tile IR operation.
//
// Six of the twelve names here did not exist: the rules said mma_f, add_f and
// mul_f where the dialect spells them mmaf, addf and mulf. A forbid-rule naming
// an operation that cannot occur never fires, so a kernel that really did fall
// back to unfused arithmetic passed the audit and reported clean.
func TestAuditNamesExist(t *testing.T) {
	rules := map[string][]string{
		"ForbidUnfusedFallbacks": ForbidUnfusedFallbacks,
		"RequireTMAOperations":   RequireTMAOperations,
		"RequireTensorCoreMMA":   RequireTensorCoreMMA,
	}
	checked := 0
	for rule, names := range rules {
		if len(names) == 0 {
			t.Errorf("%s: rule is empty, so it asserts nothing", rule)
			continue
		}
		for _, name := range names {
			if _, ok := lookupOpSpec(name); !ok {
				t.Errorf("%s: %q is not a Tile IR operation", rule, name)
			}
			checked++
		}
	}
	if checked == 0 {
		t.Fatal("checked no names; the test is vacuous")
	}
}
