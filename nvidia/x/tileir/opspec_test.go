package tileir

import "testing"

// TestOpSpecShapes pins the shapes this package's encoder depends on, so a
// regenerated table that changes them fails here rather than at the assembler.
func TestOpSpecShapes(t *testing.T) {
	tests := []struct {
		mnemonic  string
		opcode    uint32
		results   int
		operands  int
		countPfx  bool
		segments  bool
		attrCount int
	}{
		{"cuda_tile.make_token", 0x44, 1, 0, false, false, 0},
		{"cuda_tile.constant", 0x10, 1, 0, false, false, 1},
		{"cuda_tile.return", 0x5C, 0, 1, true, false, 0},
		{"cuda_tile.store_view_tko", 0x66, 1, 4, true, true, 4},
		{"cuda_tile.load_view_tko", 0x3E, 2, 3, true, true, 4},
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
			if spec.NumAttributes != tt.attrCount {
				t.Errorf("NumAttributes = %d, want %d", spec.NumAttributes, tt.attrCount)
			}
		})
	}
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
