package sassequiv

import "testing"

// TestClassify builds both sides' values by hand. The first side reads
// sel(p, v, old), the guarded write of a load v over old, where the
// second reads another value. When that value is one of the select's
// arms, the second side may read it only where the guard decides it, as
// ptxas arranges when it moves a use under the load's guard.
func TestClassify(t *testing.T) {
	const why = "guarded value read without its guard"
	tests := []struct {
		name    string
		other   string // "v", "old", or "w", an unrelated load
		verdict Verdict
		why     string
	}{
		{"value read under the guard", "v", Unknown, why},
		{"old value read under its negation", "old", Unknown, why},
		{"unrelated value", "w", Different, "values exchanged"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newGraph()
			var x, y int
			for side := range 2 {
				mem := g.add(side, 0, "in:MEM", 0)
				p := g.add(side, 0, "P0", 0)
				v := g.add(side, 0, "LDG.E", 0, mem, g.add(side, 0, "imm:0x10", 0))
				old := g.add(side, 0, "LDG.E", 0, mem, g.add(side, 0, "imm:0x20", 0))
				w := g.add(side, 0, "LDG.E", 0, mem, g.add(side, 0, "imm:0x30", 0))
				other := map[string]int{"v": v, "old": old, "w": w}[tt.other]
				s := g.add(side, 0, "sel", 0, p, v, old)
				one, two := g.add(side, 0, "imm:0x1", 0), g.add(side, 0, "imm:0x2", 0)
				a := g.add(side, 0, "IMNMX", 2, s, one)
				b := g.add(side, 0, "IMNMX", 2, other, one)
				// Each side also reads both values elsewhere.
				g.add(side, 0, "IMNMX", 2, s, two)
				g.add(side, 0, "IMNMX", 2, other, two)
				if side == 0 {
					x = a
				} else {
					y = b
				}
			}
			class := g.classes()
			d, ok := explain(g, class, x, y)
			if !ok {
				t.Fatal("explain found no difference")
			}
			v, why := classify(g, class, d)
			if v != tt.verdict || why != tt.why {
				t.Errorf("classify = %v %q, want %v %q", v, why, tt.verdict, tt.why)
			}
		})
	}
}
