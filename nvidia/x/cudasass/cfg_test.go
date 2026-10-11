package cudasass_test

import (
	"bufio"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func parseFixture(t *testing.T, path string) *cudasass.Disassembly {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	d, err := cudasass.ParseString(string(data))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func fixtureFunction(t *testing.T, path, name string) *cudasass.Function {
	t.Helper()
	fn := parseFixture(t, path).Function(name)
	if fn == nil {
		t.Fatalf("no %s in %s", name, path)
	}
	return fn
}

func newCFG(t *testing.T, fn *cudasass.Function) *cudasass.CFG {
	t.Helper()
	g, err := cudasass.NewCFG(fn)
	if err != nil {
		t.Fatalf("NewCFG(%s): %v", fn.Name, err)
	}
	return g
}

func starts(g *cudasass.CFG) []uint64 {
	var s []uint64
	for _, b := range g.Blocks {
		s = append(s, b.Start)
	}
	return s
}

// fe abbreviates a FlowEdge in the tables below.
func fe(from, to int, kind cudasass.FlowKind) cudasass.FlowEdge {
	return cudasass.FlowEdge{From: from, To: to, Kind: kind}
}

const (
	ft    = cudasass.FallThrough
	taken = cudasass.Taken
	call  = cudasass.Call
)

// TestSASS_CFGRecovery_LinearAndBranching checks recovered graphs against
// edges derived by hand from the listings, not from the code under test.
//
// k4 in sm121a-loop-13.4.sass has conditional forward branches through
// uniform predicates (BRA.U !UP0), two backward loops that branch to their
// own block, a guarded EXIT that falls through, and the unconditional
// self-branch nvdisasm pads every function with.
func TestSASS_CFGRecovery_LinearAndBranching(t *testing.T) {
	tests := []struct {
		name       string
		fn         *cudasass.Function
		wantStarts []uint64
		wantEdges  []cudasass.FlowEdge
	}{
		{
			name: "linear intmix",
			fn:   fixtureFunction(t, roundTripFixture, "intmix"),
			// One block through EXIT, then the self-branch padding and the
			// NOPs after it.
			wantStarts: []uint64{0x000, 0x0d0, 0x0e0},
			wantEdges:  []cudasass.FlowEdge{fe(1, 1, taken)},
		},
		{
			name:       "guarded exit addk",
			fn:         fixtureFunction(t, roundTripFixture, "addk"),
			wantStarts: []uint64{0x000, 0x080, 0x140, 0x150},
			wantEdges:  []cudasass.FlowEdge{fe(0, 1, ft), fe(2, 2, taken)},
		},
		{
			name: "branching k4",
			fn:   fixtureFunction(t, "testdata/sm121a-loop-13.4.sass", "k4"),
			wantStarts: []uint64{
				0x000, 0x0b0, 0x100, 0x170, 0x3d0, 0x3f0, 0x420, 0x570, 0x580,
				0x5b0, 0x680, 0x690, 0x6c0, 0x730, 0x750, 0x790, 0x7a0,
			},
			wantEdges: []cudasass.FlowEdge{
				fe(0, 1, ft), fe(0, 13, taken), // 0x0a0 BRA.U !UP0 .L_x_0
				fe(1, 2, ft), fe(1, 5, taken), // 0x0f0 BRA.U !UP0 .L_x_1
				fe(2, 3, ft),                  // falls into the loop at .L_x_2
				fe(3, 3, taken), fe(3, 4, ft), // 0x3c0 BRA.U UP0 .L_x_2, backward
				fe(4, 5, ft), fe(4, 13, taken), // 0x3e0 BRA.U !UP0 .L_x_0
				fe(5, 6, ft), fe(5, 8, taken), // 0x410 BRA.U !UP0 .L_x_3
				fe(6, 7, ft), fe(6, 13, taken), // 0x560 BRA.U !UP0 .L_x_0
				fe(7, 8, ft),
				fe(8, 9, ft), fe(8, 11, taken), // 0x5a0 BRA.U !UP0 .L_x_4
				fe(9, 10, ft), fe(9, 13, taken), // 0x670 BRA.U !UP0 .L_x_0
				fe(10, 11, ft),
				fe(11, 12, ft),
				fe(12, 12, taken), fe(12, 13, ft), // 0x720 BRA.U UP0 .L_x_5, backward
				fe(13, 14, ft),    // 0x740 @P0 EXIT
				fe(15, 15, taken), // 0x790 BRA .L_x_6, padding
			},
		},
		{
			name: "call",
			fn: parseListing(t, "caller", `
        /*0000*/                   CALL.REL.NOINC `+"`"+`($caller$sub) ;
        /*0010*/                   EXIT ;
$caller$sub:
        /*0020*/                   MOV R0, 0x1 ;
        /*0030*/                   RET.REL.NODEC R20 `+"`"+`(caller) ;
`),
			wantStarts: []uint64{0x00, 0x10, 0x20},
			wantEdges:  []cudasass.FlowEdge{fe(0, 1, ft), fe(0, 2, call)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := newCFG(t, tt.fn)
			if got := starts(g); !slices.Equal(got, tt.wantStarts) {
				t.Fatalf("block starts:\n got %#x\nwant %#x", got, tt.wantStarts)
			}
			if !slices.Equal(g.Edges, tt.wantEdges) {
				t.Errorf("edges:\n got %v\nwant %v", g.Edges, tt.wantEdges)
			}
			if g.IsDegenerate || len(g.Indirect) != 0 {
				t.Errorf("IsDegenerate = %v, Indirect = %v; the function has no indirect branch", g.IsDegenerate, g.Indirect)
			}
			if !slices.EqualFunc(g.Blocks, cudasass.BasicBlocks(tt.fn), func(a, b cudasass.BasicBlock) bool {
				return a.Start == b.Start && a.End == b.End && len(a.Instructions) == len(b.Instructions)
			}) {
				t.Errorf("CFG blocks are not the BasicBlocks partition")
			}
		})
	}
}

// parseListing parses instruction lines as the body of one function.
func parseListing(t *testing.T, name, body string) *cudasass.Function {
	t.Helper()
	text := "\t.section\t.text." + name + ",\"ax\",@progbits\n" + name + ":\n.text." + name + ":\n" + body
	d, err := cudasass.ParseString(text)
	if err != nil {
		t.Fatal(err)
	}
	fn := d.Function(name)
	if fn == nil || len(fn.Instructions) == 0 {
		t.Fatalf("listing for %s parsed to no instructions", name)
	}
	return fn
}

// TestSASS_CFG_DegenerateIndirectBranch checks that BRX and JMX, whose targets
// come from registers, mark the graph degenerate and contribute no static
// edge, as NVBit's CFG_t.is_degenerate does (core/nvbit.h:114-117).
func TestSASS_CFG_DegenerateIndirectBranch(t *testing.T) {
	fn := parseListing(t, "ind", `
        /*0000*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;
        /*0010*/                   BRX R2 -0x20 ;
.L_x_0:
        /*0020*/                   MOV R3, 0x1 ;
        /*0030*/               @P0 BRX R4 -0x40 ;
        /*0040*/                   JMX R6 ;
.L_x_1:
        /*0050*/                   BRA `+"`"+`(.L_x_0) ;
        /*0060*/                   EXIT ;
`)
	g := newCFG(t, fn)
	if want := []uint64{0x00, 0x20, 0x40, 0x50, 0x60}; !slices.Equal(starts(g), want) {
		t.Fatalf("block starts %#x, want %#x", starts(g), want)
	}
	if !g.IsDegenerate {
		t.Error("IsDegenerate = false for a function with BRX and JMX")
	}
	if want := []int{0, 1, 2}; !slices.Equal(g.Indirect, want) {
		t.Errorf("Indirect = %v, want %v", g.Indirect, want)
	}
	// The unguarded BRX and JMX have no successor at all; the guarded BRX
	// only falls through. The one direct branch keeps its edge.
	want := []cudasass.FlowEdge{fe(1, 2, ft), fe(3, 1, taken)}
	if !slices.Equal(g.Edges, want) {
		t.Errorf("edges %v, want %v", g.Edges, want)
	}
	for _, e := range g.Edges {
		if slices.Contains(g.Indirect, e.From) && e.Kind != ft {
			t.Errorf("indirect block %d has a static %v edge to %d", e.From, e.Kind, e.To)
		}
	}
	dot := g.DOT()
	if !strings.Contains(dot, "degenerate: indirect branch in BB0, BB1, BB2") || strings.Count(dot, "style=dashed") != 3 {
		t.Errorf("DOT does not mark the degenerate graph:\n%s", dot)
	}

	// Adjacent control: a function with only direct branches is not
	// degenerate, so the flag is not simply always set.
	if g := newCFG(t, fixtureFunction(t, "testdata/sm121a-loop-13.4.sass", "k4")); g.IsDegenerate {
		t.Errorf("k4 has only direct branches but IsDegenerate = true, Indirect = %v", g.Indirect)
	}
}

func TestNewCFGRejectsUnresolvedTarget(t *testing.T) {
	fn := parseListing(t, "bad", `
        /*0000*/                   BRA `+"`"+`(.L_x_9) ;
        /*0010*/                   EXIT ;
`)
	if g, err := cudasass.NewCFG(fn); err == nil {
		t.Fatalf("NewCFG = %v, nil; want an error for the unresolved .L_x_9", g.Edges)
	}
	if _, err := cudasass.NewCFG(&cudasass.Function{Name: "empty"}); err == nil {
		t.Fatal("NewCFG of an empty function succeeded")
	}
}

// fixtureLabels reads label addresses from a listing independently of Parse:
// a line ".L_x_N:" names the address of the next instruction line, if one
// comes before the next function.
func fixtureLabels(t *testing.T, path string) map[string]uint64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	label := regexp.MustCompile(`^(\.L_x_\d+):$`)
	addr := regexp.MustCompile(`^/\*([0-9a-f]{4})\*/`)
	out := map[string]uint64{}
	var pending []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if m := label.FindStringSubmatch(line); m != nil {
			pending = append(pending, m[1])
			continue
		}
		if strings.HasPrefix(line, ".section") {
			pending = nil
		}
		if m := addr.FindStringSubmatch(line); m != nil {
			a, err := strconv.ParseUint(m[1], 16, 64)
			if err != nil {
				t.Fatal(err)
			}
			for _, l := range pending {
				out[l] = a
			}
			pending = nil
		}
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestSASS_BasicBlock_InstructionBoundaryAlignment checks that every label is
// a block boundary, that blocks tile the function on instruction boundaries,
// and that an operand which only looks like hex is not taken for an address.
func TestSASS_BasicBlock_InstructionBoundaryAlignment(t *testing.T) {
	tests := []struct {
		path, fn string
		notStart []uint64
	}{
		{path: "testdata/sm121a-loop-13.4.sass", fn: "k4"},
		// BSSY.RECONVERGENT B0 at 0x090: B0 is a barrier register, and 0x0b0
		// (an SHF) is no branch target.
		{path: "testdata/sm121a-reconvergence.sass", fn: "reconv", notStart: []uint64{0x0b0}},
	}
	for _, tt := range tests {
		t.Run(tt.fn, func(t *testing.T) {
			fn := fixtureFunction(t, tt.path, tt.fn)
			want := fixtureLabels(t, tt.path)
			if len(want) == 0 {
				t.Fatalf("no labels read from %s", tt.path)
			}
			if len(fn.Labels) != len(want) {
				t.Errorf("Labels has %d entries, fixture has %d: %v", len(fn.Labels), len(want), fn.Labels)
			}
			blocks := cudasass.BasicBlocks(fn)
			isStart := map[uint64]bool{}
			for _, b := range blocks {
				isStart[b.Start] = true
			}
			for l, a := range want {
				if got, ok := fn.Labels[l]; !ok || got != a {
					t.Errorf("Labels[%s] = %#x, %v; fixture says %#x", l, got, ok, a)
				}
				if !isStart[a] {
					t.Errorf("label %s at %#x does not start a block", l, a)
				}
			}
			for _, a := range tt.notStart {
				if isStart[a] {
					t.Errorf("%#x starts a block but no branch targets it", a)
				}
			}
			n := 0
			for i, b := range blocks {
				if b.Start%16 != 0 || b.End%16 != 0 || b.End <= b.Start {
					t.Errorf("block %d [%#x, %#x) is not aligned to instructions", i, b.Start, b.End)
				}
				if i > 0 && blocks[i-1].End != b.Start {
					t.Errorf("block %d starts at %#x, previous ends at %#x", i, b.Start, blocks[i-1].End)
				}
				for j, inst := range b.Instructions {
					if inst.Address != fn.Instructions[n].Address {
						t.Fatalf("block %d instruction %d is %#x, want %#x", i, j, inst.Address, fn.Instructions[n].Address)
					}
					if inst.Address < b.Start || inst.Address >= b.End {
						t.Errorf("block %d [%#x, %#x) holds %#x", i, b.Start, b.End, inst.Address)
					}
					if c, _ := cudasass.Classify(inst.Opcode); c.Branch && j != len(b.Instructions)-1 {
						t.Errorf("branch %s at %#x is not the last instruction of block %d", inst.Opcode, inst.Address, i)
					}
					n++
				}
			}
			if n != len(fn.Instructions) {
				t.Errorf("blocks hold %d instructions, function has %d", n, len(fn.Instructions))
			}
		})
	}
}

var (
	dotNode   = regexp.MustCompile(`(?s)"([^"]+)"\s*\[[^\[]*?label="(.*?)"\]`)
	dotOffset = regexp.MustCompile(`([0-9a-f]{4}):`)
	dotEdge   = regexp.MustCompile(`"([^"]+)"(?::\w+(?::\w+)?)?\s*->\s*"([^"]+)"`)
)

// dotStructure reads the block starts and edges of one graph from DOT text,
// either nvdisasm's or ours, naming each block by its first offset. Blocks
// beyond limit, and edges touching them, are dropped.
func dotStructure(t *testing.T, dot string, limit uint64) (blocks []uint64, edges [][2]uint64) {
	t.Helper()
	start := map[string]uint64{}
	for _, m := range dotNode.FindAllStringSubmatch(dot, -1) {
		offs := dotOffset.FindStringSubmatch(m[2])
		if offs == nil {
			continue
		}
		a, err := strconv.ParseUint(offs[1], 16, 64)
		if err != nil {
			t.Fatal(err)
		}
		if a <= limit {
			start[m[1]] = a
			blocks = append(blocks, a)
		}
	}
	for _, m := range dotEdge.FindAllStringSubmatch(dot, -1) {
		from, ok1 := start[m[1]]
		to, ok2 := start[m[2]]
		if ok1 && ok2 {
			edges = append(edges, [2]uint64{from, to})
		}
	}
	slices.Sort(blocks)
	slices.SortFunc(edges, func(a, b [2]uint64) int {
		if a[0] != b[0] {
			return int(a[0]) - int(b[0])
		}
		return int(a[1]) - int(b[1])
	})
	return blocks, edges
}

// nvdisasmCluster returns the part of nvdisasm's DOT for one function.
func nvdisasmCluster(t *testing.T, dot, fn string) string {
	t.Helper()
	cluster := `subgraph "cluster_` + fn + `" {`
	i := strings.Index(dot, cluster)
	if i < 0 {
		t.Fatalf("no cluster for %s", fn)
	}
	rest := dot[i+len(cluster):]
	if j := strings.Index(rest, `subgraph "cluster_`); j >= 0 {
		rest = rest[:j]
	}
	return rest
}

// TestSASS_DOTExport_GraphvizStructure compares the rendered CFG with
// nvdisasm's basic-block graph of the same functions: the committed one for
// the fixture, and, where nvdisasm is installed, one produced now by
// [cudasass.DisassembleCFG]. nvdisasm stops at the function's last EXIT, so
// the padding after it is left out of the comparison.
func TestSASS_DOTExport_GraphvizStructure(t *testing.T) {
	d := readRoundTripFixture(t)
	committed, err := os.ReadFile("testdata/roundtrip/sm121a-roundtrip.bbcfg.dot")
	if err != nil {
		t.Fatal(err)
	}
	compare := func(t *testing.T, nv string) {
		for _, name := range []string{"addk", "intmix"} {
			t.Run(name, func(t *testing.T) {
				fn := d.Function(name)
				g := newCFG(t, fn)
				dot := g.DOT()
				_, last := nvdisasmLeaders(t, nv, name)
				wantBlocks, wantEdges := dotStructure(t, nvdisasmCluster(t, nv, name), last)
				gotBlocks, gotEdges := dotStructure(t, dot, last)
				if len(wantBlocks) == 0 {
					t.Fatal("no blocks read from nvdisasm's graph")
				}
				if !slices.Equal(gotBlocks, wantBlocks) {
					t.Errorf("blocks %#x, nvdisasm %#x", gotBlocks, wantBlocks)
				}
				if !slices.Equal(gotEdges, wantEdges) {
					t.Errorf("edges %#x, nvdisasm %#x", gotEdges, wantEdges)
				}
				if names := cudasass.ListDOTGraphs(dot); !slices.Equal(names, []string{name}) {
					t.Errorf("ListDOTGraphs = %q, want [%s]", names, name)
				}
				for _, inst := range fn.Instructions {
					if off := strconv.FormatUint(inst.Address+0x10000, 16)[1:] + ": "; !strings.Contains(dot, off) {
						t.Errorf("DOT has no line for %s", off)
					}
				}
			})
		}
	}
	t.Run("committed", func(t *testing.T) { compare(t, string(committed)) })
	t.Run("nvdisasm", func(t *testing.T) {
		tool, err := exec.LookPath(cudasass.ToolName)
		if err != nil {
			t.Skipf("%s not found in PATH; the live comparison with DisassembleCFG did not run", cudasass.ToolName)
		}
		// The fixture is an SM121a CUBIN.
		skipUnlessBinaryArch(t, tool, fixtureArch)
		nv, err := cudasass.DisassembleCFG(readRoundTripCubin(t), true)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, nv)
	})
}
