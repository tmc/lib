package tileir_test

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func attentionGraph(t testing.TB, n, d int) *cudair.Snapshot {
	return attentionGraphEdit(t, n, d, cudair.Float32, nil)
}

func attentionGraphEdit(t testing.TB, n, d int, dtype cudair.DataType, edit func([]*cudair.Node)) *cudair.Snapshot {
	t.Helper()
	g := cudair.NewGraph()
	s := g.AddInput("scores", []int{n}, dtype)
	v := g.AddInput("values", []int{n, d}, dtype)
	maximum := g.AddReduce("max", s, 0)
	shift := g.AddBinary("sub", s, maximum)
	e := g.AddUnary("exp", shift)
	sum := g.AddReduce("sum", e, 0)
	p := g.AddBinary("div", e, sum)
	reshape := g.AddReshape(p, []int{1, n})
	out := g.AddMatMul(reshape, v)
	out.SourceLoc = cudair.SourceLocation{File: "attention_fixture.go", Line: 12}
	if edit != nil {
		edit([]*cudair.Node{s, v, maximum, shift, e, sum, p, reshape, out})
	}
	if err := g.SetOutputs(out); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return snap
}

func attentionCandidate(t testing.TB, n, d, tile int, mask *egraph.AttentionRowMask) (*egraph.OnlineAttentionCandidate, tileir.AttentionOptions) {
	t.Helper()
	s, v := make([]float32, n), make([]float32, n*d)
	for k := range s {
		s[k] = float32(k%7)*1.25 - 4
	}
	for i := range v {
		v[i] = float32(i%17-8) / 3
	}
	opts := tileir.AttentionOptions{TileSize: tile, Absolute: 2e-5, Relative: 2e-5}
	inputs := map[string][]float32{"scores": s, "values": v}
	policy := egraph.AttentionCandidatePolicy{Absolute: opts.Absolute, Relative: opts.Relative}
	var c *egraph.OnlineAttentionCandidate
	var err error
	if mask == nil {
		c, err = egraph.NewCheckedOnlineAttentionCandidate(attentionGraph(t, n, d), inputs, tile, policy)
	} else {
		opts.Excluded = mask.Dense()
		for k, excluded := range opts.Excluded {
			if excluded {
				s[k] = float32(math.Inf(-1))
			}
		}
		c, err = egraph.NewCheckedMaskedOnlineAttentionCandidate(attentionGraph(t, n, d), inputs, *mask, tile, policy)
	}
	if err != nil {
		t.Fatal(err)
	}
	return c, opts
}

// materializedAttention is independent of the online recurrence and IR builder.
func materializedAttention(scores, values []float32, d int) []float32 {
	m := math.Inf(-1)
	for _, s := range scores {
		m = math.Max(m, float64(s))
	}
	out := make([]float32, d)
	if math.IsInf(m, -1) {
		for j := range out {
			out[j] = math.Float32frombits(0x7fc00000)
		}
		return out
	}
	p, sum := make([]float64, len(scores)), 0.0
	for k, s := range scores {
		p[k] = math.Exp(float64(s) - m)
		sum += p[k]
	}
	for j := range out {
		x := 0.0
		for k := range scores {
			x += p[k] / sum * float64(values[k*d+j])
		}
		out[j] = float32(x)
	}
	return out
}

func TestAttentionTypedLowering(t *testing.T) {
	for _, shape := range [][2]int{{1, 1}, {2, 3}, {7, 5}, {64, 64}} {
		for _, tile := range []int{1, min(3, shape[0]), shape[0]} {
			t.Run(fmt.Sprint(shape, tile), func(t *testing.T) {
				c, opts := attentionCandidate(t, shape[0], shape[1], tile, nil)
				checkAttention(t, c, opts)
			})
		}
	}
	causal, _ := egraph.NewCausalAttentionRowMask(2, 7)
	prefix, _ := egraph.NewPrefixAttentionRowMask(4, 7)
	all, _ := egraph.NewPrefixAttentionRowMask(0, 7)
	tree, _ := egraph.NewTreeAttentionRowMask([]int{-1, 0, 0, 2, 3}, 4, 2)
	for _, mask := range []egraph.AttentionRowMask{causal, prefix, all, tree} {
		t.Run(mask.Kind()+fmt.Sprint(mask.Dense()), func(t *testing.T) {
			for _, tile := range []int{1, 3, 7} {
				c, opts := attentionCandidate(t, 7, 3, tile, &mask)
				checkAttention(t, c, opts)
			}
		})
	}
}

func checkAttention(t *testing.T, c *egraph.OnlineAttentionCandidate, opts tileir.AttentionOptions) {
	t.Helper()
	a, err := tileir.LowerOnlineAttention(c, opts)
	if err != nil {
		t.Fatal(err)
	}
	r := a.Result()
	if err := a.Verify(); err != nil {
		t.Fatal(err)
	}
	if r.Source != "" || r.Grid != [3]uint32{1, 1, 1} || len(r.Schema.Params) != 3 || r.Schema.TotalParamBytes != 24 {
		t.Fatal("not a three-pointer standalone typed lowering")
	}
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tileir.DecodeBytecodeBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	// Execute the decoded operations, not the builder's separate host check.
	inputs := c.Inputs()
	d := c.Snapshot().Outputs()[0].Shape[1]
	want := materializedAttention(inputs["scores"], inputs["values"], d)
	got := executeAttention(t, decoded.Functions[0].Operations, r.Module.Functions[0].Params, inputs, d)
	assertAttention(t, got, want)
	if a.Identity() == "" || a.Identity() == c.Identity() {
		t.Fatal("lowering identity does not bind its recipe")
	}
	refs := r.Memory.References()
	ids := make(map[int]bool)
	located := false
	for _, ref := range refs {
		ids[ref.NodeID] = true
		located = located || ref.Location.File == "attention_fixture.go" && ref.Location.Line == 12
	}
	if len(ids) != 9 || !located {
		t.Fatal("semantic IDs or source locations lost")
	}
	copy := a.Inputs()
	copy["values"][0]++
	if a.Inputs()["values"][0] == copy["values"][0] {
		t.Fatal("input accessor aliases retained binding")
	}
	r.Grid[0]++
	if a.Verify() == nil {
		t.Fatal("changed launch grid retained admission")
	}
}

func assertAttention(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("output extent")
	}
	for i, v := range got {
		if math.Float32bits(v) == math.Float32bits(want[i]) {
			continue
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.IsNaN(float64(want[i])) || math.Abs(float64(v-want[i])) > 2e-5*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("output[%d] = %g, want %g", i, v, want[i])
		}
	}
}

type attentionOverride struct {
	tileir.AttentionCandidate
	inputs map[string][]float32
	output []float32
	snap   *cudair.Snapshot
}

func (c attentionOverride) Inputs() map[string][]float32 {
	if c.inputs != nil {
		return c.inputs
	}
	return c.AttentionCandidate.Inputs()
}
func (c attentionOverride) Output() []float32 {
	if c.output != nil {
		return c.output
	}
	return c.AttentionCandidate.Output()
}
func (c attentionOverride) Snapshot() *cudair.Snapshot {
	if c.snap != nil {
		return c.snap
	}
	return c.AttentionCandidate.Snapshot()
}

func TestAttentionRefusals(t *testing.T) {
	c, opts := attentionCandidate(t, 7, 3, 3, nil)
	for _, test := range []struct {
		name string
		edit func(*tileir.AttentionOptions)
	}{
		{"zero tile", func(o *tileir.AttentionOptions) { o.TileSize = 0 }},
		{"large tile", func(o *tileir.AttentionOptions) { o.TileSize = 8 }},
		{"short mask", func(o *tileir.AttentionOptions) { o.Excluded = []bool{false} }},
		{"invented mask", func(o *tileir.AttentionOptions) { o.Excluded = []bool{true, false, false, false, false, false, false} }},
		{"NaN policy", func(o *tileir.AttentionOptions) { o.Absolute = math.NaN() }},
		{"infinite policy", func(o *tileir.AttentionOptions) { o.Relative = math.Inf(1) }},
		{"negative policy", func(o *tileir.AttentionOptions) { o.Absolute = -1 }},
		{"overflowing policy", func(o *tileir.AttentionOptions) { o.Absolute, o.Relative = math.MaxFloat64, math.MaxFloat64 }},
		{"identifier", func(o *tileir.AttentionOptions) { o.KernelName = "bad-name" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			o := opts
			test.edit(&o)
			if got, err := tileir.LowerOnlineAttention(c, o); got != nil || err == nil {
				t.Fatal("invalid attention options admitted")
			}
		})
	}
	if got, err := tileir.LowerOnlineAttention(nil, opts); got != nil || err == nil {
		t.Fatal("nil candidate admitted")
	}
	var nilCandidate *egraph.OnlineAttentionCandidate
	if got, err := tileir.LowerOnlineAttention(nilCandidate, opts); got != nil || err == nil {
		t.Fatal("typed nil candidate admitted")
	}
	for _, what := range []string{"output", "inputs", "graph"} {
		t.Run("forged "+what, func(t *testing.T) {
			fake := attentionOverride{AttentionCandidate: c}
			switch what {
			case "output":
				fake.output = c.Output()
				fake.output[1] += 100
			case "inputs":
				fake.inputs = c.Inputs()
				fake.inputs["values"][1] = float32(math.NaN())
			case "graph":
				g := cudair.NewGraph()
				g.AddUnary("exp", g.AddInput("x", []int{7}, cudair.Float32))
				fake.snap, _ = g.Snapshot()
			}
			if got, err := tileir.LowerOnlineAttention(fake, opts); got != nil || !errors.Is(err, tileir.ErrUnsupportedTileOp) {
				t.Fatalf("forged candidate accepted: %v", err)
			}
		})
	}
	mask, _ := egraph.NewCausalAttentionRowMask(2, 7)
	masked, maskedOpts := attentionCandidate(t, 7, 3, 3, &mask)
	maskedOpts.Excluded = nil
	if got, err := tileir.LowerOnlineAttention(masked, maskedOpts); got != nil || err == nil {
		t.Fatal("masked candidate silently lowered as dense")
	}
}

func TestAttentionBorrowedInputMap(t *testing.T) {
	c, opts := attentionCandidate(t, 7, 3, 3, nil)
	borrowed := c.Inputs()
	original := slices.Clone(borrowed["values"])
	wrapped := attentionOverride{AttentionCandidate: c, inputs: borrowed}
	a, err := tileir.LowerOnlineAttention(wrapped, opts)
	if err != nil {
		t.Fatal(err)
	}
	identity := a.Identity()
	borrowed["values"][0]++
	borrowed["values"] = []float32{999}
	delete(borrowed, "scores")
	if !slices.Equal(a.Inputs()["values"], original) || len(a.Inputs()["scores"]) != 7 || a.Identity() != identity || a.Verify() != nil {
		t.Fatal("caller map or slices alias the lowering")
	}
}

func TestAttentionCanonicalGraphGate(t *testing.T) {
	c, opts := attentionCandidate(t, 7, 3, 3, nil)
	for _, test := range []struct {
		name  string
		n, d  int
		dtype cudair.DataType
		edit  func([]*cudair.Node)
	}{
		{"key bound", 65, 3, cudair.Float32, nil},
		{"output bound", 7, 65, cudair.Float32, nil},
		{"dtype", 7, 3, cudair.Float16, nil},
		{"other maximum", 7, 3, cudair.Float32, func(nodes []*cudair.Node) { nodes[2].Op = "sum" }},
		{"other shift", 7, 3, cudair.Float32, func(nodes []*cudair.Node) { nodes[3].Op = "add" }},
		{"other exponential", 7, 3, cudair.Float32, func(nodes []*cudair.Node) { nodes[4].Op = "relu" }},
		{"wrong denominator", 7, 3, cudair.Float32, func(nodes []*cudair.Node) { nodes[5].Inputs[0] = nodes[0] }},
		{"wrong probability", 7, 3, cudair.Float32, func(nodes []*cudair.Node) { nodes[6].Op = "mul" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			snap := attentionGraphEdit(t, test.n, test.d, test.dtype, test.edit)
			fake := attentionOverride{AttentionCandidate: c, snap: snap}
			if got, err := tileir.LowerOnlineAttention(fake, opts); got != nil || !errors.Is(err, tileir.ErrUnsupportedTileOp) {
				t.Fatalf("noncanonical graph admitted: %v", err)
			}
		})
	}
}

func TestAttentionOmissionControls(t *testing.T) {
	for _, mutant := range []string{"inverse rescale", "normalizer", "value operand", "fully masked policy"} {
		t.Run(mutant, func(t *testing.T) {
			var mask *egraph.AttentionRowMask
			if mutant == "fully masked policy" {
				m, _ := egraph.NewPrefixAttentionRowMask(0, 7)
				mask = &m
			}
			c, opts := attentionCandidate(t, 7, 3, 3, mask)
			a, err := tileir.LowerOnlineAttention(c, opts)
			if err != nil {
				t.Fatal(err)
			}
			r := a.Result()
			ops := r.Module.Functions[0].Operations
			changed := 0
			for _, op := range ops {
				switch {
				case mutant == "normalizer" && op.OpCode == "cuda_tile.divf":
					// Keep a type-correct divide but replace its denominator by one.
					for _, constant := range ops {
						if constant.OpCode == "cuda_tile.constant" && constant.Attributes == "{value = 1.0}" {
							op.Operands[1] = constant.Results[0]
							changed++
							break
						}
					}
				case mutant == "inverse rescale" && op.OpCode == "cuda_tile.exp":
					adds, multiplies := 0, 0
					for _, use := range ops {
						for _, operand := range use.Operands {
							if operand == op.Results[0] {
								if use.OpCode == "cuda_tile.addf" {
									adds++
								}
								if use.OpCode == "cuda_tile.mulf" {
									multiplies++
								}
							}
						}
					}
					if adds == 0 && multiplies == 2 {
						for _, shift := range ops {
							if len(shift.Results) == 1 && shift.Results[0] == op.Operands[0] && shift.OpCode == "cuda_tile.subf" {
								shift.Operands[0], shift.Operands[1] = shift.Operands[1], shift.Operands[0]
								changed++
							}
						}
					}
				case mutant == "fully masked policy" && op.OpCode == "cuda_tile.constant" && strings.Contains(op.Attributes, "0x0000c07f"):
					op.Attributes = "{value = 0.0}"
					changed++
				}
			}
			if mutant == "value operand" {
				origin := map[string]int{"%arg0": 0, "%arg1": 1, "%arg2": 2}
				scorePointer := ""
				for _, op := range ops {
					if op.OpCode == "cuda_tile.reshape" || op.OpCode == "cuda_tile.broadcast" || op.OpCode == "cuda_tile.offset" {
						origin[op.Results[0]] = origin[op.Operands[0]]
					}
					if op.OpCode == "cuda_tile.load_ptr_tko" {
						if origin[op.Operands[0]] == 0 {
							scorePointer = op.Operands[0]
						} else if origin[op.Operands[0]] == 1 && scorePointer != "" {
							op.Operands[0] = scorePointer
							changed++
						}
					}
				}
			}
			if changed == 0 || a.Verify() == nil {
				t.Fatal("omission was not applied or source binding failed to detect it")
			}
			data, err := r.Module.EncodeBytecode()
			if err != nil {
				t.Fatalf("omission must remain well-typed: %v", err)
			}
			decoded, err := tileir.DecodeBytecodeBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			inputs := c.Inputs()
			got := executeAttention(t, decoded.Functions[0].Operations, r.Module.Functions[0].Params, inputs, 3)
			want := materializedAttention(inputs["scores"], inputs["values"], 3)
			different := false
			for i, v := range got {
				if math.Float32bits(v) == math.Float32bits(want[i]) {
					continue
				}
				if math.IsNaN(float64(v)) || math.IsNaN(float64(want[i])) || math.Abs(float64(v-want[i])) > 2e-5*(1+math.Abs(float64(want[i]))) {
					different = true
				}
			}
			if !different {
				t.Fatal("independent materialized oracle failed to distinguish well-typed omission")
			}
		})
	}
}

type attentionPointer struct{ arg, offset int }
type attentionValue struct {
	n []float64
	p []attentionPointer
}

// executeAttention interprets only emitted typed opcodes. It is a test oracle,
// not a production fallback, compiler model or GPU execution claim.
func executeAttention(t *testing.T, ops []*tileir.Operation, params []*tileir.Param, inputs map[string][]float32, d int) []float32 {
	t.Helper()
	mem := [][]float32{slices.Clone(inputs["scores"]), slices.Clone(inputs["values"]), make([]float32, d)}
	vs := make(map[string]attentionValue)
	for i, p := range params {
		vs[p.Name] = attentionValue{p: []attentionPointer{{arg: i}}}
	}
	for _, op := range ops {
		code := strings.TrimPrefix(op.OpCode, "cuda_tile.")
		args := make([]attentionValue, len(op.Operands))
		for i, name := range op.Operands {
			args[i] = vs[name]
		}
		var out attentionValue
		typ, _ := op.TypeVal.(*tileir.TileType)
		width := 1
		if typ != nil && len(typ.Shape) == 1 {
			width = int(typ.Shape[0])
		}
		switch code {
		case "return":
			return mem[2]
		case "make_token":
			out.n = []float64{0}
		case "constant":
			text := strings.TrimSuffix(strings.TrimPrefix(op.Attributes, "{value = "), "}")
			var v float64
			if strings.HasPrefix(text, "dense<\"0x") {
				data, err := hex.DecodeString(strings.TrimSuffix(strings.TrimPrefix(text, "dense<\"0x"), "\">"))
				if err != nil {
					t.Fatal(err)
				}
				switch typ.ElementType.Kind() {
				case tileir.KindFloat32:
					v = float64(math.Float32frombits(binary.LittleEndian.Uint32(data)))
				case tileir.KindFloat64:
					v = math.Float64frombits(binary.LittleEndian.Uint64(data))
				case tileir.KindInt1:
					v = float64(data[0] & 1)
				default:
					v = float64(binary.LittleEndian.Uint64(data))
				}
			} else {
				var err error
				v, err = strconv.ParseFloat(text, 64)
				if err != nil {
					t.Fatal(err)
				}
			}
			out.n = make([]float64, width)
			for i := range out.n {
				out.n[i] = v
			}
		case "reshape", "broadcast":
			if len(args[0].p) != 0 {
				out.p = make([]attentionPointer, width)
				for i := range out.p {
					out.p[i] = args[0].p[0]
				}
			} else {
				t.Fatal("unexpected numeric shape op")
			}
		case "iota":
			out.n = make([]float64, width)
			for i := range out.n {
				out.n[i] = float64(i)
			}
		case "offset":
			out.p = slices.Clone(args[0].p)
			for i := range out.p {
				out.p[i].offset += int(args[1].n[i])
			}
		case "load_ptr_tko":
			out.n = slices.Clone(args[2].n)
			for i, p := range args[0].p {
				if args[1].n[i] != 0 {
					if p.offset < 0 || p.offset >= len(mem[p.arg]) {
						t.Fatal("out-of-bounds typed load")
					}
					out.n[i] = float64(mem[p.arg][p.offset])
				}
			}
			vs[op.Results[1]] = attentionValue{n: []float64{0}}
		case "store_ptr_tko":
			for i, p := range args[0].p {
				if args[2].n[i] != 0 {
					if p.arg != 2 || p.offset < 0 || p.offset >= d {
						t.Fatal("out-of-bounds typed store")
					}
					mem[p.arg][p.offset] = float32(args[1].n[i])
				}
			}
			out.n = []float64{0}
		default:
			out.n = make([]float64, len(args[0].n))
			for i, a := range args[0].n {
				v := a
				switch code {
				case "ftof":
					if typ.ElementType.Kind() == tileir.KindFloat32 {
						v = float64(float32(a))
					}
				case "exp":
					v = math.Exp(a)
				case "cmpi", "cmpf":
					ok := a > args[1].n[i]
					if strings.Contains(op.Attributes, "less_than") {
						ok = a < args[1].n[i]
					}
					v = 0
					if ok {
						v = 1
					}
				case "select":
					v = args[2].n[i]
					if a != 0 {
						v = args[1].n[i]
					}
				case "addi", "addf":
					v = a + args[1].n[i]
				case "subf":
					v = a - args[1].n[i]
				case "mulf":
					v = a * args[1].n[i]
				case "divf":
					v = a / args[1].n[i]
				default:
					t.Fatalf("unmodeled opcode %s", code)
				}
				out.n[i] = v
			}
		}
		if len(op.Results) > 0 {
			vs[op.Results[0]] = out
		}
	}
	t.Fatal("no typed return")
	return nil
}
