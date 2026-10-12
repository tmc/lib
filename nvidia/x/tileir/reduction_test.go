package tileir_test

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// reductionValue holds a scalar or vector, optionally referring to an ABI
// buffer. The model interprets emitted SSA, rather than the semantic graph.
type reductionValue struct {
	lanes  []float64
	buffer string
}

func modelReduction(fn *tileir.Function, buffers map[string][]float32, blocks int) error {
	// These fixtures have one input, followed by outputs with two guards.
	lengths := make(map[string]int)
	for i, p := range fn.Params {
		lengths[p.Name] = len(buffers[p.Name])
		if i > 0 {
			lengths[p.Name] -= 2
		}
	}
	for block := 0; block < blocks; block++ {
		values := make(map[string]reductionValue)
		for _, p := range fn.Params {
			values[p.Name] = reductionValue{[]float64{0}, p.Name}
		}
		for _, op := range fn.Operations {
			code := strings.TrimPrefix(op.OpCode, "cuda_tile.")
			if code == "make_token" || code == "return" {
				continue
			}
			args := make([]reductionValue, len(op.Operands))
			for i, name := range op.Operands {
				args[i] = values[name]
			}
			typ := op.TypeVal
			if ft, ok := typ.(*tileir.FunctionType); ok {
				typ = ft.Results[0]
			}
			n := 1
			elem := typ
			if tt, ok := typ.(*tileir.TileType); ok {
				n = int(tt.Shape[0])
				elem = tt.ElementType
			}
			get := func(i, lane int) float64 {
				a := args[i].lanes
				if len(a) == 1 {
					return a[0]
				}
				return a[lane]
			}
			v := reductionValue{lanes: make([]float64, n)}
			for lane := 0; lane < n; lane++ {
				switch code {
				case "constant":
					literal := strings.TrimPrefix(op.Attributes, "value = ")
					literal = strings.Split(literal, " semantic_id")[0]
					switch literal {
					case "-std::numeric_limits<float>::infinity()":
						v.lanes[lane] = math.Inf(-1)
					case "std::numeric_limits<float>::quiet_NaN()":
						v.lanes[lane] = math.NaN()
					default:
						number, err := strconv.ParseFloat(strings.TrimRight(literal, "fu"), 64)
						if err != nil {
							return err
						}
						v.lanes[lane] = number
					}
				case "get_tile_block_id":
					v.lanes[lane] = float64(block)
				case "iota":
					v.lanes[lane] = float64(lane)
				case "addi":
					v.lanes[lane] = get(0, lane) + get(1, lane)
				case "muli":
					v.lanes[lane] = get(0, lane) * get(1, lane)
				case "offset":
					v.buffer = args[0].buffer
					v.lanes[lane] = get(0, lane) + get(1, lane)
					if lengths[v.buffer] == 0 || v.lanes[lane] < 0 || v.lanes[lane] > float64(lengths[v.buffer]) {
						return fmt.Errorf("pointer formation outside logical ABI object %s: %g extent %d", v.buffer, v.lanes[lane], lengths[v.buffer])
					}
				case "cmpi":
					if get(0, lane) < get(1, lane) {
						v.lanes[lane] = 1
					}
				case "cmpf":
					yes := get(0, lane) > get(1, lane)
					if strings.Contains(op.Attributes, "comparison_ordering = unordered") {
						yes = math.IsNaN(get(0, lane)) || math.IsNaN(get(1, lane)) || get(0, lane) != get(1, lane)
					}
					if yes {
						v.lanes[lane] = 1
					}
				case "select":
					if get(0, lane) != 0 {
						v.lanes[lane] = get(1, lane)
					} else {
						v.lanes[lane] = get(2, lane)
					}
				case "ftof":
					v.lanes[lane] = get(0, lane)
					if elem.Kind() == tileir.KindFloat32 {
						v.lanes[lane] = float64(float32(v.lanes[lane]))
					}
				case "addf":
					v.lanes[lane] = get(0, lane) + get(1, lane)
				case "load_ptr_tko":
					v.lanes[lane] = get(2, lane)
					if get(1, lane) != 0 {
						index := int(get(0, lane))
						buffer := buffers[args[0].buffer]
						if index < 0 || index >= len(buffer) {
							return fmt.Errorf("load outside ABI buffer: %d", index)
						}
						v.lanes[lane] = float64(buffer[index])
					}
				case "store_ptr_tko":
					// Store result is a token, so its width comes from the pointer operand.
				default:
					return fmt.Errorf("unknown modeled operation %s", code)
				}
			}
			if code == "store_ptr_tko" {
				for lane := range args[0].lanes {
					if get(2, lane) == 0 {
						continue
					}
					index := int(get(0, lane))
					buffer := buffers[args[0].buffer]
					if index < 0 || index >= len(buffer)-2 {
						return fmt.Errorf("store outside logical output: %d", index)
					}
					buffer[index] = float32(get(1, lane))
				}
			} else {
				values[op.Results[0]] = v
			}
		}
	}
	return nil
}

func reductionGraph(t *testing.T, op string, shape []int, axis int) *cudair.Snapshot {
	t.Helper()
	g := cudair.NewGraph()
	a := g.AddInput("a", shape, cudair.Float32)
	out := g.AddReduce(op, a, axis)
	out.SourceLoc = cudair.SourceLocation{File: "reduction.go", Line: 37, Function: "Reduce"}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return snap
}

func equalReduction(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || math.IsNaN(float64(a)) && math.IsNaN(float64(b))
}

func TestOrderedReductionModel(t *testing.T) {
	negzero := float32(math.Copysign(0, -1))
	cases := []struct {
		name, op string
		shape    []int
		axis     int
		input    []float32
	}{
		{"rows and tails", "sum", []int{5, 3}, 1, []float32{16777216, 1, -16777216, 2, 3, 4, -2, -3, -4, 1e30, 1, -1e30, 1, 2, 3}},
		{"column addresses", "sum", []int{3, 5}, 0, []float32{1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500}},
		{"rank one", "sum", []int{3}, 0, []float32{16777216, 1, -16777216}},
		{"subnormal", "sum", []int{2}, 0, []float32{math.SmallestNonzeroFloat32, math.SmallestNonzeroFloat32}},
		{"infinities", "sum", []int{2, 2}, 1, []float32{float32(math.Inf(1)), 1, float32(math.Inf(1)), float32(math.Inf(-1))}},
		{"max zero order", "max", []int{2, 2}, 1, []float32{negzero, 0, 0, negzero}},
		{"max columns", "max", []int{2, 5}, 0, []float32{1, 9, -3, 0, negzero, 4, 5, -1, negzero, 0}},
		{"max nan first", "max", []int{3}, 0, []float32{float32(math.NaN()), 10, 20}},
		{"max nan later", "max", []int{3}, 0, []float32{10, float32(math.NaN()), 20}},
		{"empty sum", "sum", []int{3, 0}, 1, nil},
		{"empty max", "max", []int{0, 3}, 0, nil},
		{"no outputs", "sum", []int{0, 3}, 1, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			snap := reductionGraph(t, tc.op, tc.shape, tc.axis)
			result, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{TileM: 4})
			if err != nil {
				t.Fatal(err)
			}
			expected, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"a": tc.input})
			if err != nil {
				t.Fatal(err)
			}
			want := expected[snap.Outputs()[0].Name]
			out := make([]float32, len(want)+2)
			for i := range out {
				out[i] = 999
			}
			buffers := map[string][]float32{"%arg0": tc.input, "%arg1": out}
			// One extra block challenges the mask even when the extent is a tile multiple.
			if err := modelReduction(result.Module.Functions[0], buffers, int(result.Grid[0])+1); err != nil {
				t.Fatal(err)
			}
			for i, w := range want {
				if !equalReduction(out[i], w) {
					t.Fatalf("output %d = %v (%08x), want %v (%08x)", i, out[i], math.Float32bits(out[i]), w, math.Float32bits(w))
				}
			}
			if out[len(want)] != 999 || out[len(want)+1] != 999 {
				t.Fatal("output guard changed")
			}
			hazards, shapes := result.Module.Verify()
			if len(hazards)+len(shapes) != 0 {
				t.Fatalf("verification: %v %v", hazards, shapes)
			}
			if len(result.Module.Locations) != 1 {
				t.Fatal("missing semantic source location")
			}
			calls := []string{"ct::iota<ct::tile<unsigned int, ct::shape<4>>>"}
			if len(want) > 0 {
				calls = append(calls, "ct::store_masked(", "ct::select(")
			}
			for _, call := range calls {
				if !strings.Contains(result.Source, call) {
					t.Fatalf("missing documented vector call %q", call)
				}
			}
			if tc.op == "sum" && len(tc.input) > 0 {
				for _, call := range []string{"ct::element_cast<double>", "ct::element_cast<float>", "ct::round_ties_to_even_t{}", "ct::preserve_subnormals_t{}"} {
					if !strings.Contains(result.Source, call) {
						t.Fatalf("missing strict arithmetic %q", call)
					}
				}
			}
		})
	}
}

func TestOrderedReductionRefusals(t *testing.T) {
	for _, tc := range []struct {
		name, op string
		shape    []int
		axis     int
		width    int64
	}{
		{"mean", "mean", []int{3}, 0, 4}, {"term budget", "sum", []int{65}, 0, 4}, {"rank", "sum", []int{2, 2, 2}, 0, 4}, {"width", "sum", []int{3}, 0, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tileir.LowerSemanticRegion(reductionGraph(t, tc.op, tc.shape, tc.axis), tileir.LowerOptions{TileM: tc.width}); err == nil {
				t.Fatal("unsupported region accepted")
			}
		})
	}
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{3}, cudair.Float32)
	b := g.AddUnary("neg", a)
	g.AddReduce("sum", b, 0)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = tileir.LowerSemanticRegion(snap, tileir.LowerOptions{}); err == nil {
		t.Fatal("composed reduction accepted")
	}
	// The earlier scalar source route still accepts its supported mean graph.
	if _, _, err := tileir.GenerateSemanticTileCPP(reductionGraph(t, "mean", []int{3}, 0), tileir.LowerOptions{}); err != nil {
		t.Fatal(err)
	}
}

func TestOrderedReductionCausalControls(t *testing.T) {
	snap := reductionGraph(t, "sum", []int{3}, 0)
	for _, tc := range []struct {
		name   string
		mutate func(*tileir.Operation) bool
	}{
		{"undefined accumulator", func(op *tileir.Operation) bool {
			if op.OpCode != "cuda_tile.addf" {
				return false
			}
			op.Operands[0] = "%missing"
			return true
		}},
		{"narrow accumulator", func(op *tileir.Operation) bool {
			if op.OpCode != "cuda_tile.addf" {
				return false
			}
			op.TypeVal = tileir.NewTile([]int64{4}, tileir.Float32)
			return true
		}},
		{"token predecessor", func(op *tileir.Operation) bool {
			if op.OpCode != "cuda_tile.store_ptr_tko" {
				return false
			}
			op.Operands[3] = "%tok0"
			return true
		}},
		{"mask omitted", func(op *tileir.Operation) bool {
			if op.OpCode != "cuda_tile.load_ptr_tko" {
				return false
			}
			op.Operands = op.Operands[:3]
			return true
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{TileM: 4})
			if err != nil {
				t.Fatal(err)
			}
			for _, op := range r.Module.Functions[0].Operations {
				if tc.mutate(op) {
					break
				}
			}
			_, shapes := r.Module.Verify()
			if len(shapes) == 0 {
				t.Fatal("omission control escaped SSA verification")
			}
		})
	}
	r, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{TileM: 4})
	if err != nil {
		t.Fatal(err)
	}
	// A plausible wrong address remains well typed; the numeric model must reject it.
	for _, op := range r.Module.Functions[0].Operations {
		if strings.HasPrefix(op.Attributes, "value = 1u semantic_id") {
			op.Attributes = strings.Replace(op.Attributes, "value = 1u", "value = 0u", 1)
			break
		}
	}
	out := []float32{999, 999, 999}
	if err := modelReduction(r.Module.Functions[0], map[string][]float32{"%arg0": {1, 2, 4}, "%arg1": out}, 1); err != nil {
		t.Fatal(err)
	}
	if out[0] == 7 {
		t.Fatal("wrong reduction address escaped numeric control")
	}
}

func ExampleLowerSemanticRegion() {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{5, 3}, cudair.Float32)
	g.AddReduce("sum", a, 1)
	snap, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	r, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{KernelName: "row_sum", TileM: 4})
	if err != nil {
		panic(err)
	}
	fmt.Println(r.Module.Functions[0].Name, len(r.Schema.Params), r.Grid)
	// Output: row_sum 2 [2 1 1]
}

func TestOrderedReductionMultipleOutputs(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 5}, cudair.Float32)
	g.AddReduce("sum", a, 0)
	g.AddReduce("max", a, 1)
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	input := []float32{1, 2, 3, 4, 5, 10, 20, 30, 40, 50}
	result, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{TileM: 4})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Schema.Params) != 3 || result.Grid != [3]uint32{2, 1, 1} {
		t.Fatalf("ABI/grid: %v %v", result.Schema, result.Grid)
	}
	expected, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"a": input})
	if err != nil {
		t.Fatal(err)
	}
	buffers := map[string][]float32{"%arg0": input}
	for i, n := range snap.Outputs() {
		out := make([]float32, n.NumElements()+2)
		for j := range out {
			out[j] = 999
		}
		buffers[fmt.Sprintf("%%arg%d", i+1)] = out
	}
	if err := modelReduction(result.Module.Functions[0], buffers, 3); err != nil {
		t.Fatal(err)
	}
	for i, n := range snap.Outputs() {
		out := buffers[fmt.Sprintf("%%arg%d", i+1)]
		for j, w := range expected[n.Name] {
			if !equalReduction(out[j], w) {
				t.Fatalf("output %s[%d] = %v, want %v", n.Name, j, out[j], w)
			}
		}
		if out[len(out)-1] != 999 || out[len(out)-2] != 999 {
			t.Fatal("guard changed")
		}
	}
}

func TestOrderedReductionEmptyPointerSafety(t *testing.T) {
	for _, shape := range [][]int{{0}, {0, 3}, {3, 0}, {0, 0}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			g := cudair.NewGraph()
			a := g.AddInput("a", shape, cudair.Float32)
			for axis := range shape {
				for _, op := range []string{"sum", "max"} {
					g.AddReduce(op, a, axis)
				}
			}
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			result, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{TileM: 4})
			if err != nil {
				t.Fatal(err)
			}
			expected, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"a": nil})
			if err != nil {
				t.Fatal(err)
			}
			buffers := map[string][]float32{"%arg0": nil}
			zeroOutputIDs := make(map[int]bool)
			for i, n := range snap.Outputs() {
				out := make([]float32, n.NumElements()+2)
				for j := range out {
					out[j] = 999
				}
				buffers[fmt.Sprintf("%%arg%d", i+1)] = out
				if n.NumElements() == 0 {
					zeroOutputIDs[n.ID] = true
				}
			}
			for _, op := range result.Module.Functions[0].Operations {
				if op.OpCode == "cuda_tile.load_ptr_tko" || op.OpCode == "cuda_tile.offset" && op.Operands[0] == "%arg0" {
					t.Fatal("empty input has pointer formation or load")
				}
				for id := range zeroOutputIDs {
					if strings.HasSuffix(op.Attributes, fmt.Sprintf("semantic_id = %d", id)) && (op.OpCode == "cuda_tile.offset" || op.OpCode == "cuda_tile.store_ptr_tko") {
						t.Fatal("zero output has pointer formation or store")
					}
				}
			}
			if len(result.Module.Locations) != 0 {
				t.Fatal("graph without source locations acquired locations")
			}
			if err := modelReduction(result.Module.Functions[0], buffers, int(result.Grid[0])+1); err != nil {
				t.Fatal(err)
			}
			for i, n := range snap.Outputs() {
				out := buffers[fmt.Sprintf("%%arg%d", i+1)]
				for j, w := range expected[n.Name] {
					if !equalReduction(out[j], w) {
						t.Fatalf("output %s[%d] = %v, want %v", n.Name, j, out[j], w)
					}
				}
				if out[len(out)-1] != 999 || out[len(out)-2] != 999 {
					t.Fatal("guard changed")
				}
			}
		})
	}
}

func TestOrderedReductionCompleteOffsetControl(t *testing.T) {
	// Strip integer offset selections. The mask still protects loads, but
	// the independent formation check must observe an invalid pointer first.
	snap := reductionGraph(t, "sum", []int{1, 3}, 1)
	result, err := tileir.LowerSemanticRegion(snap, tileir.LowerOptions{TileM: 4})
	if err != nil {
		t.Fatal(err)
	}
	fn := result.Module.Functions[0]
	definitions := make(map[string]*tileir.Operation)
	for _, op := range fn.Operations {
		for _, name := range op.Results {
			definitions[name] = op
		}
	}
	for _, op := range fn.Operations {
		if op.OpCode == "cuda_tile.offset" && op.Operands[0] == "%arg0" {
			choice := definitions[op.Operands[1]]
			if choice.OpCode != "cuda_tile.select" {
				t.Fatal("complete offset is not sanitized")
			}
			op.Operands[1] = choice.Operands[1]
		}
	}
	// Remove the earlier row selection too, so complete address sanitation is
	// the only possible protection in this deliberate omission model.
	for _, choice := range fn.Operations {
		if choice.OpCode != "cuda_tile.select" {
			continue
		}
		tile, ok := choice.TypeVal.(*tileir.TileType)
		if !ok || tile.ElementType.Kind() != tileir.KindInt32 {
			continue
		}
		for _, op := range fn.Operations {
			for i, operand := range op.Operands {
				if operand == choice.Results[0] {
					op.Operands[i] = choice.Operands[1]
				}
			}
		}
	}
	// The second block has no active outputs; its full offset exceeds extent.
	err = modelReduction(fn, map[string][]float32{"%arg0": {1, 2, 3}, "%arg1": {999, 999, 999}}, 2)
	if err == nil {
		t.Fatal("unsanitized full pointer offsets escaped formation control")
	}
}
