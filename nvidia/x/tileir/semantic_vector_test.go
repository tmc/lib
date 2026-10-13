package tileir

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func vectorGraph(t *testing.T, n int) (*cudair.Graph, *cudair.Snapshot) {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{n}, cudair.Float32)
	y := g.AddInput("y", []int{n}, cudair.Float32)
	x.SourceLoc = cudair.SourceLocation{File: "vector.go", Line: 10}
	out := g.AddBinary("add", x, y)
	out.SourceLoc = cudair.SourceLocation{File: "vector.go", Line: 20}
	if err := g.SetOutputs(out); err != nil {
		t.Fatal(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return g, s
}

func vectorLowering(t *testing.T, n int) *LoweringResult {
	t.Helper()
	_, s := vectorGraph(t, n)
	r, err := LowerSemanticVector(s, LowerOptions{KernelName: "vector", TileM: 4})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

type vectorPointer struct {
	slot  int
	index int64
}
type vectorValue struct {
	integers []int64
	floats   []float32
	pointers []vectorPointer
}

// Interpret decoded SSA, including every inactive-lane pointer addition.
// Scalar/address topology is modeled on CPU; no CUDA code is executed.
func evaluateVector(data []byte, grid [3]uint32, x, y []float32) ([]float32, error) {
	bm, err := DecodeBytecodeBytes(data)
	if err != nil {
		return nil, err
	}
	if len(bm.Functions) != 1 || bm.Functions[0].BodyDecodeError != nil {
		return nil, fmt.Errorf("unsupported decoded body")
	}
	output := make([]float32, len(x))
	writes := make([]int, len(x))
	for i := range output {
		output[i] = float32(777)
	}
	for block := uint32(0); block < grid[0]; block++ {
		values := map[string]vectorValue{"%arg0": {pointers: []vectorPointer{{slot: 0}}}, "%arg1": {pointers: []vectorPointer{{slot: 1}}}, "%arg2": {pointers: []vectorPointer{{slot: 2}}}}
		buffers := [][]float32{x, y, output}
		for _, op := range bm.Functions[0].Operations {
			get := func(i int) vectorValue { return values[op.Operands[i]] }
			var result vectorValue
			count := 1
			if typ, ok := op.TypeVal.(*TileType); ok {
				count = int(typ.NumElements())
			}
			switch op.OpCode {
			case "cuda_tile.get_tile_block_id":
				for i, name := range op.Results {
					n := int64(0)
					if i == 0 {
						n = int64(block)
					}
					values[name] = vectorValue{integers: []int64{n}}
				}
				continue
			case "cuda_tile.constant":
				text := strings.TrimSuffix(strings.TrimPrefix(op.Attributes, "{value = dense<\"0x"), "\">}")
				raw, err := hex.DecodeString(text)
				if err != nil {
					return nil, err
				}
				typ := op.TypeVal.(*TileType)
				if typ.ElementType.Kind() == KindInt64 {
					result.integers = make([]int64, count)
					for i := range result.integers {
						j := i * 8
						if len(raw) == 8 {
							j = 0
						}
						result.integers[i] = int64(binary.LittleEndian.Uint64(raw[j:]))
					}
				} else if typ.ElementType.Kind() == KindFloat32 {
					result.floats = make([]float32, count)
					for i := range result.floats {
						j := i * 4
						if len(raw) == 4 {
							j = 0
						}
						result.floats[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[j:]))
					}
				} else {
					return nil, fmt.Errorf("oracle constant type")
				}
			case "cuda_tile.exti":
				result.integers = []int64{int64(uint32(get(0).integers[0]))}
			case "cuda_tile.reshape":
				result = get(0)
			case "cuda_tile.broadcast":
				a := get(0)
				if len(a.integers) != 0 {
					result.integers = make([]int64, count)
					for i := range result.integers {
						result.integers[i] = a.integers[0]
					}
				} else if len(a.pointers) != 0 {
					result.pointers = make([]vectorPointer, count)
					for i := range result.pointers {
						result.pointers[i] = a.pointers[0]
					}
				} else {
					return nil, fmt.Errorf("oracle broadcast")
				}
			case "cuda_tile.iota":
				result.integers = make([]int64, count)
				for i := range result.integers {
					result.integers[i] = int64(i)
				}
			case "cuda_tile.muli", "cuda_tile.addi", "cuda_tile.cmpi":
				a, b := get(0).integers, get(1).integers
				result.integers = make([]int64, count)
				for i := range result.integers {
					switch op.OpCode {
					case "cuda_tile.muli":
						result.integers[i] = a[i] * b[i]
					case "cuda_tile.addi":
						result.integers[i] = a[i] + b[i]
					case "cuda_tile.cmpi":
						if uint64(a[i]) < uint64(b[i]) {
							result.integers[i] = 1
						}
					}
				}
			case "cuda_tile.select":
				mask, a, b := get(0).integers, get(1).integers, get(2).integers
				result.integers = make([]int64, count)
				for i := range result.integers {
					result.integers[i] = b[i]
					if mask[i] != 0 {
						result.integers[i] = a[i]
					}
				}
			case "cuda_tile.offset":
				a, b := get(0).pointers, get(1).integers
				result.pointers = make([]vectorPointer, count)
				for i := range result.pointers {
					pointer := a[i]
					pointer.index += b[i]
					if pointer.index < 0 || pointer.index >= int64(len(buffers[pointer.slot])) {
						return nil, fmt.Errorf("out-of-bounds pointer before masking: slot%d index%d", pointer.slot, pointer.index)
					}
					result.pointers[i] = pointer
				}
			case "cuda_tile.make_token":
				result.integers = []int64{0}
			case "cuda_tile.load_ptr_tko":
				p, mask, padding, token := get(0).pointers, get(1).integers, get(2).floats, get(3).integers
				result.floats = make([]float32, len(p))
				for i, pointer := range p {
					result.floats[i] = padding[i]
					if mask[i] != 0 {
						result.floats[i] = buffers[pointer.slot][pointer.index]
					}
				}
				values[op.Results[0]] = result
				values[op.Results[1]] = vectorValue{integers: []int64{token[0] + 1}}
				continue
			case "cuda_tile.addf":
				a, b := get(0).floats, get(1).floats
				result.floats = make([]float32, count)
				for i := range result.floats {
					result.floats[i] = a[i] + b[i]
				}
			case "cuda_tile.store_ptr_tko":
				p, a, mask := get(0).pointers, get(1).floats, get(2).integers
				for i, pointer := range p {
					if mask[i] != 0 {
						buffers[pointer.slot][pointer.index] = a[i]
						if pointer.slot == 2 {
							writes[pointer.index]++
						}
					}
				}
				result.integers = []int64{get(3).integers[0] + 1}
			case "cuda_tile.return":
				continue
			default:
				return nil, fmt.Errorf("unknown oracle opcode %s", op.OpCode)
			}
			values[op.Results[0]] = result
		}
	}
	for i, n := range writes {
		if n != 1 {
			return nil, fmt.Errorf("output%d written %d times", i, n)
		}
	}
	return output, nil
}

func TestSemanticVectorDecodedOracle(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 9, 10, 13} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			r := vectorLowering(t, n)
			data, err := r.Module.EncodeBytecode()
			if err != nil {
				t.Fatal(err)
			}
			x, y, want := make([]float32, n), make([]float32, n), make([]float32, n)
			for i := range x {
				x[i] = float32(i*7-13) + 0.25
				y[i] = float32(3-i*2) + 0.5
				want[i] = x[i] + y[i]
			}
			if n > 0 {
				x[0] = math.SmallestNonzeroFloat32
				y[0] = math.SmallestNonzeroFloat32
				want[0] = 2 * math.SmallestNonzeroFloat32
			}
			got, err := evaluateVector(data, r.Grid, x, y)
			if err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("output%d=%g want%g", i, got[i], want[i])
				}
			}
			if err := r.Memory.Verify(r); err != nil {
				t.Fatal(err)
			}
			decoded, err := DecodeBytecodeBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			round, err := decoded.ToModule().EncodeBytecode()
			if err != nil || !bytes.Equal(data, round) {
				t.Fatalf("round trip: %v", err)
			}
			if n == 0 && (len(r.Schema.Params) != 0 || len(r.Module.Functions[0].Params) != 0 || len(r.Module.Functions[0].Operations) != 1 || hasMemoryOps(r.Module.Functions[0].Operations) || r.Grid[0] != 0) {
				t.Fatal("empty branch forms addresses or memory")
			}
		})
	}
}

func TestSemanticVectorTopologyOmissions(t *testing.T) {
	for _, mutate := range []func(*LoweringResult){
		func(r *LoweringResult) { r.Module.Functions[0].Operations[14].Operands[1] = "%indices" },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[23].Operands[0] = "%ptr1" },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[25].Operands[1] = "%x" },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[26].Operands[0] = "%ptr0" },
	} {
		r := vectorLowering(t, 5)
		mutate(r)
		data, err := r.Module.EncodeBytecode()
		if err != nil {
			t.Fatal(err)
		}
		x, y := []float32{1, 2, 3, 4, 5}, []float32{10, 20, 30, 40, 50}
		got, err := evaluateVector(data, r.Grid, x, y)
		wrong := err != nil
		if !wrong {
			for i := range got {
				wrong = wrong || got[i] != float32((i+1)*11)
			}
		}
		if !wrong || r.Memory.Verify(r) == nil {
			t.Fatal("omission escaped actual output/binding controls")
		}
	}
	r := vectorLowering(t, 5)
	r.Module.Functions[0].Operations[24].Operands[3] = "%token0"
	if data, err := r.Module.EncodeBytecode(); data != nil || err == nil {
		t.Fatal("broken load predecessor encoded")
	}
}

func TestSemanticVectorGeometryAndPointers(t *testing.T) {
	for _, tt := range []struct {
		n, w int64
		want bool
	}{{0, 4, true}, {1, 4, true}, {5, 4, true}, {(1 << 24) - 1, 1, true}, {1 << 24, 1, false}, {((1 << 24) - 1) * 1024, 1024, uint64(^uint(0)>>1)/4 >= uint64(((1<<24)-1)*1024)}, {-1, 4, false}, {5, 3, false}, {5, 2048, false}, {math.MaxInt64, 4, false}} {
		v, grid, err := vectorGeometry(tt.n, tt.w)
		if (err == nil) != tt.want {
			t.Fatalf("geometry(%d,%d): %v", tt.n, tt.w, err)
		}
		if err == nil && (uint64(grid[0]) != uint64(tt.n/tt.w)+boolCount(tt.n%tt.w != 0) || v.BufferBytes != uint64(tt.n)*4) {
			t.Fatal("wrong geometry")
		}
	}
	r := vectorLowering(t, 5)
	for _, p := range [][3]uint64{{0, 0x2000, 0x3000}, {0x1001, 0x2000, 0x3000}, {0x1000, 0x1004, 0x3000}, {^uint64(0) - 3, 0x2000, 0x3000}} {
		if r.Vector.ValidatePointers(p) == nil {
			t.Fatal("invalid pointer/alias admitted")
		}
	}
	if err := r.Vector.ValidatePointers([3]uint64{0x1000, 0x1014, 0x1028}); err != nil {
		t.Fatal(err)
	}
	if err := vectorLowering(t, 0).Vector.ValidatePointers([3]uint64{}); err != nil {
		t.Fatal(err)
	}
	r.Vector.Length++
	if r.Memory.Verify(r) == nil {
		t.Fatal("extent mutation escaped binding")
	}
}

func boolCount(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

func TestSemanticVectorRefusalsAndArtifact(t *testing.T) {
	for _, mutate := range []func(*cudair.Graph){
		func(g *cudair.Graph) {
			for _, n := range g.Nodes() {
				n.Type = cudair.Float16
			}
		},
		func(g *cudair.Graph) { g.Nodes()[0].Strides = []int{2} },
		func(g *cudair.Graph) { g.Nodes()[0].Layout = cudashape.LayoutColumnMajor },
		func(g *cudair.Graph) { g.Nodes()[2].Effect = cudair.EffectMutating },
		func(g *cudair.Graph) { g.Nodes()[2].StorageAlias = g.Nodes()[0] },
		func(g *cudair.Graph) { g.Nodes()[2].Op = "mul" },
		func(g *cudair.Graph) { _ = g.SetOutputs(g.Nodes()[0], g.Nodes()[2]) },
	} {
		g, _ := vectorGraph(t, 5)
		mutate(g)
		s, err := g.Snapshot()
		if err != nil {
			continue
		}
		if r, err := LowerSemanticVector(s, LowerOptions{}); r != nil || err == nil {
			t.Fatal("unsupported graph admitted")
		}
	}
	r := vectorLowering(t, 5)
	want, _ := r.Module.EncodeBytecode()
	calls := 0
	artifact, err := assembleSemanticMemory(context.Background(), r, "sm_90", func(_ context.Context, data []byte, target string) (*tileiras.Result, error) {
		calls++
		if !bytes.Equal(data, want) || target != "sm_90" {
			t.Fatal("typed vector bytes lost")
		}
		return &tileiras.Result{CUBIN: syntheticMemoryELF(t, "vector", 90)}, nil
	})
	if err != nil || calls != 1 || artifact.Artifact.EntryPoint != "vector" || artifact.BytecodeDigest != r.Memory.BytecodeDigest() {
		t.Fatalf("vector artifact composition: %v", err)
	}
	empty := vectorLowering(t, 0)
	if _, err := assembleSemanticMemory(context.Background(), empty, "sm_90", func(_ context.Context, data []byte, _ string) (*tileiras.Result, error) {
		bm, err := DecodeBytecodeBytes(data)
		if err != nil || len(bm.Functions[0].Operations) != 1 {
			t.Fatal("empty typed artifact body changed")
		}
		return &tileiras.Result{CUBIN: syntheticMemoryELF(t, "vector", 90)}, nil
	}); err != nil {
		t.Fatal(err)
	}
	_, s := vectorGraph(t, 5)
	dispatched, err := LowerSemanticRegion(s, LowerOptions{TileM: 4})
	if err != nil || dispatched.Vector == nil {
		t.Fatalf("vector dispatch: %v", err)
	}
}

func ExampleLowerSemanticVector() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{5}, cudair.Float32)
	y := g.AddInput("y", []int{5}, cudair.Float32)
	_ = g.SetOutputs(g.AddBinary("add", x, y))
	s, _ := g.Snapshot()
	r, err := LowerSemanticVector(s, LowerOptions{TileM: 4})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Grid, r.Vector.Length, r.Vector.BufferBytes)
	// Output: [2 1 1] 5 20
}

func ExampleSemanticVectorLayout_ValidatePointers() {
	v := &SemanticVectorLayout{Length: 5, TileSize: 4, BufferBytes: 20}
	fmt.Println(v.ValidatePointers([3]uint64{0x1000, 0x1014, 0x1028}))
	fmt.Println(v.ValidatePointers([3]uint64{0x1000, 0x1004, 0x1028}) != nil)
	// Output:
	// <nil>
	// true
}
