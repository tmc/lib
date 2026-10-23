package egraph_test

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/tileir"
)

// projectionOracle is a scalar float64 specification over row-major matrices.
// It does not inspect lowering code, generated source, or the SSA evaluator.
func projectionOracle(x, norm []float32, weights [][]float32, m, n, k int, eps float32) []float64 {
	out := make([]float64, m*n)
	for r := 0; r < m; r++ {
		var squares float64
		for a := 0; a < k; a++ {
			v := float64(x[r*k+a])
			squares += v * v
		}
		scale := 1 / math.Sqrt(squares/float64(k)+float64(eps))
		for c := 0; c < n; c++ {
			sums := make([]float64, len(weights))
			for p, w := range weights {
				for a := 0; a < k; a++ {
					sums[p] += float64(x[r*k+a]) * scale * float64(norm[a]) * float64(w[a*n+c])
				}
			}
			out[r*n+c] = sums[0]
			if len(sums) == 3 {
				out[r*n+c] = sums[0] + (sums[1] + sums[2])
			}
		}
	}
	return out
}

type projectionPointer struct{ buffer, index int }
type projectionValue struct {
	n []float64
	p []projectionPointer
}

// executeProjection interprets decoded typed SSA operations, not RMSNorm or
// projection formulas. It checks pointer formation before masks and tracks all
// stores. This is a host model of the encoded artifact, not native execution.
func executeProjection(mod *tileir.Module, grid [3]uint32, buffers [][]float32, eps float32) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("invalid SSA: %v", r)
		}
	}()
	fn := mod.Functions[0]
	stores := make([]int, len(buffers[len(buffers)-1]))
	for by := uint32(0); by < grid[1]; by++ {
		for bx := uint32(0); bx < grid[0]; bx++ {
			values := make(map[string]projectionValue)
			for j, p := range fn.Params {
				if j < len(buffers) {
					values[p.Name] = projectionValue{p: []projectionPointer{{j, 0}}}
				} else {
					values[p.Name] = projectionValue{n: []float64{float64(eps)}}
				}
			}
			for _, op := range fn.Operations {
				args := make([]projectionValue, len(op.Operands))
				for j, a := range op.Operands {
					v, ok := values[a]
					if !ok {
						return fmt.Errorf("undefined %s", a)
					}
					args[j] = v
				}
				if op.OpCode == "cuda_tile.return" {
					continue
				}
				if op.OpCode == "cuda_tile.get_tile_block_id" {
					for j, v := range []float64{float64(bx), float64(by), 0} {
						values[op.Results[j]] = projectionValue{n: []float64{v}}
					}
					continue
				}
				if op.OpCode == "cuda_tile.make_token" {
					values[op.Results[0]] = projectionValue{}
					continue
				}
				typ := op.TypeVal
				if t, ok := typ.(*tileir.FunctionType); ok {
					typ = t.Results[0]
				}
				if op.OpCode == "cuda_tile.store_ptr_tko" {
					for lane, p := range args[0].p {
						if args[2].n[lane] != 0 {
							if p.buffer != len(buffers)-1 {
								return fmt.Errorf("store to input")
							}
							buffers[p.buffer][p.index] = float32(args[1].n[lane])
							stores[p.index]++
						}
					}
					values[op.Results[0]] = projectionValue{}
					continue
				}
				tile := typ.(*tileir.TileType)
				width := 1
				for _, d := range tile.Shape {
					width *= int(d)
				}
				result := projectionValue{n: make([]float64, width)}
				switch code := strings.TrimPrefix(op.OpCode, "cuda_tile."); code {
				case "constant":
					field := strings.TrimSuffix(strings.TrimPrefix(op.Attributes, "{value = dense<\"0x"), "\">}")
					raw, e := hex.DecodeString(field)
					if e != nil {
						return e
					}
					size := len(raw) / width
					for lane := range result.n {
						p := raw[lane*size:]
						switch tile.ElementType.Kind() {
						case tileir.KindFloat32:
							result.n[lane] = float64(math.Float32frombits(binary.LittleEndian.Uint32(p)))
						case tileir.KindInt64:
							result.n[lane] = float64(int64(binary.LittleEndian.Uint64(p)))
						default:
							return fmt.Errorf("constant kind %s", tile.ElementType)
						}
					}
				case "reshape", "exti":
					result = args[0]
				case "broadcast":
					if args[0].p != nil {
						result.n = nil
						result.p = make([]projectionPointer, width)
						for j := range result.p {
							result.p[j] = args[0].p[0]
						}
					} else {
						for j := range result.n {
							result.n[j] = args[0].n[0]
						}
					}
				case "iota":
					for j := range result.n {
						result.n[j] = float64(j)
					}
				case "offset":
					result.n = nil
					result.p = make([]projectionPointer, width)
					for j, p := range args[0].p {
						p.index += int(args[1].n[j])
						if p.index < 0 || p.index >= len(buffers[p.buffer]) {
							return fmt.Errorf("pointer formation outside buffer %d: %d", p.buffer, p.index)
						}
						result.p[j] = p
					}
				case "load_ptr_tko":
					for j, p := range args[0].p {
						result.n[j] = args[2].n[j]
						if args[1].n[j] != 0 {
							result.n[j] = float64(buffers[p.buffer][p.index])
						}
					}
					values[op.Results[1]] = projectionValue{}
				default:
					for j := range result.n {
						a := args[0].n[j]
						var c float64
						if len(args) > 1 {
							c = args[1].n[j]
						}
						switch code {
						case "select":
							if a != 0 {
								result.n[j] = c
							} else {
								result.n[j] = args[2].n[j]
							}
						case "andi":
							if a != 0 && c != 0 {
								result.n[j] = 1
							}
						case "cmpi":
							if a < c {
								result.n[j] = 1
							}
						case "addi":
							result.n[j] = a + c
						case "muli":
							result.n[j] = a * c
						case "addf":
							result.n[j] = float64(float32(a) + float32(c))
						case "mulf":
							result.n[j] = float64(float32(a) * float32(c))
						case "divf":
							result.n[j] = float64(float32(a) / float32(c))
						case "rsqrt":
							result.n[j] = float64(float32(1 / math.Sqrt(a)))
						default:
							return fmt.Errorf("unsupported SSA operation %s", code)
						}
					}
				}
				values[op.Results[0]] = result
			}
		}
	}
	for j, n := range stores {
		if n != 1 {
			return fmt.Errorf("output %d stored %d times", j, n)
		}
	}
	return nil
}
