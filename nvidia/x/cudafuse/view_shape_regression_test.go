package cudafuse_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// The expected shapes follow parrot.hpp at NVlabs/parrot 6990700:
// gather yields one element per index (a one-dimensional array of the
// indices' size), cycle takes the requested shape, and replicate(n) yields a
// one-dimensional array of size()*n.

func TestLazyViewShape(t *testing.T) {
	tests := []struct {
		name  string
		build func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray
		want  []int
	}{
		{"replicate_2", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Replicate(2)
		}, []int{6}},
		{"replicate_1", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Replicate(1)
		}, []int{3}},
		{"replicate_rank2_flattens", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{2, 3}, cudair.Float32).Replicate(2)
		}, []int{12}},
		{"replicate_empty", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32).Replicate(4)
		}, []int{0}},
		{"cycle_larger", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{6})
		}, []int{6}},
		{"cycle_smaller_truncates", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{6}, cudair.Float32).Cycle([]int{4})
		}, []int{4}},
		{"cycle_from_rank2", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{2, 3}, cudair.Float32).Cycle([]int{9})
		}, []int{9}},
		{"cycle_rank2", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{2, 3})
		}, []int{2, 3}},
		{"cycle_rank2_smaller", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{6}, cudair.Float32).Cycle([]int{2, 2})
		}, []int{2, 2}},
		{"gather_int32", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{5}, cudair.Int32))
		}, []int{5}},
		{"gather_int64_rank2_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{6}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2, 2}, cudair.Int64))
		}, []int{4}},
		{"gather_uint8_from_rank2", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.Uint8))
		}, []int{2}},
		{"gather_int8_empty_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{0}, cudair.Int8))
		}, []int{0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.build(cudafuse.NewLazyContext())
			if got.Node.Err != nil {
				t.Fatalf("Err = %v, want nil", got.Node.Err)
			}
			if !slices.Equal(got.Shape, tt.want) || !slices.Equal(got.Node.Shape, tt.want) {
				t.Fatalf("shape = %v, node shape = %v, want %v", got.Shape, got.Node.Shape, tt.want)
			}
			if got.DType != cudair.Float32 || got.Node.DType != cudair.Float32 {
				t.Fatalf("dtype = %v/%v, want float32 from the source", got.DType, got.Node.DType)
			}
			if got.Node.Evaluated || got.Ctx.Tracker.Snapshot().Allocations != 0 {
				t.Fatal("view construction evaluated or allocated")
			}
		})
	}
}

func TestLazyViewShapeInvalid(t *testing.T) {
	tests := []struct {
		name  string
		build func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray
	}{
		{"replicate_0", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Replicate(0)
		}},
		{"replicate_negative", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Replicate(-1)
		}},
		{"replicate_overflow", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{1 << 40}, cudair.Float32).Replicate(1 << 40)
		}},
		{"cycle_zero_size", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{0})
		}},
		{"cycle_negative_size", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{-2})
		}},
		{"cycle_no_shape", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{})
		}},
		{"cycle_zero_dim", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{2, 0})
		}},
		{"cycle_negative_dims", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{-2, -3})
		}},
		{"cycle_empty_input", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32).Cycle([]int{3})
		}},
		{"gather_float32_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{5}, cudair.Float32))
		}},
		{"gather_float16_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.Float16))
		}},
		{"gather_bfloat16_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.BFloat16))
		}},
		{"gather_unknown_index_dtype", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.DataType(99)))
		}},
		{"gather_nil_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Gather(nil)
		}},
		{"gather_invalid_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			bad := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Int32).Reshape([]int{2})
			return x.Gather(bad)
		}},
		{"gather_invalid_receiver", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Reshape([]int{2})
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{5}, cudair.Int32))
		}},
		{"cycle_invalid_input", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Reshape([]int{2}).Cycle([]int{4})
		}},
		{"replicate_invalid_input", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Reshape([]int{2}).Replicate(2)
		}},
		{"replicate_zero_then_negative_input", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{0, -1}, cudair.Float32).Replicate(2)
		}},
		{"replicate_negative_then_zero_input", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{-1, 0}, cudair.Float32).Replicate(2)
		}},
		{"cycle_zero_then_negative_input", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{0, -1}, cudair.Float32).Cycle([]int{3})
		}},
		{"cycle_zero_then_negative_target", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{0, -1})
		}},
		{"cycle_negative_then_zero_target", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32).Cycle([]int{-1, 0})
		}},
		{"gather_zero_then_negative_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{0, -1}, cudair.Int32))
		}},
		{"gather_negative_then_zero_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{-1, 0}, cudair.Int32))
		}},
		{"gather_negative_receiver", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{-1}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.Int32))
		}},
		{"gather_zero_then_negative_receiver_empty_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{0, -1}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{0}, cudair.Int32))
		}},
		{"gather_mutated_negative_receiver", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			// The receiver was valid at construction; Gather must still
			// check the shape it reads.
			x := cudafuse.NewLazyArray(ctx, []int{3}, cudair.Float32)
			x.Shape = []int{-1}
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.Int32))
		}},
		{"gather_empty_source_nonempty_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{0}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.Int32))
		}},
		{"gather_empty_rank2_source_nonempty_indices", func(ctx *cudafuse.LazyContext) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{2, 0}, cudair.Float32)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{1}, cudair.Int64))
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.build(cudafuse.NewLazyContext())
			if got.Node.Err == nil {
				t.Fatalf("Err = nil with shape %v, want an error", got.Shape)
			}
			// Cycle retains its requested target on an invalid node. Materialize
			// must still refuse it below.
			if got.Node.Op != "cycle" && (got.Shape != nil || got.Node.Shape != nil) {
				t.Fatalf("invalid view reports shape %v/%v, want unknown (nil)", got.Shape, got.Node.Shape)
			}
			if _, err := got.Materialize(); err == nil {
				t.Fatal("Materialize accepted an invalid view")
			}
		})
	}
}

// TestLazyViewShapeEdge covers valid empty and rank-0 scalar views and views
// of integer data. The result keeps the source data type.
func TestLazyViewShapeEdge(t *testing.T) {
	tests := []struct {
		name  string
		dtype cudair.DataType
		build func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray
		want  []int
	}{
		{"gather_empty_source_empty_indices", cudair.Float32, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{0}, dtype)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{0}, cudair.Int32))
		}, []int{0}},
		{"gather_empty_rank2_source_empty_rank2_indices", cudair.Int32, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{2, 0}, dtype)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{3, 0}, cudair.Int64))
		}, []int{0}},
		{"gather_scalar_source", cudair.Float32, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{}, dtype)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{2}, cudair.Int32))
		}, []int{2}},
		{"gather_scalar_indices", cudair.Float32, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{3}, dtype)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{}, cudair.Int32))
		}, []int{1}},
		{"gather_int32_source", cudair.Int32, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			x := cudafuse.NewLazyArray(ctx, []int{4}, dtype)
			return x.Gather(cudafuse.NewLazyArray(ctx, []int{3}, cudair.Int64))
		}, []int{3}},
		{"replicate_scalar", cudair.Float32, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{}, dtype).Replicate(3)
		}, []int{3}},
		{"replicate_int32_empty_rank2", cudair.Int32, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{2, 0}, dtype).Replicate(2)
		}, []int{0}},
		{"cycle_scalar", cudair.Int64, func(ctx *cudafuse.LazyContext, dtype cudair.DataType) *cudafuse.LazyArray {
			return cudafuse.NewLazyArray(ctx, []int{}, dtype).Cycle([]int{2, 2})
		}, []int{2, 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.build(cudafuse.NewLazyContext(), tt.dtype)
			if got.Node.Err != nil {
				t.Fatalf("Err = %v, want nil", got.Node.Err)
			}
			if !slices.Equal(got.Shape, tt.want) || !slices.Equal(got.Node.Shape, tt.want) {
				t.Fatalf("shape = %v, node shape = %v, want %v", got.Shape, got.Node.Shape, tt.want)
			}
			if got.DType != tt.dtype || got.Node.DType != tt.dtype {
				t.Fatalf("dtype = %v/%v, want %v from the source", got.DType, got.Node.DType, tt.dtype)
			}
			_, err := got.Materialize()
			if got.Node.Op == "cycle" && tt.dtype != cudair.Float32 {
				// Indexed cycle lowering currently admits FP32 only.
				if !errors.Is(err, cudair.ErrTypeMismatch) {
					t.Fatalf("integer cycle lowering = %v, want type mismatch", err)
				}
			} else if err != nil {
				t.Fatalf("Materialize: %v", err)
			}
		})
	}
}

// TestNewLazyArrayShape checks that the constructor records an invalid shape
// in Node.Err and that Materialize refuses it before recording an allocation.
func TestNewLazyArrayShape(t *testing.T) {
	tests := []struct {
		name      string
		shape     []int
		dtype     cudair.DataType
		wantErr   bool
		wantBytes int64
	}{
		{"empty", []int{0}, cudair.Float32, false, 0},
		{"empty_rank3_int32", []int{2, 0, 3}, cudair.Int32, false, 0},
		{"rank0_scalar", []int{}, cudair.Float32, false, 4},
		{"nil_scalar", nil, cudair.Int64, false, 8},
		{"rank2_int8", []int{2, 3}, cudair.Int8, false, 6},
		{"zero_then_negative", []int{0, -1}, cudair.Float32, true, 0},
		{"negative_then_zero", []int{-1, 0}, cudair.Float32, true, 0},
		{"zero_then_negative_int32", []int{0, -1}, cudair.Int32, true, 0},
		{"negative", []int{-1}, cudair.Float32, true, 0},
		{"overflow", []int{1 << 32, 1 << 32}, cudair.Float32, true, 0},
		{"exceeds_int", []int{1 << 62, 2}, cudair.Int8, true, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := cudafuse.NewLazyContext()
			x := cudafuse.NewLazyArray(ctx, tt.shape, tt.dtype)
			if (x.Node.Err != nil) != tt.wantErr {
				t.Fatalf("Node.Err = %v, want error %v", x.Node.Err, tt.wantErr)
			}
			plan, err := x.Materialize()
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Materialize accepted shape %v", tt.shape)
				}
				if n := ctx.Tracker.Snapshot().Allocations; n != 0 {
					t.Fatalf("refused shape recorded %d allocations", n)
				}
				return
			}
			if err != nil {
				t.Fatalf("Materialize: %v", err)
			}
			if plan.AllocatedBytes != tt.wantBytes {
				t.Fatalf("AllocatedBytes = %d, want %d", plan.AllocatedBytes, tt.wantBytes)
			}
		})
	}
}

// TestLazyShapeErrorPropagates checks that every lazy operation over an array
// whose constructor recorded a shape error records an error of its own, and
// that the public materializing entry points refuse it.
func TestLazyShapeErrorPropagates(t *testing.T) {
	type op struct {
		name  string
		build func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray
	}
	ops := []op{
		{"map", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Map("relu") }},
		{"add_receiver", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Add(good) }},
		{"add_operand", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return good.Add(bad) }},
		{"reshape", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Reshape([]int{0}) }},
		{"transpose", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Transpose([]int{1, 0}) }},
		{"where", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Where(good, good) }},
		{"filter", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return good.Filter(bad) }},
		{"keep", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return good.Keep(bad) }},
		{"replicate", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Replicate(2) }},
		{"cycle", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Cycle([]int{3}) }},
		{"gather", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray {
			return bad.Gather(cudafuse.NewLazyArray(bad.Ctx, []int{0}, cudair.Int32))
		}},
		{"map_chain", func(bad, good *cudafuse.LazyArray) *cudafuse.LazyArray { return bad.Map("relu").Map("exp") }},
	}
	for _, tt := range ops {
		t.Run(tt.name, func(t *testing.T) {
			ctx := cudafuse.NewLazyContext()
			bad := cudafuse.NewLazyArray(ctx, []int{0, -1}, cudair.Float32)
			good := cudafuse.NewLazyArray(ctx, []int{0, 1}, cudair.Float32)
			got := tt.build(bad, good)
			if got.Node.Err == nil || !strings.Contains(got.Node.Err.Error(), "negative") {
				t.Fatalf("Node.Err = %v, want the input's negative-dimension error", got.Node.Err)
			}
			if _, err := got.Materialize(); err == nil {
				t.Fatal("Materialize accepted an array built from an invalid shape")
			}
			if n := ctx.Tracker.Snapshot().Allocations; n != 0 {
				t.Fatalf("refused graph recorded %d allocations", n)
			}
		})
	}
	for _, tt := range []struct {
		name string
		run  func(bad *cudafuse.LazyArray) (*cudafuse.MaterializedPlan, error)
	}{
		{"Materialize", (*cudafuse.LazyArray).Materialize},
		{"ToHost", (*cudafuse.LazyArray).ToHost},
		{"Sum", (*cudafuse.LazyArray).Sum},
		{"Sort", (*cudafuse.LazyArray).Sort},
		{"Scan", func(bad *cudafuse.LazyArray) (*cudafuse.MaterializedPlan, error) { return bad.Scan("add") }},
		{"Replicate.Sum", func(bad *cudafuse.LazyArray) (*cudafuse.MaterializedPlan, error) { return bad.Replicate(2).Sum() }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := cudafuse.NewLazyContext()
			bad := cudafuse.NewLazyArray(ctx, []int{0, -1}, cudair.Float32)
			plan, err := tt.run(bad)
			if err == nil || plan != nil {
				t.Fatalf("%s = %v, %v; want refusal", tt.name, plan, err)
			}
			if n := ctx.Tracker.Snapshot().Allocations; n != 0 {
				t.Fatalf("refused graph recorded %d allocations", n)
			}
		})
	}
}

// TestLazyViewLoweringRefused checks that gather and replicate keep their explicit
// materializing boundary: no kernel is emitted for them and execution refuses
// the plan before any device call.
func TestLazyViewLoweringRefused(t *testing.T) {
	tests := []struct {
		op    string
		build func(x *cudafuse.LazyArray) *cudafuse.LazyArray
	}{
		{"gather", func(x *cudafuse.LazyArray) *cudafuse.LazyArray {
			return x.Gather(cudafuse.NewLazyArray(x.Ctx, []int{8}, cudair.Int32))
		}},
		{"replicate", func(x *cudafuse.LazyArray) *cudafuse.LazyArray { return x.Replicate(2) }},
	}
	for _, tt := range tests {
		t.Run(tt.op, func(t *testing.T) {
			ctx := cudafuse.NewLazyContext()
			x := cudafuse.NewLazyArray(ctx, []int{4}, cudair.Float32)
			view := tt.build(x.Map("relu"))
			if !slices.Equal(view.Shape, []int{8}) {
				t.Fatalf("view shape = %v, want [8]", view.Shape)
			}
			plan, err := view.Materialize()
			if err != nil {
				t.Fatalf("Materialize: %v", err)
			}
			found := false
			for _, r := range plan.Regions {
				if r.Boundary != tt.op {
					continue
				}
				found = true
				if r.Kernel != nil || r.Reason == "" || !cudafuse.IsMaterializingBoundary(r.Boundary) {
					t.Fatalf("view region kernel=%v reason=%q, want explicit refusal", r.Kernel, r.Reason)
				}
			}
			if !found {
				t.Fatalf("no %s boundary region in %+v", tt.op, plan.Regions)
			}
			if e, err := plan.Execute(context.Background(), nil, nil); err == nil || e != nil {
				t.Fatalf("Execute = %v, %v; want refusal", e, err)
			}
		})
	}
}

// Cycle's arguments are the result shape, not a repeat count: Cycle(4) on a
// [2 3] array yields its first four elements as a [4] array.
func ExampleLazyArray_Cycle_shape() {
	ctx := cudafuse.NewLazyContext()
	x := cudafuse.NewLazyArray(ctx, []int{2, 3}, cudair.Float32)
	fmt.Println(x.Cycle([]int{4}).Shape)
	fmt.Println(x.Cycle([]int{12}).Shape)
	fmt.Println(x.Cycle([]int{3, 4}).Shape)
	// Output:
	// [4]
	// [12]
	// [3 4]
}
