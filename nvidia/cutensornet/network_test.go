package cutensornet

import (
	"strings"
	"testing"
)

// TestNewNetworkRejects checks the descriptions NewNetwork refuses before
// calling the library. Each of these is accepted by the C API as a well-formed
// argument list and then either fails obscurely or reads past the end of an
// array, so catching them in Go is the point. Every case here returns before
// any C call, so the test needs no GPU and no library.
func TestNewNetworkRejects(t *testing.T) {
	matmul := func() []Tensor {
		return []Tensor{
			{Modes: []int32{'m', 'k'}, Extents: []int64{4, 3}},
			{Modes: []int32{'k', 'n'}, Extents: []int64{3, 2}},
		}
	}
	for _, tt := range []struct {
		name   string
		inputs []Tensor
		output *Tensor
		want   string
	}{
		{
			name:   "no inputs",
			inputs: nil,
			want:   "no input tensors",
		},
		{
			name:   "no modes",
			inputs: []Tensor{{}},
			want:   "has no modes",
		},
		{
			name:   "extent count mismatch",
			inputs: []Tensor{{Modes: []int32{'i', 'j'}, Extents: []int64{4}}},
			want:   "2 modes but 1 extents",
		},
		{
			name:   "stride count mismatch",
			inputs: []Tensor{{Modes: []int32{'i', 'j'}, Extents: []int64{4, 3}, Strides: []int64{1}}},
			want:   "2 modes but 1 strides",
		},
		{
			name:   "zero extent",
			inputs: []Tensor{{Modes: []int32{'i'}, Extents: []int64{0}}},
			want:   "want >= 1",
		},
		{
			name:   "negative extent",
			inputs: []Tensor{{Modes: []int32{'i'}, Extents: []int64{-4}}},
			want:   "want >= 1",
		},
		{
			name:   "repeated mode within one tensor",
			inputs: []Tensor{{Modes: []int32{'i', 'i'}, Extents: []int64{4, 4}}},
			want:   "repeats mode label",
		},
		{
			name: "contracted mode with inconsistent extent",
			inputs: []Tensor{
				{Modes: []int32{'m', 'k'}, Extents: []int64{4, 3}},
				{Modes: []int32{'k', 'n'}, Extents: []int64{5, 2}},
			},
			want: "elsewhere",
		},
		{
			name:   "output mode absent from the inputs",
			inputs: matmul(),
			output: &Tensor{Modes: []int32{'m', 'z'}, Extents: []int64{4, 2}},
			want:   "does not appear in any input",
		},
		{
			name:   "output extent disagrees with the inputs",
			inputs: matmul(),
			output: &Tensor{Modes: []int32{'m', 'n'}, Extents: []int64{4, 7}},
			want:   "in the output tensor",
		},
		{
			name:   "output stride count mismatch",
			inputs: matmul(),
			output: &Tensor{Modes: []int32{'m', 'n'}, Extents: []int64{4, 2}, Strides: []int64{1}},
			want:   "2 modes but 1 strides",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var h Handle
			net, err := h.NewNetwork(tt.inputs, tt.output, R32F, Compute32F)
			if err == nil {
				if net != nil {
					net.Close()
				}
				t.Fatalf("NewNetwork accepted %s", tt.name)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("NewNetwork error = %q, want it to mention %q", err, tt.want)
			}
			if net != nil {
				t.Error("NewNetwork returned a network alongside an error")
			}
		})
	}
}

// TestNewNetworkAcceptsValidShapes checks that the descriptions NewNetwork is
// meant to accept get past validation and reach the library. The library is not
// installed on the machine that runs this test, so what is asserted is that the
// failure is the library's, not a validation message: an over-strict validator
// would fail here.
func TestNewNetworkAcceptsValidShapes(t *testing.T) {
	if Loaded() {
		t.Skip("libcutensornet is loaded; this test distinguishes validation errors from load errors and needs the library absent")
	}
	for _, tt := range []struct {
		name   string
		inputs []Tensor
		output *Tensor
	}{
		{
			name: "matrix product, inferred output",
			inputs: []Tensor{
				{Modes: []int32{'m', 'k'}, Extents: []int64{4, 3}},
				{Modes: []int32{'k', 'n'}, Extents: []int64{3, 2}},
			},
		},
		{
			name: "matrix product, explicit row-major strides",
			inputs: []Tensor{
				{Modes: []int32{'m', 'k'}, Extents: []int64{4, 3}, Strides: []int64{3, 1}},
				{Modes: []int32{'k', 'n'}, Extents: []int64{3, 2}, Strides: []int64{2, 1}},
			},
			output: &Tensor{Modes: []int32{'m', 'n'}, Extents: []int64{4, 2}, Strides: []int64{2, 1}},
		},
		{
			name: "mixed strided and column-major inputs",
			inputs: []Tensor{
				{Modes: []int32{'m', 'k'}, Extents: []int64{4, 3}, Strides: []int64{3, 1}},
				{Modes: []int32{'k', 'n'}, Extents: []int64{3, 2}},
			},
		},
		{
			name: "full reduction to a scalar",
			inputs: []Tensor{
				{Modes: []int32{'i'}, Extents: []int64{8}},
				{Modes: []int32{'i'}, Extents: []int64{8}},
			},
			output: &Tensor{},
		},
		{
			name: "constant operand qualifier",
			inputs: []Tensor{
				{Modes: []int32{'m', 'k'}, Extents: []int64{4, 3}, Qualifiers: TensorQualifiers{IsConstant: 1}},
				{Modes: []int32{'k', 'n'}, Extents: []int64{3, 2}},
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var h Handle
			net, err := h.NewNetwork(tt.inputs, tt.output, R32F, Compute32F)
			if err == nil {
				net.Close()
				t.Fatal("NewNetwork succeeded without the library loaded")
			}
			if strings.HasPrefix(err.Error(), "cutensornet: ") && !strings.Contains(err.Error(), "cannot open") {
				t.Errorf("NewNetwork rejected a valid description: %v", err)
			}
		})
	}
}

// TestNetworkClosedRejectsUse checks that a closed network does not reach the
// library with a destroyed descriptor.
func TestNetworkClosedRejectsUse(t *testing.T) {
	n := &Network{closed: true}
	if err := n.Close(); err != nil {
		t.Errorf("Close on a closed network = %v, want nil", err)
	}
	if err := n.SetInt32(NetworkComputeType, int32(Compute32F)); err != ErrNetworkClosed {
		t.Errorf("SetInt32 after Close = %v, want ErrNetworkClosed", err)
	}
	if _, err := n.Int32(NetworkComputeType); err != ErrNetworkClosed {
		t.Errorf("Int32 after Close = %v, want ErrNetworkClosed", err)
	}
	w := &Workspace{}
	if _, err := w.NewPlan(n, 0); err != ErrNetworkClosed {
		t.Errorf("NewPlan on a closed network = %v, want ErrNetworkClosed", err)
	}
	if err := w.ComputeContractionSizes(n, 0); err != ErrNetworkClosed {
		t.Errorf("ComputeContractionSizes on a closed network = %v, want ErrNetworkClosed", err)
	}
	if err := w.ComputeContractionSizes(nil, 0); err != ErrNetworkClosed {
		t.Errorf("ComputeContractionSizes(nil) = %v, want ErrNetworkClosed", err)
	}
}
