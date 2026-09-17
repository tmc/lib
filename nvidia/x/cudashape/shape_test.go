package cudashape_test

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

func TestShapeChecked(t *testing.T) {
	tests := []struct {
		name      string
		dims      []int
		wantErr   error
		wantElems int64
		wantBytes uint64
	}{
		{
			name:      "scalar",
			dims:      []int{},
			wantErr:   nil,
			wantElems: 1,
			wantBytes: 4,
		},
		{
			name:      "1D vector",
			dims:      []int{128},
			wantErr:   nil,
			wantElems: 128,
			wantBytes: 512,
		},
		{
			name:      "3D tensor",
			dims:      []int{2, 4, 8},
			wantErr:   nil,
			wantElems: 64,
			wantBytes: 256,
		},
		{
			name:      "zero dimension",
			dims:      []int{2, 0, 8},
			wantErr:   nil,
			wantElems: 0,
			wantBytes: 0,
		},
		{
			name:    "negative dimension",
			dims:    []int{2, -1, 8},
			wantErr: cudashape.ErrNegativeDimension,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, err := cudashape.New(tc.dims...)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("expected error %v, got %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			elems, err := s.NumElements()
			if err != nil {
				t.Fatalf("unexpected NumElements error: %v", err)
			}
			if elems != tc.wantElems {
				t.Errorf("elements: got %d, want %d", elems, tc.wantElems)
			}
			bytes, err := s.SizeBytes(4)
			if err != nil {
				t.Fatalf("unexpected SizeBytes error: %v", err)
			}
			if bytes != tc.wantBytes {
				t.Errorf("bytes: got %d, want %d", bytes, tc.wantBytes)
			}
		})
	}
}

func TestShapeOverflow(t *testing.T) {
	// Acceptance test: TestShapeOverflow
	tests := []struct {
		name string
		dims []int
	}{
		{
			name: "single extent overflow exceeding max limit",
			dims: []int{math.MaxInt32, math.MaxInt32},
		},
		{
			name: "multi-dim overflow beyond 48-bit address space",
			dims: []int{1024 * 1024, 1024 * 1024, 1024},
		},
		{
			name: "large product exceeding MaxByteLimit",
			dims: []int{1 << 25, 1 << 25},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, err := cudashape.New(tc.dims...)
			if err != nil {
				if !errors.Is(err, cudashape.ErrShapeOverflow) {
					t.Fatalf("expected ErrShapeOverflow on New, got %v", err)
				}
				return
			}
			// If New succeeded, SizeBytes must catch overflow
			_, err = s.SizeBytes(4)
			if !errors.Is(err, cudashape.ErrShapeOverflow) {
				t.Fatalf("expected ErrShapeOverflow from SizeBytes, got %v", err)
			}
		})
	}

	// Mutant control: request an overflowing allocation
	t.Run("mutant_overflowing_allocation", func(t *testing.T) {
		s, err := cudashape.New(1 << 20, 1 << 20) // 1 Tera-elements
		if err != nil {
			t.Fatalf("unexpected New error: %v", err)
		}
		// Requesting 4096 bytes per element will demand 4 PiB, far exceeding MaxByteLimit
		_, err = s.SizeBytes(4096)
		if !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Fatalf("mutant control failed: expected ErrShapeOverflow for 4 PiB allocation, got %v", err)
		}
	})
}

func TestShapeBindingRejected(t *testing.T) {
	// Acceptance test: TestShapeBindingRejected
	tests := []struct {
		name      string
		setup     func() (cudashape.SymbolicShape, *cudashape.Environment, error)
		expectErr error
	}{
		{
			name: "unbound symbol in shape",
			setup: func() (cudashape.SymbolicShape, *cudashape.Environment, error) {
				env := cudashape.NewEnvironment()
				_ = env.Bind("batch", 16)
				sym := cudashape.NewSymbolicShape(
					cudashape.NamedDim("batch"),
					cudashape.NamedDim("seq_len"), // un-bound!
				)
				return sym, env, nil
			},
			expectErr: cudashape.ErrUnboundSymbol,
		},
		{
			name: "contradictory binding",
			setup: func() (cudashape.SymbolicShape, *cudashape.Environment, error) {
				env := cudashape.NewEnvironment()
				if err := env.Bind("batch", 16); err != nil {
					return cudashape.SymbolicShape{}, nil, err
				}
				// Rebinding to different value must be rejected immediately
				err := env.Bind("batch", 32)
				return cudashape.SymbolicShape{}, env, err
			},
			expectErr: cudashape.ErrContradictoryBinding,
		},
		{
			name: "equality constraint violation between symbols",
			setup: func() (cudashape.SymbolicShape, *cudashape.Environment, error) {
				env := cudashape.NewEnvironment()
				_ = env.AddConstraint(cudashape.EqualSymbols("dim_a", "dim_b"))
				_ = env.Bind("dim_a", 128)
				err := env.Bind("dim_b", 256) // violates equality!
				return cudashape.SymbolicShape{}, env, err
			},
			expectErr: cudashape.ErrConstraintViolation,
		},
		{
			name: "divisibility constraint violation",
			setup: func() (cudashape.SymbolicShape, *cudashape.Environment, error) {
				env := cudashape.NewEnvironment()
				_ = env.AddConstraint(cudashape.DivisibleBy("hidden", 64))
				err := env.Bind("hidden", 100) // 100 is not divisible by 64
				return cudashape.SymbolicShape{}, env, err
			},
			expectErr: cudashape.ErrConstraintViolation,
		},
		{
			name: "range bound violation",
			setup: func() (cudashape.SymbolicShape, *cudashape.Environment, error) {
				env := cudashape.NewEnvironment()
				_ = env.AddConstraint(cudashape.RangeBound("heads", 1, 32))
				err := env.Bind("heads", 64) // 64 > 32
				return cudashape.SymbolicShape{}, env, err
			},
			expectErr: cudashape.ErrConstraintViolation,
		},
		{
			name: "negative dimension binding",
			setup: func() (cudashape.SymbolicShape, *cudashape.Environment, error) {
				env := cudashape.NewEnvironment()
				err := env.Bind("batch", -4)
				return cudashape.SymbolicShape{}, env, err
			},
			expectErr: cudashape.ErrNegativeDimension,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			symShape, env, err := tc.setup()
			if err != nil {
				if !errors.Is(err, tc.expectErr) {
					t.Fatalf("expected error %v, got %v", tc.expectErr, err)
				}
				return
			}
			_, err = symShape.Bind(env)
			if !errors.Is(err, tc.expectErr) {
				t.Fatalf("expected error %v on Bind, got %v", tc.expectErr, err)
			}
		})
	}
}

func TestLayoutsAndStrides(t *testing.T) {
	dims := []int{2, 3, 4}
	rmStrides, err := cudashape.RowMajorStrides(dims)
	if err != nil {
		t.Fatalf("RowMajorStrides: %v", err)
	}
	expectedRM := []int{12, 4, 1}
	for i := range expectedRM {
		if rmStrides[i] != expectedRM[i] {
			t.Errorf("RM stride[%d]: got %d, want %d", i, rmStrides[i], expectedRM[i])
		}
	}
	if !cudashape.IsContiguousRowMajor(dims, rmStrides) {
		t.Errorf("expected IsContiguousRowMajor to be true")
	}

	cmStrides, err := cudashape.ColumnMajorStrides(dims)
	if err != nil {
		t.Fatalf("ColumnMajorStrides: %v", err)
	}
	expectedCM := []int{1, 2, 6}
	for i := range expectedCM {
		if cmStrides[i] != expectedCM[i] {
			t.Errorf("CM stride[%d]: got %d, want %d", i, cmStrides[i], expectedCM[i])
		}
	}
	if !cudashape.IsContiguousColumnMajor(dims, cmStrides) {
		t.Errorf("expected IsContiguousColumnMajor to be true")
	}

	customStrides := []int{24, 4, 1}
	if cudashape.LayoutFromStrides(dims, customStrides) != cudashape.LayoutStrided {
		t.Errorf("expected LayoutStrided for non-contiguous strides")
	}

	byteStrides, err := cudashape.ByteStrides(4, rmStrides)
	if err != nil {
		t.Fatalf("ByteStrides: %v", err)
	}
	if byteStrides[0] != 48 || byteStrides[1] != 16 || byteStrides[2] != 4 {
		t.Errorf("unexpected byte strides: %v", byteStrides)
	}
}
