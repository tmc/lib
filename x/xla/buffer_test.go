package xla

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/openxla"
)

func TestBufferSize(t *testing.T) {
	for _, tt := range []struct {
		name string
		typ  ElementType
		dims []int64
		want int
		err  string
	}{
		{"scalar", F32, nil, 4, ""},
		{"matrix", F32, []int64{2, 3}, 24, ""},
		{"empty", F32, []int64{math.MaxInt64, 0}, 0, ""},
		{"complex", openxla.Buffer_Type_C128, []int64{3}, 48, ""},
		{"bf16", openxla.Buffer_Type_BF16, []int64{3}, 6, ""},
		{"negative", F32, []int64{-1}, 0, "negative"},
		{"negative empty", F32, []int64{0, -1}, 0, "negative"},
		{"byte overflow", F32, []int64{math.MaxInt64 / 2}, 0, "overflows"},
		{"shape overflow", U8, []int64{math.MaxInt64, 2}, 0, "overflows"},
		{"packed", openxla.Buffer_Type_S4, []int64{2}, 0, "unsupported"},
		{"invalid", openxla.Buffer_Type_INVALID, nil, 0, "unsupported"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := bufferSize(tt.typ, tt.dims)
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("got %d, %v; want %q", got, err, tt.err)
				}
			} else if err != nil || got != tt.want {
				t.Fatalf("got %d, %v; want %d", got, err, tt.want)
			}
		})
	}
}

func TestHostValidationBeforeDispatch(t *testing.T) {
	for _, tt := range []struct {
		name string
		data []byte
		dims []int64
		err  string
	}{
		{"short", make([]byte, 1), []int64{100}, "has 1 bytes, want 400"},
		{"long", make([]byte, 5), nil, "has 5 bytes, want 4"},
		{"negative", nil, []int64{-1}, "negative"},
		{"overflow", nil, []int64{math.MaxInt64}, "overflows"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			// A nil engine cannot dispatch. Validation must reject the input first.
			var e *Engine
			_, err := e.NewBufferFromHost(tt.data, F32, tt.dims, nil)
			if err == nil || !strings.Contains(err.Error(), tt.err) {
				t.Fatalf("got %v; want %q", err, tt.err)
			}
		})
	}
}

func TestHostShapeIsCopied(t *testing.T) {
	exec, _, _ := testExecutable(t, nil, false, false)
	dims := []int64{3}
	b, err := exec.engine.NewBufferFromHost([]byte{1, 2, 3}, U8, dims, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	dims[0] = 100
	if b.dims[0] != 3 {
		t.Fatalf("buffer shape follows caller mutation: %v", b.dims)
	}
}

func TestBufferShapeCopiesDimensions(t *testing.T) {
	b := &Buffer{wrapper: &openxla.BufferWrapper{}, dims: []int64{2, 3}, elemTyp: F32}
	typ, shape, err := b.Shape()
	if err != nil || typ != F32 {
		t.Fatalf("shape: %v %v", typ, err)
	}
	shape[0] = 99
	_, again, err := b.Shape()
	if err != nil || again[0] != 2 {
		t.Fatalf("shape aliases caller: %v %v", again, err)
	}
	b.wrapper = nil
	if _, _, err := b.Shape(); err == nil {
		t.Fatal("closed buffer shape succeeded")
	}
}
