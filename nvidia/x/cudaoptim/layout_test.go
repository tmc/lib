package cudaoptim

import (
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

// TestABI pins the Go mirrors of the kernel structs to the layout the CUDA
// source declares.
func TestABI(t *testing.T) {
	tests := []struct {
		name       string
		size, want uintptr
	}{
		{"cudaoptim_tensor", unsafe.Sizeof(tensorEntry{}), 48},
		{"cudaoptim_tensor.n", unsafe.Offsetof(tensorEntry{}.n), 32},
		{"cudaoptim_tensor.weight_decay", unsafe.Offsetof(tensorEntry{}.wd), 44},
		{"cudaoptim_chunk", unsafe.Sizeof(chunkEntry{}), 16},
		{"cudaoptim_chunk.start", unsafe.Offsetof(chunkEntry{}.start), 8},
		{"cudaoptim_state", unsafe.Sizeof(State{}), 32},
		{"cudaoptim_state.lr", unsafe.Offsetof(State{}.LR), 8},
		{"cudaoptim_state.clip_scale", unsafe.Offsetof(State{}.ClipScale), 16},
		{"cudaoptim_state.inv_sqrt_bias2", unsafe.Offsetof(State{}.invSqrtBias2), 24},
	}
	for _, tt := range tests {
		if tt.size != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, tt.size, tt.want)
		}
	}
	for _, name := range []string{"struct cudaoptim_tensor", "struct cudaoptim_chunk", "struct cudaoptim_state"} {
		if !strings.Contains(kernelSource, name) {
			t.Errorf("kernel source does not declare %s", name)
		}
	}
}

func TestSplitChunks(t *testing.T) {
	tests := []struct {
		name string
		ns   []int64
		size int64
		want []chunkEntry
	}{
		{"none", nil, 4, nil},
		{"exact", []int64{8}, 4, []chunkEntry{{tensor: 0, start: 0}, {tensor: 0, start: 4}}},
		{"ragged", []int64{5, 1, 4}, 4, []chunkEntry{
			{tensor: 0, start: 0}, {tensor: 0, start: 4}, {tensor: 1, start: 0}, {tensor: 2, start: 0},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var entries []tensorEntry
			for _, n := range tt.ns {
				entries = append(entries, tensorEntry{n: n})
			}
			got, err := splitChunks(entries, tt.size)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("splitChunks = %v, want %v", got, tt.want)
			}
		})
	}
}
