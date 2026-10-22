//go:build linux

package main

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
)

func TestCUDAV41AlignedQ8Roles(t *testing.T) {
	roles := []struct {
		name    string
		k, m    uint64
		aligned bool
	}{
		{"attn_q_a.weight", 5120, 1280, true}, {"attn_kv.weight", 5120, 512, true},
		{"attn_output_b.weight", 8192, 5120, true},
		{"ffn_gate_shexp.weight", 5120, 2304, true}, {"ffn_up_shexp.weight", 5120, 2304, true},
		{"ffn_down_shexp.weight", 2304, 5120, false},
		{"attn_q_b.weight", 1280, 32768, false}, {"attn_output_a.weight", 4096, 8192, false},
	}
	for layer := 0; layer < 40; layer++ {
		for _, role := range roles {
			name := fmt.Sprintf("blk.%d.%s", layer, role.name)
			t.Run(name, func(t *testing.T) {
				d := dsv41.TensorDescriptor{Name: name, Encoding: 8, Offset: 4096, Bytes: int64(role.k / 32 * 34 * role.m), Dimensions: []uint64{role.k, role.m}}
				if cudaV41AlignedQ8Eligible(d) != role.aligned || cudaV41AlignedQ8RoleMatches(d, d, layer, role.name) != role.aligned {
					t.Fatal("wrong named role admission")
				}
				for _, encoding := range []uint32{0, 1, 2, 12, 16, 39} {
					bad := d
					bad.Encoding = encoding
					if cudaV41AlignedQ8Eligible(bad) {
						t.Fatal("non-Q8 encoding admitted")
					}
				}
				if !role.aligned {
					return
				}
				for _, mutate := range []func(*dsv41.TensorDescriptor){
					func(d *dsv41.TensorDescriptor) { d.Bytes-- },
					func(d *dsv41.TensorDescriptor) { d.Offset = -1 },
					func(d *dsv41.TensorDescriptor) { d.Dimensions = []uint64{role.m, role.k} },
					func(d *dsv41.TensorDescriptor) { d.Dimensions = []uint64{role.k, role.m, 1} },
					func(d *dsv41.TensorDescriptor) { d.Name = "same-shape-not-a-model-role" },
				} {
					bad := d
					mutate(&bad)
					if cudaV41AlignedQ8Eligible(bad) || cudaV41AlignedQ8RoleMatches(bad, d, layer, role.name) {
						t.Fatal("malformed role admitted")
					}
				}
				foreign := d
				foreign.Name = fmt.Sprintf("blk.%d.%s", (layer+1)%40, role.name)
				if cudaV41AlignedQ8RoleMatches(foreign, foreign, layer, role.name) {
					t.Fatal("foreign role admitted")
				}
				if role.name == "ffn_gate_shexp.weight" {
					foreign = d
					foreign.Name = fmt.Sprintf("blk.%d.ffn_up_shexp.weight", layer)
					if cudaV41AlignedQ8RoleMatches(foreign, foreign, layer, role.name) {
						t.Fatal("same-shaped shared up admitted as gate")
					}
				}
				foreign = d
				foreign.Offset += 32
				if cudaV41AlignedQ8RoleMatches(d, foreign, layer, role.name) {
					t.Fatal("foreign model extent admitted")
				}
			})
		}
	}
	head := dsv41.TensorDescriptor{Name: "output.weight", Encoding: 8, Offset: 4096, Bytes: 5120 / 32 * 34 * 129280, Dimensions: []uint64{5120, 129280}}
	if !cudaV41AlignedQ8Eligible(head) || !cudaV41AlignedQ8RoleMatches(head, head, -1, "output.weight") {
		t.Fatal("output head refused")
	}
	for _, mutate := range []func(*dsv41.TensorDescriptor){
		func(d *dsv41.TensorDescriptor) { d.Encoding = 12 },
		func(d *dsv41.TensorDescriptor) { d.Bytes-- },
		func(d *dsv41.TensorDescriptor) { d.Offset = -1 },
		func(d *dsv41.TensorDescriptor) { d.Dimensions = []uint64{5120, 129152} },
	} {
		bad := head
		mutate(&bad)
		if cudaV41AlignedQ8Eligible(bad) || cudaV41AlignedQ8RoleMatches(bad, head, -1, "output.weight") {
			t.Fatal("malformed output head admitted")
		}
	}
	for _, layer := range []int{-2, 0, 39, 40} {
		if cudaV41AlignedQ8RoleMatches(head, head, layer, "output.weight") {
			t.Fatal("head bound as layer")
		}
	}
	for _, name := range []string{"blk.-1.attn_q_a.weight", "blk.40.attn_q_a.weight", "blk.01.attn_q_a.weight", "output_norm.weight", "token_embd.weight"} {
		bad := head
		bad.Name = name
		if cudaV41AlignedQ8Eligible(bad) {
			t.Fatal("noncanonical name admitted", name)
		}
	}
	if owner, err := newCUDAV41AlignedQ8Role(nil, nil, 0, "ffn_gate_shexp.weight", cudaV41Layer0Q8Weight{}); owner != nil || err == nil {
		t.Fatal("unbound model admitted")
	}
}

// Same full-value/literal-C/actual-mutant gate, not shape aliases. Each shape
// lives in one subtest; no all-layer duplicate model residency is constructed.
func TestCUDAV41AlignedSharedAndHeadValuesAndMutants(t *testing.T) {
	testCUDAV41AlignedQ8ValuesAndMutants(t, []v41AlignedShape{
		{"blk.1.ffn_gate_shexp.weight", 5120, 2304},
		{"blk.39.ffn_up_shexp.weight", 5120, 2304},
		{"output.weight", 5120, 129280},
	})
}
