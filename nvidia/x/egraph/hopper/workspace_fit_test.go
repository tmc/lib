package hopper

import (
	"fmt"
	"strings"
	"testing"
)

func TestWorkspaceFitPagedAttention(t *testing.T) {
	tests := []struct {
		name string
		attn PagedAttention
	}{
		{
			name: "h16_kv2_q1_c256",
			attn: PagedAttention{Name: "attn_q1", Hq: 16, Hkv: 2, QLen: 1, Page: 1, Q: E4M3, Out: BF16},
		},
		{
			name: "h16_kv2_q1_c320_fp8_gate",
			attn: PagedAttention{
				Name: "attn_c320", Hq: 16, Hkv: 2, QLen: 1, Page: 1,
				Chunk: 320, QKFP8: true, Q: E4M3, Out: BF16,
				Gate: true, GateStride: 2304, GateOffset: 1280,
			},
		},
		{
			name: "h128_kv16_q8_p64",
			attn: PagedAttention{Name: "attn_q8", Hq: 128, Hkv: 16, QLen: 8, Window: 2048, Page: 64, Q: E4M3, Out: BF16},
		},
		{
			name: "h16_kv2_hpb2",
			attn: PagedAttention{Name: "attn_hpb2", Hq: 16, Hkv: 2, QLen: 1, Page: 1, KVHeadsPerBlock: 2, Q: E4M3, Out: BF16},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, err := FitWorkspaceFormula(tt.attn.WorkspaceBytes)
			if err != nil {
				t.Fatalf("FitWorkspaceFormula: %v", err)
			}
			w2, err := tt.attn.WorkspaceFormula()
			if err != nil {
				t.Fatalf("PagedAttention.WorkspaceFormula: %v", err)
			}
			if w != w2 {
				t.Fatalf("FitWorkspaceFormula = %+v, WorkspaceFormula = %+v", w, w2)
			}
			if w.Group != attnGroup {
				t.Errorf("Group = %d, want %d", w.Group, attnGroup)
			}
			for _, batch := range []int{1, 2, 7, 16, 32} {
				for _, maxKeys := range []int{1, 64, 255, 256, 257, 320, 321, 4096, 4097, 80000} {
					got := w.Bytes(batch, maxKeys)
					want := tt.attn.WorkspaceBytes(batch, maxKeys)
					if got != want {
						t.Fatalf("Bytes(%d, %d) = %d, want %d", batch, maxKeys, got, want)
					}
				}
			}
		})
	}
}

func TestWorkspaceFitRejectsDisagreement(t *testing.T) {
	valid := PagedAttention{Name: "attn_q1", Hq: 16, Hkv: 2, QLen: 1, Page: 1, Q: E4M3, Out: BF16}
	tests := []struct {
		name string
		fn   func(batch, maxKeys int) int
		want string
	}{
		{
			name: "nil function",
			fn:   nil,
			want: "nil function",
		},
		{
			name: "nonpositive at origin",
			fn:   func(batch, maxKeys int) int { return 0 },
			want: "want > 0",
		},
		{
			name: "constant in maxKeys",
			fn:   func(batch, maxKeys int) int { return batch * 64 },
			want: "did not vary with maxKeys",
		},
		{
			name: "nonlinear in batch",
			fn: func(batch, maxKeys int) int {
				base := valid.WorkspaceBytes(batch, maxKeys)
				if batch > 1 {
					return base + 64
				}
				return base
			},
			want: "model",
		},
		{
			name: "quadratic in splits",
			fn: func(batch, maxKeys int) int {
				s := (maxKeys + 255) / 256
				return batch * (8 + 1024*s*s)
			},
			want: "model",
		},
		{
			name: "late perturbation at 33000",
			fn: func(batch, maxKeys int) int {
				v := valid.WorkspaceBytes(batch, maxKeys)
				if maxKeys == 33000 {
					return v + 4
				}
				return v
			},
			want: "batch=8 maxKeys=33000",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := FitWorkspaceFormula(tt.fn)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("FitWorkspaceFormula() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func ExampleFitWorkspaceFormula() {
	a := PagedAttention{Name: "attn", Hq: 16, Hkv: 2, QLen: 1, Page: 1, Q: E4M3, Out: BF16}
	w, err := FitWorkspaceFormula(a.WorkspaceBytes)
	fmt.Println(err == nil, w.Chunk, w.Group, w.Bytes(4, 8192) == a.WorkspaceBytes(4, 8192))
	// Output:
	// true 256 16 true
}
