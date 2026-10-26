package cudahostemu

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

func TestMultiMemLdReduceAndPeerPtr(t *testing.T) {
	const src = `
__global__ void multimem_collectives(
    float* sym_f32,
    __nv_bfloat16* sym_bf16,
    float* peer_buf,
    uint32_t* bar_flag,
    float* out_f32,
    __nv_bfloat16* out_bf16,
    float* out_peer,
    int elems_per_rank) {
    const int r = eg_emu_rank();
    const int nr = eg_emu_num_ranks();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems_per_rank) return;

    // Stage 1: Each rank populates its own slice of the symmetric buffers.
    const float fval = (float)(r + 1) * 1.25f + (float)idx * 0.5f;
    sym_f32[(size_t)r * elems_per_rank + idx] = fval;
    sym_bf16[(size_t)r * elems_per_rank + idx] = f2bf((float)(r + 1) * 0.5f + (float)idx * 0.25f);
    peer_buf[(size_t)r * elems_per_rank + idx] = (float)(r * 100 + idx);

    // Synchronize all ranks across NVLink before cross-rank reads/reductions.
    eg_emu_nvlink_barrier(bar_flag, nr, 1);

    // Stage 2: Perform multimem.ld_reduce across all ranks and ring peer reads.
    out_f32[(size_t)r * elems_per_rank + idx] = eg_emu_multimem_ld_reduce_add_f32(sym_f32, idx);
    out_bf16[(size_t)r * elems_per_rank + idx] = eg_emu_multimem_ld_reduce_add_bf16(sym_bf16, idx);

    const int next_rank = (r + 1) % nr;
    const size_t bytes_per_rank = (size_t)elems_per_rank * sizeof(float);
    const float* my_peer_base = peer_buf + (size_t)r * elems_per_rank;
    const float* remote_ptr = eg_emu_peer_ptr(my_peer_base, next_rank, r, bytes_per_rank);
    out_peer[(size_t)r * elems_per_rank + idx] = remote_ptr[idx];
}
`
	for _, numRanks := range []int{4, 8} {
		t.Run(string(rune('0'+numRanks))+"_ranks", func(t *testing.T) {
			const elemsPerRank = 64
			totalElems := numRanks * elemsPerRank

			symF32 := make([]byte, totalElems*4)
			symBF16 := make([]byte, totalElems*2)
			peerBuf := make([]byte, totalElems*4)
			barFlag := make([]byte, 4)
			outF32 := Poison(totalElems * 4)
			outBF16 := Poison(totalElems * 2)
			outPeer := Poison(totalElems * 4)

			err := RunMultiRank(context.Background(), src, numRanks, Launch{
				Kernel: "multimem_collectives",
				Grid:   Dim3{X: 2},
				Block:  Dim3{X: 32},
			},
				BufferArg("sym_f32", "float", symF32),
				BufferArg("sym_bf16", "__nv_bfloat16", symBF16),
				BufferArg("peer_buf", "float", peerBuf),
				BufferArg("bar_flag", "uint32_t", barFlag),
				BufferArg("out_f32", "float", outF32),
				BufferArg("out_bf16", "__nv_bfloat16", outBF16),
				BufferArg("out_peer", "float", outPeer),
				IntArg(elemsPerRank),
			)
			if errors.Is(err, ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunMultiRank(%d): %v", numRanks, err)
			}

			gotF32 := f32s(outF32)
			gotPeer := f32s(outPeer)
			for idx := 0; idx < elemsPerRank; idx++ {
				var wantF32 float32
				var wantBF16Sum float32
				for r := 0; r < numRanks; r++ {
					wantF32 += float32(r+1)*1.25 + float32(idx)*0.5
					b := bf16(float32(r+1)*0.5 + float32(idx)*0.25)
					wantBF16Sum += float32(fromBF16(b))
				}
				wantBF16 := bf16(wantBF16Sum)

				for r := 0; r < numRanks; r++ {
					off := r*elemsPerRank + idx
					if gotF32[off] != wantF32 {
						t.Fatalf("rank %d idx %d out_f32 = %v, want %v", r, idx, gotF32[off], wantF32)
					}
					gotB := binary.LittleEndian.Uint16(outBF16[2*off:])
					if gotB != wantBF16 {
						t.Fatalf("rank %d idx %d out_bf16 = %#x (%v), want %#x (%v)",
							r, idx, gotB, fromBF16(gotB), wantBF16, fromBF16(wantBF16))
					}
					nextRank := (r + 1) % numRanks
					wantP := float32(nextRank*100 + idx)
					if gotPeer[off] != wantP {
						t.Fatalf("rank %d idx %d out_peer = %v, want %v", r, idx, gotPeer[off], wantP)
					}
				}
			}
		})
	}
}

func TestMultiMemMulticastStore(t *testing.T) {
	const src = `
__global__ void multicast_broadcast(float* sym_buf, uint32_t* bar_flag, float* out, int elems_per_rank) {
    const int r = eg_emu_rank();
    const int nr = eg_emu_num_ranks();
    const int idx = threadIdx.x;
    if (r == 0 && idx < elems_per_rank) {
        eg_emu_multimem_st_f32(sym_buf, idx, (float)(idx * 3 + 7));
    }
    eg_emu_nvlink_barrier(bar_flag, nr, 1);
    if (idx < elems_per_rank) {
        out[(size_t)r * elems_per_rank + idx] = sym_buf[(size_t)r * elems_per_rank + idx];
    }
}
`
	const (
		numRanks     = 4
		elemsPerRank = 32
	)
	symBuf := make([]byte, numRanks*elemsPerRank*4)
	barFlag := make([]byte, 4)
	out := Poison(numRanks * elemsPerRank * 4)

	err := RunMultiRank(context.Background(), src, numRanks, Launch{
		Kernel: "multicast_broadcast",
		Grid:   Dim3{X: 1},
		Block:  Dim3{X: 32},
	},
		BufferArg("sym_buf", "float", symBuf),
		BufferArg("bar_flag", "uint32_t", barFlag),
		BufferArg("out", "float", out),
		IntArg(elemsPerRank),
	)
	if errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("RunMultiRank: %v", err)
	}
	got := f32s(out)
	for r := 0; r < numRanks; r++ {
		for idx := 0; idx < elemsPerRank; idx++ {
			want := float32(idx*3 + 7)
			if got[r*elemsPerRank+idx] != want {
				t.Fatalf("rank %d idx %d = %v, want %v", r, idx, got[r*elemsPerRank+idx], want)
			}
		}
	}
}

func TestMultiMemAdversarialBarrierCheck(t *testing.T) {
	tests := []struct {
		name string
		src  string
	}{
		{
			name: "multimem_ld_reduce_without_barrier",
			src: `
__global__ void bad_reduce(float* sym, float* out) {
    const int r = eg_emu_rank();
    const int idx = threadIdx.x;
    sym[r * 32 + idx] = (float)(r + 1);
    // Missing eg_emu_nvlink_barrier!
    out[r * 32 + idx] = eg_emu_multimem_ld_reduce_add_f32(sym, idx);
}
`,
		},
		{
			name: "remote_peer_ptr_without_barrier",
			src: `
__global__ void bad_peer(float* sym, float* out) {
    const int r = eg_emu_rank();
    const int nr = eg_emu_num_ranks();
    const int idx = threadIdx.x;
    sym[r * 32 + idx] = (float)(r + 10);
    // Missing eg_emu_nvlink_barrier!
    const float* peer = eg_emu_peer_ptr(sym + r * 32, (r + 1) % nr, r, 32 * sizeof(float));
    out[r * 32 + idx] = peer[idx];
}
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const numRanks = 4
			sym := make([]byte, numRanks*32*4)
			out := make([]byte, numRanks*32*4)
			err := RunMultiRank(context.Background(), tt.src, numRanks, Launch{
				Grid:  Dim3{X: 1},
				Block: Dim3{X: 32},
			},
				BufferArg("sym", "float", sym),
				BufferArg("out", "float", out),
			)
			if errors.Is(err, ErrNoCompiler) {
				t.Skip(err)
			}
			if !errors.Is(err, ErrNVLinkOrderingViolation) {
				t.Fatalf("got err = %v, want %v", err, ErrNVLinkOrderingViolation)
			}
		})
	}
}

func TestMultiMemPrePopulatedReadNeedsNoBarrier(t *testing.T) {
	const src = `
__global__ void read_only_reduce(const float* sym, float* out) {
    const int r = eg_emu_rank();
    const int idx = threadIdx.x;
    out[r * 32 + idx] = eg_emu_multimem_ld_reduce_add_f32(sym, idx);
}
`
	const numRanks = 4
	sym := make([]byte, numRanks*32*4)
	for i := 0; i < numRanks*32; i++ {
		binary.LittleEndian.PutUint32(sym[4*i:], math.Float32bits(float32(i+1)))
	}
	out := Poison(numRanks * 32 * 4)
	err := RunMultiRank(context.Background(), src, numRanks, Launch{
		Grid:  Dim3{X: 1},
		Block: Dim3{X: 32},
	},
		BufferArg("sym", "float", sym),
		BufferArg("out", "float", out),
	)
	if errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("RunMultiRank: %v", err)
	}
	got := f32s(out)
	for idx := 0; idx < 32; idx++ {
		var want float32
		for r := 0; r < numRanks; r++ {
			want += float32(r*32 + idx + 1)
		}
		for r := 0; r < numRanks; r++ {
			if got[r*32+idx] != want {
				t.Fatalf("rank %d idx %d = %v, want %v", r, idx, got[r*32+idx], want)
			}
		}
	}
}
