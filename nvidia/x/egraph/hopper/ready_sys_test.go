package hopper

import (
	"fmt"
	"strings"
	"testing"
)

func TestReadySysPreludeEmission(t *testing.T) {
	tests := []struct {
		name     string
		mode     SysBarrierMode
		wantStr  string
		wantSub  []string
		forbid   []string
	}{
		{
			name:    "peer_atomic",
			mode:    SysBarrierPeerAtomic,
			wantStr: "peer_atomic",
			wantSub: []string{
				"EG_DEV void eg_ready_signal_sys(unsigned long long* slot, unsigned int epoch, unsigned int units = 1)",
				"EG_DEV void eg_ready_wait_sys(unsigned long long* slot, unsigned int epoch, unsigned int need)",
				`asm volatile("fence.acq_rel.sys;" ::: "memory");`,
				`atomicMax(slot, (unsigned long long)epoch << 32);`,
				`atomicAdd_system(slot, (unsigned long long)units);`,
				`asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");`,
				`__nanosleep(ns);`,
				`eg_emu_ready_signal(slot, epoch, units);`,
				`eg_emu_ready_wait(slot, epoch, need);`,
			},
			forbid: []string{
				"multimem.red.release.sys.global.add.u64",
			},
		},
		{
			name:    "multimem",
			mode:    SysBarrierMultimem,
			wantStr: "multimem",
			wantSub: []string{
				"EG_DEV void eg_ready_signal_sys(unsigned long long* slot, unsigned int epoch, unsigned int units = 1)",
				"EG_DEV void eg_ready_wait_sys(unsigned long long* slot, unsigned int epoch, unsigned int need)",
				`asm volatile("fence.acq_rel.sys;" ::: "memory");`,
				`atomicMax(slot, (unsigned long long)epoch << 32);`,
				`#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !defined(EG_HOST_EMU)`,
				`asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;" :: "l"(slot), "l"((unsigned long long)units) : "memory");`,
				`atomicAdd_system(slot, (unsigned long long)units);`,
				`asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");`,
				`__nanosleep(ns);`,
				`eg_emu_ready_signal(slot, epoch, units);`,
				`eg_emu_ready_wait(slot, epoch, need);`,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.mode.String(); got != tc.wantStr {
				t.Errorf("mode.String() = %q, want %q", got, tc.wantStr)
			}
			src := ReadySysPrelude(tc.mode)
			if tc.mode == SysBarrierPeerAtomic && src != ReadySysPreludePeer {
				t.Error("ReadySysPrelude(SysBarrierPeerAtomic) != ReadySysPreludePeer")
			}
			if tc.mode == SysBarrierMultimem && src != ReadySysPreludeMultimem {
				t.Error("ReadySysPrelude(SysBarrierMultimem) != ReadySysPreludeMultimem")
			}
			for _, sub := range tc.wantSub {
				if !strings.Contains(src, sub) {
					t.Errorf("ReadySysPrelude(%s) missing substring %q", tc.mode, sub)
				}
			}
			for _, bad := range tc.forbid {
				if strings.Contains(src, bad) {
					t.Errorf("ReadySysPrelude(%s) unexpectedly contains %q", tc.mode, bad)
				}
			}
		})
	}
}

func TestReadySysEpochBarrierSource(t *testing.T) {
	tests := []struct {
		name      string
		cfg       SysEpochBarrierConfig
		wantNeed  int
		wantSub   []string
		wantError bool
	}{
		{
			name: "tp2_peer_default_name",
			cfg: SysEpochBarrierConfig{
				WorldSize:     2,
				BlocksPerRank: 4,
				Slot:          1,
				Mode:          SysBarrierPeerAtomic,
			},
			wantNeed: 8,
			wantSub: []string{
				"EG_DEV void sys_epoch_barrier_sync(unsigned long long* ready, unsigned int epoch)",
				"eg_ready_signal_sys(&ready[1], epoch, 1);",
				"eg_ready_wait_sys(&ready[1], epoch, 8);",
				"__global__ void __launch_bounds__(256) sys_epoch_barrier(",
				"const dim3 grid(8), block(256);",
				"atomicAdd_system",
			},
		},
		{
			name: "tp4_multimem_named",
			cfg: SysEpochBarrierConfig{
				Name:          "tp4_nvls_barrier",
				WorldSize:     4,
				BlocksPerRank: 2,
				Slot:          3,
				Mode:          SysBarrierMultimem,
			},
			wantNeed: 8,
			wantSub: []string{
				"EG_DEV void tp4_nvls_barrier_sync(unsigned long long* ready, unsigned int epoch)",
				"eg_ready_signal_sys(&ready[3], epoch, 1);",
				"eg_ready_wait_sys(&ready[3], epoch, 8);",
				"multimem.red.release.sys.global.add.u64",
				"ld.acquire.sys.global.u64",
				"fence.acq_rel.sys;",
			},
		},
		{
			name: "tp8_multimem_default_blocks",
			cfg: SysEpochBarrierConfig{
				Name:      "tp8_barrier",
				WorldSize: 8,
				Slot:      0,
				Mode:      SysBarrierMultimem,
			},
			wantNeed: 8,
			wantSub: []string{
				"eg_ready_signal_sys(&ready[0], epoch, 1);",
				"eg_ready_wait_sys(&ready[0], epoch, 8);",
				"const dim3 grid(8), block(256);",
			},
		},
		{
			name:      "bad_name_eg_prefix",
			cfg:       SysEpochBarrierConfig{Name: "eg_reserved", WorldSize: 2},
			wantError: true,
		},
		{
			name:      "zero_world_size",
			cfg:       SysEpochBarrierConfig{Name: "b", WorldSize: 0},
			wantError: true,
		},
		{
			name:      "negative_blocks_per_rank",
			cfg:       SysEpochBarrierConfig{Name: "b", WorldSize: 2, BlocksPerRank: -1},
			wantError: true,
		},
		{
			name:      "negative_slot",
			cfg:       SysEpochBarrierConfig{Name: "b", WorldSize: 2, Slot: -1},
			wantError: true,
		},
		{
			name:      "invalid_mode",
			cfg:       SysEpochBarrierConfig{Name: "b", WorldSize: 2, Mode: SysBarrierMode(99)},
			wantError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src, err := EmitSysEpochBarrierSource(tc.cfg)
			if tc.wantError {
				if err == nil {
					t.Fatalf("EmitSysEpochBarrierSource(%+v) succeeded, want error", tc.cfg)
				}
				return
			}
			if err != nil {
				t.Fatalf("EmitSysEpochBarrierSource(%+v): %v", tc.cfg, err)
			}
			if got := tc.cfg.TotalNeed(); got != tc.wantNeed {
				t.Errorf("TotalNeed() = %d, want %d", got, tc.wantNeed)
			}
			for _, sub := range tc.wantSub {
				if !strings.Contains(src, sub) {
					t.Errorf("emitted source missing %q\n--- source ---\n%s", sub, src)
				}
			}
		})
	}
}

func TestReadySysEmulatedMultiRank(t *testing.T) {
	for _, mode := range []SysBarrierMode{SysBarrierPeerAtomic, SysBarrierMultimem} {
		t.Run(mode.String(), func(t *testing.T) {
			const (
				ranks         = 4
				blocksPerRank = 2
				totalBlocks   = ranks * blocksPerRank
			)
			src := ReadySysPrelude(mode) + fmt.Sprintf(`
__global__ void sys_allreduce_pipe(const float* in, float* shared_sum, float* out, unsigned long long* ready, unsigned int epoch) {
    const int bid = blockIdx.x, tid = threadIdx.x;
    if (bid < %[1]d) {
        // Producer CTAs across %[2]d simulated ranks atomically accumulate into shared_sum.
        atomicAdd(&shared_sum[tid], in[bid * 32 + tid] * (float)epoch);
        __syncthreads();
        if (tid == 0) {
            eg_ready_signal_sys(&ready[0], epoch, 1);
        }
        return;
    }
    // Consumer CTAs wait at system scope until all %[1]d producer CTAs across all ranks signal.
    if (tid == 0) {
        eg_ready_wait_sys(&ready[0], epoch, %[1]d);
    }
    __syncthreads();
    const int cid = bid - %[1]d;
    out[cid * 32 + tid] = shared_sum[tid] + (float)(cid + 1);
}
`, totalBlocks, ranks)

			in := make([]float32, totalBlocks*32)
			for i := range in {
				in[i] = float32((i%7)+1) * 0.5
			}
			launch := strings.Join([]string{
				fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(32), 0, sys_allreduce_pipe, in, sum1, out1, ready, 1u)", 2*totalBlocks),
				fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(32), 0, sys_allreduce_pipe, in, sum2, out2, ready, 2u)", 2*totalBlocks),
			}, ";\n    ")

			res := runEmu(t, src, launch,
				emuBuf{"in", "float", f32Bytes(in)},
				emuBuf{"sum1", "float", make([]byte, 4*32)},
				emuBuf{"sum2", "float", make([]byte, 4*32)},
				emuBuf{"out1", "float", poison(4 * totalBlocks * 32)},
				emuBuf{"out2", "float", poison(4 * totalBlocks * 32)},
				emuBuf{"ready", "unsigned long long", make([]byte, 8)},
			)

			for ep, outName := range []string{"out1", "out2"} {
				epoch := float32(ep + 1)
				got := bytesF32(res[outName])
				for cid := 0; cid < totalBlocks; cid++ {
					for tid := 0; tid < 32; tid++ {
						var wantSum float32
						for b := 0; b < totalBlocks; b++ {
							wantSum += in[b*32+tid] * epoch
						}
						want := wantSum + float32(cid+1)
						if got[cid*32+tid] != want {
							t.Fatalf("epoch %d out[%d,%d] = %g, want %g", ep+1, cid, tid, got[cid*32+tid], want)
						}
					}
				}
			}

			gotReady := bytesU64(res["ready"])
			if want := PackReadyWord(2, totalBlocks); gotReady[0] != want {
				t.Fatalf("ready[0] = %#x, want %#x", gotReady[0], want)
			}
		})
	}
}
