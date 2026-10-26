package cudahostemu

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func checkHazards(t *testing.T, src string, launch Launch, args ...Arg) *HazardReport {
	t.Helper()
	rep, err := CheckHazards(context.Background(), src, launch, args...)
	if errors.Is(err, ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatalf("CheckHazards: %v", err)
	}
	return rep
}

func TestHazardSharedRace(t *testing.T) {
	tests := []struct {
		name     string
		src      string
		block    int
		wantRace bool
	}{
		{
			name: "warp 0 writes smem[0] and warp 1 reads without syncthreads",
			src: `
__global__ void k(float* out) {
    __shared__ float smem[64];
    if (threadIdx.x == 0) {
        smem[0] = 42.0f;
    }
    if (threadIdx.x == 32) {
        out[0] = smem[0];
    }
}`,
			block:    64,
			wantRace: true,
		},
		{
			name: "warp 0 writes smem[0] and warp 1 reads with syncthreads",
			src: `
__global__ void k(float* out) {
    __shared__ float smem[64];
    if (threadIdx.x == 0) {
        smem[0] = 42.0f;
    }
    __syncthreads();
    if (threadIdx.x == 32) {
        out[0] = smem[0];
    }
}`,
			block:    64,
			wantRace: false,
		},
		{
			name: "explicit eg_emu_smem_write and eg_emu_smem_read without barrier",
			src: `
__global__ void k(uint32_t* out) {
    uint32_t* smem = (uint32_t*)eg_emu_dynamic_smem();
    if (threadIdx.x == 0) {
        eg_emu_smem_write(&smem[0], sizeof(uint32_t));
        smem[0] = 99u;
    }
    if (threadIdx.x == 32) {
        eg_emu_smem_read(&smem[0], sizeof(uint32_t));
        out[0] = smem[0];
    }
}`,
			block:    64,
			wantRace: true,
		},
		{
			name: "explicit eg_emu_smem_write and eg_emu_smem_read synchronized by bar_sync",
			src: `
__global__ void k(uint32_t* out) {
    uint32_t* smem = (uint32_t*)eg_emu_dynamic_smem();
    if (threadIdx.x == 0) {
        eg_emu_smem_write(&smem[0], sizeof(uint32_t));
        smem[0] = 99u;
    }
    eg_emu_bar_sync(1, 64);
    if (threadIdx.x == 32) {
        eg_emu_smem_read(&smem[0], sizeof(uint32_t));
        out[0] = smem[0];
    }
}`,
			block:    64,
			wantRace: false,
		},
		{
			name: "same warp write and read without CTA barrier",
			src: `
__global__ void k(float* out) {
    __shared__ float smem[32];
    if (threadIdx.x == 0) {
        smem[0] = 7.0f;
    }
    __syncwarp();
    if (threadIdx.x == 1) {
        out[0] = smem[0];
    }
}`,
			block:    32,
			wantRace: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rep := checkHazards(t, tt.src, Launch{
				Grid:        Dim3{X: 1},
				Block:       Dim3{X: tt.block},
				SharedBytes: 1024,
			})
			if got := rep.Has(HazardSharedRace); got != tt.wantRace {
				t.Fatalf("Has(%q) = %v, want %v (hazards: %+v)", HazardSharedRace, got, tt.wantRace, rep.Hazards)
			}
			if !tt.wantRace && len(rep.Hazards) != 0 {
				t.Fatalf("got unexpected hazards: %+v", rep.Hazards)
			}
		})
	}
}

func TestHazardDSMEMBeforeWait(t *testing.T) {
	tests := []struct {
		name      string
		src       string
		wantDSMEM bool
	}{
		{
			name: "map_shared_rank to peer CTA before cluster_wait",
			src: `
__global__ void k(int* out) {
    EG_EMU_SHARED(int, smem, [32]);
    if (threadIdx.x == 0) {
        smem[0] = 100 + blockIdx.x;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int* peer = eg_emu_map_shared_rank(smem, 1);
        out[0] = peer[0];
    }
}`,
			wantDSMEM: true,
		},
		{
			name: "cluster_arrive and cluster_wait before map_shared_rank",
			src: `
__global__ void k(int* out) {
    EG_EMU_SHARED(int, smem, [32]);
    if (threadIdx.x == 0) {
        smem[0] = 100 + blockIdx.x;
    }
    eg_emu_cluster_arrive();
    eg_emu_cluster_wait();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int* peer = eg_emu_map_shared_rank(smem, 1);
        out[0] = peer[0];
    }
}`,
			wantDSMEM: false,
		},
		{
			name: "peer CTA exits early before DSMEM access",
			src: `
__global__ void k(int* out) {
    EG_EMU_SHARED(int, smem, [32]);
    if (blockIdx.x == 1) {
        return;
    }
    eg_emu_cluster_arrive();
    eg_emu_cluster_wait();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int* peer = eg_emu_map_shared_rank(smem, 1);
        out[0] = peer[0];
    }
}`,
			wantDSMEM: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rep := checkHazards(t, tt.src, Launch{
				Grid:        Dim3{X: 2},
				Block:       Dim3{X: 32},
				Cluster:     2,
				SharedBytes: 1024,
			})
			if got := rep.Has(HazardDSMEMBeforeWait); got != tt.wantDSMEM {
				t.Fatalf("Has(%q) = %v, want %v (hazards: %+v)", HazardDSMEMBeforeWait, got, tt.wantDSMEM, rep.Hazards)
			}
			if !tt.wantDSMEM && len(rep.Hazards) != 0 {
				t.Fatalf("got unexpected hazards: %+v", rep.Hazards)
			}
		})
	}
}

func TestHazardWGMMADivergence(t *testing.T) {
	tests := []struct {
		name           string
		src            string
		wantDivergence bool
	}{
		{
			name: "half warpgroup calls eg_emu_wgmma_commit_group",
			src: `
__global__ void k() {
    if (threadIdx.x < 64) {
        eg_emu_wgmma_commit_group();
    }
}`,
			wantDivergence: true,
		},
		{
			name: "thread-dependent WGMMA loop bound across warps in warpgroup",
			src: `
__global__ void k(float* out) {
    float d[32] = {};
    uint64_t da = 0, db = 0;
    eg_emu_wgmma_fence();
    int trips = 1 + (threadIdx.x / 32);
    for (int step = 0; step < trips; ++step) {
        eg_emu_wgmma_m64n64k16_f16(d, da, db);
    }
    eg_emu_wgmma_commit_group();
    eg_emu_wgmma_wait_group(0);
    out[threadIdx.x] = d[0];
}`,
			wantDivergence: true,
		},
		{
			name: "uniform 128-thread warpgroup WGMMA loop",
			src: `
__global__ void k(float* out) {
    float d[32] = {};
    uint64_t da = 0, db = 0;
    eg_emu_wgmma_fence();
    for (int step = 0; step < 4; ++step) {
        eg_emu_wgmma_m64n64k16_f16(d, da, db);
    }
    eg_emu_wgmma_commit_group();
    eg_emu_wgmma_wait_group(0);
    out[threadIdx.x] = d[0];
}`,
			wantDivergence: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rep := checkHazards(t, tt.src, Launch{
				Grid:        Dim3{X: 1},
				Block:       Dim3{X: 128},
				SharedBytes: 16384,
			})
			if got := rep.Has(HazardWGMMADivergence); got != tt.wantDivergence {
				t.Fatalf("Has(%q) = %v, want %v (hazards: %+v)", HazardWGMMADivergence, got, tt.wantDivergence, rep.Hazards)
			}
			if tt.wantDivergence {
				for _, h := range rep.Hazards {
					if h.Kind == HazardWGMMADivergence && !strings.Contains(h.Detail, "C7519") {
						t.Errorf("expected C7519 in divergence detail, got %q", h.Detail)
					}
				}
			} else if len(rep.Hazards) != 0 {
				t.Fatalf("got unexpected hazards: %+v", rep.Hazards)
			}
		})
	}
}

func TestHazardNamedBarrierCount(t *testing.T) {
	tests := []struct {
		name         string
		src          string
		block        int
		wantMismatch bool
	}{
		{
			name: "only 32 threads arrive at eg_emu_bar_sync(1, 64)",
			src: `
__global__ void k() {
    if (threadIdx.x < 32) {
        eg_emu_bar_sync(1, 64);
    }
}`,
			block:        64,
			wantMismatch: true,
		},
		{
			name: "non-multiple of 32 count at eg_emu_bar_sync",
			src: `
__global__ void k() {
    eg_emu_bar_sync(1, 48);
}`,
			block:        64,
			wantMismatch: true,
		},
		{
			name: "partial named barrier hits syncthreads",
			src: `
__global__ void k() {
    if (threadIdx.x < 32) {
        eg_emu_bar_sync(2, 64);
    } else {
        __syncthreads();
    }
}`,
			block:        64,
			wantMismatch: true,
		},
		{
			name: "matching 64 threads arrive at eg_emu_bar_sync(1, 64)",
			src: `
__global__ void k() {
    if (threadIdx.x < 64) {
        eg_emu_bar_sync(1, 64);
    }
}`,
			block:        64,
			wantMismatch: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rep := checkHazards(t, tt.src, Launch{
				Grid:  Dim3{X: 1},
				Block: Dim3{X: tt.block},
			})
			if got := rep.Has(HazardNamedBarrierCount); got != tt.wantMismatch {
				t.Fatalf("Has(%q) = %v, want %v (hazards: %+v)", HazardNamedBarrierCount, got, tt.wantMismatch, rep.Hazards)
			}
			if !tt.wantMismatch && len(rep.Hazards) != 0 {
				t.Fatalf("got unexpected hazards: %+v", rep.Hazards)
			}
		})
	}
}
