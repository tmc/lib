package cudaoccupancy

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// TestArchitectureLimits pins each architecture's per-SM limits to the CUDA
// Programming Guide 13.x, Compute Capabilities, Tables 30 and 31.
func TestArchitectureLimits(t *testing.T) {
	tests := []struct {
		arch       string
		warps      int
		blocks     int
		smemPerSM  int
		smemOptin  int
		tmemCols   int
		maxCluster int
	}{
		{"sm_75", 32, 16, 64 << 10, 64 << 10, 0, 0},
		{"sm_80", 64, 32, 164 << 10, 163 << 10, 0, 0},
		{"sm_86", 48, 16, 100 << 10, 99 << 10, 0, 0},
		{"sm_87", 48, 16, 164 << 10, 163 << 10, 0, 0},
		{"sm_89", 48, 24, 100 << 10, 99 << 10, 0, 0},
		{"sm_90a", 64, 32, 228 << 10, 227 << 10, 0, 16},
		{"sm_100a", 64, 32, 228 << 10, 227 << 10, 512, 16},
		{"sm_103a", 64, 32, 228 << 10, 227 << 10, 512, 16},
		{"gb300", 64, 32, 228 << 10, 227 << 10, 512, 16},
		{"sm_110a", 48, 24, 228 << 10, 227 << 10, 512, 8},
		{"sm_120a", 48, 24, 100 << 10, 99 << 10, 0, 8},
		{"sm_121a", 48, 24, 100 << 10, 99 << 10, 0, 8},
	}
	for _, tt := range tests {
		t.Run(tt.arch, func(t *testing.T) {
			d, err := LookupDevice(tt.arch)
			if err != nil {
				t.Fatal(err)
			}
			got := []int{d.MaxWarpsPerSM, d.MaxBlocksPerSM, d.MaxSharedMemoryPerSM, d.MaxSharedMemoryOptin, d.TMEMColumnsPerSM, d.MaxClusterSize}
			want := []int{tt.warps, tt.blocks, tt.smemPerSM, tt.smemOptin, tt.tmemCols, tt.maxCluster}
			for i := range got {
				if got[i] != want[i] {
					t.Errorf("limits = %v, want %v", got, want)
					break
				}
			}
		})
	}
}

// TestHandTableAgreesWithProfiles checks the per-SM limits here against the
// datasheet profiles in cudadevicetable wherever both describe a target, so
// the two tables cannot drift apart.
func TestHandTableAgreesWithProfiles(t *testing.T) {
	compared := 0
	for _, name := range cudadevicetable.ProfileNames() {
		p, err := cudadevicetable.LookupProfile(name)
		if err != nil {
			t.Fatal(err)
		}
		d, err := LookupDevice(p.Target)
		if err != nil {
			t.Errorf("%s: profile target %s unknown here: %v", name, p.Target, err)
			continue
		}
		for _, c := range []struct {
			field string
			hand  int
			sheet cudadevicetable.Value
		}{
			{"shared memory per SM", d.MaxSharedMemoryPerSM, p.SharedPerSM},
			{"shared memory opt-in", d.MaxSharedMemoryOptin, p.SharedPerBlockOptin},
			{"threads per SM", d.MaxWarpsPerSM * d.WarpSize, p.ThreadsPerSM},
			{"blocks per SM", d.MaxBlocksPerSM, p.BlocksPerSM},
			{"TMEM bytes per SM", d.TMEMColumnsPerSM * 128 * 4, p.TMEMPerSM},
		} {
			compared++
			if int64(c.hand) != c.sheet.Value {
				t.Errorf("%s (%s) %s: cudaoccupancy has %d, cudadevicetable has %d (%s)", name, p.Target, c.field, c.hand, c.sheet.Value, c.sheet.Source)
			}
		}
	}
	if compared == 0 {
		t.Fatal("no profile was compared")
	}
}

func TestTensorMemoryAndClusterLimits(t *testing.T) {
	tests := []struct {
		name      string
		arch      string
		k         KernelResource
		threads   int
		blocks    int
		limitedBy string
	}{
		{"128 columns, 4 CTAs", "sm_100a", KernelResource{Registers: 32, TMEMColumns: 128}, 128, 4, LimitTensorMemory},
		{"512 columns, 1 CTA", "sm_103a", KernelResource{Registers: 32, TMEMColumns: 512}, 128, 1, LimitTensorMemory},
		{"32 columns, 16 CTAs", "sm_103a", KernelResource{Registers: 32, TMEMColumns: 32}, 64, 16, LimitTensorMemory},
		{"no TMEM on sm_121", "sm_121a", KernelResource{Registers: 32, TMEMColumns: 32}, 128, 0, LimitTensorMemory},
		{"no TMEM on sm_90", "sm_90a", KernelResource{Registers: 32, TMEMColumns: 128}, 128, 0, LimitTensorMemory},
		{"CTA pair in cluster of 2", "sm_100a", KernelResource{Registers: 32, TMEMColumns: 256, ClusterSize: 2, CTAPair: true}, 128, 2, LimitTensorMemory},
		{"CTA pair without cluster", "sm_100a", KernelResource{Registers: 32, CTAPair: true}, 128, 0, LimitCluster},
		{"CTA pair in odd cluster", "sm_100a", KernelResource{Registers: 32, ClusterSize: 3, CTAPair: true}, 128, 0, LimitCluster},
		{"cluster 16 on sm_120", "sm_120a", KernelResource{Registers: 32, ClusterSize: 16}, 128, 0, LimitCluster},
		{"cluster 16 on sm_100", "sm_100a", KernelResource{Registers: 32, ClusterSize: 16}, 128, 16, LimitNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rep, err := tt.k.Calculate(tt.arch, tt.threads, 0)
			if err != nil {
				t.Fatal(err)
			}
			if rep.ActiveBlocks != tt.blocks || rep.LimitingFactor != tt.limitedBy {
				t.Errorf("Calculate = %d blocks limited by %s, want %d by %s", rep.ActiveBlocks, rep.LimitingFactor, tt.blocks, tt.limitedBy)
			}
		})
	}
}
