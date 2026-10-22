package cudacluster_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacluster"
)

func writeSyntheticFile(t *testing.T, root, rel, content string) {
	t.Helper()
	full := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", full, err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%s): %v", full, err)
	}
}

func TestReadNUMANodeHelpers(t *testing.T) {
	if got := cudacluster.NormalizeUnsetNUMA(-1); got != 0 {
		t.Fatalf("NormalizeUnsetNUMA(-1) = %d, want 0", got)
	}
	if got := cudacluster.NormalizeUnsetNUMA(2); got != 2 {
		t.Fatalf("NormalizeUnsetNUMA(2) = %d, want 2", got)
	}

	root := t.TempDir()
	// GPU 0: procfs has explicit NUMA Node: 1.
	writeSyntheticFile(t, root, "proc/driver/nvidia/gpus/0000:17:00.0/information",
		"Model: NVIDIA H800\nIRQ: 120\nNUMA Node: 1\nBus Location: 0000:17:00.0\n")
	// GPU 1: procfs omits NUMA Node, sysfs has -1 (unset on single-socket/UMA).
	writeSyntheticFile(t, root, "proc/driver/nvidia/gpus/0000:b7:00.0/information",
		"Model: NVIDIA H800\nBus Location: 0000:b7:00.0\n")
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:b7:00.0/numa_node", "-1\n")
	// IB HCA mlx5_2 on NUMA node 1.
	writeSyntheticFile(t, root, "sys/class/infiniband/mlx5_2/device/numa_node", "1\n")

	n0, err := cudacluster.ReadProcNvidiaGPUNUMANode(root, "0000:17:00.0")
	if err != nil || n0 != 1 {
		t.Fatalf("ReadProcNvidiaGPUNUMANode(0000:17:00.0) = (%d, %v), want (1, nil)", n0, err)
	}

	n1, err := cudacluster.ReadProcNvidiaGPUNUMANode(root, "0000:b7:00.0")
	if err != nil || n1 != -1 {
		t.Fatalf("ReadProcNvidiaGPUNUMANode(0000:b7:00.0) fallback = (%d, %v), want (-1, nil)", n1, err)
	}
	if norm := cudacluster.NormalizeUnsetNUMA(n1); norm != 0 {
		t.Fatalf("NormalizeUnsetNUMA(n1) = %d, want 0", norm)
	}

	sysN1, err := cudacluster.ReadSysfsPCINUMANode(root, "0000:b7:00.0")
	if err != nil || sysN1 != -1 {
		t.Fatalf("ReadSysfsPCINUMANode(0000:b7:00.0) = (%d, %v), want (-1, nil)", sysN1, err)
	}

	ibNode, err := cudacluster.ReadSysfsIBNUMANode(root, "mlx5_2")
	if err != nil || ibNode != 1 {
		t.Fatalf("ReadSysfsIBNUMANode(mlx5_2) = (%d, %v), want (1, nil)", ibNode, err)
	}

	// Reject path traversal identifiers and corrupt negative values (< -1).
	if _, err := cudacluster.ReadProcNvidiaGPUNUMANode(root, "../etc"); err == nil {
		t.Fatal("ReadProcNvidiaGPUNUMANode accepted path traversal")
	}
	if _, err := cudacluster.ReadSysfsPCINUMANode(root, "../etc"); err == nil {
		t.Fatal("ReadSysfsPCINUMANode accepted path traversal")
	}
	if _, err := cudacluster.ReadSysfsIBNUMANode(root, "mlx5/../../etc"); err == nil {
		t.Fatal("ReadSysfsIBNUMANode accepted path traversal")
	}

	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:ff:00.0/numa_node", "-2\n")
	if _, err := cudacluster.ReadSysfsPCINUMANode(root, "0000:ff:00.0"); err == nil {
		t.Fatal("ReadSysfsPCINUMANode accepted -2 (< -1)")
	}
}

func TestDiscoverHostTopology(t *testing.T) {
	// Empty root must return ErrHardwarePending.
	emptyRoot := t.TempDir()
	if _, err := cudacluster.DiscoverHostTopology(emptyRoot); !errors.Is(err, cudacluster.ErrHardwarePending) {
		t.Fatalf("DiscoverHostTopology(empty) = %v, want wrapping ErrHardwarePending", err)
	}

	root := t.TempDir()
	// GPU 0 via procfs on NUMA 0 (written as -1 in sysfs to exercise normalization).
	writeSyntheticFile(t, root, "proc/driver/nvidia/gpus/0000:0f:00.0/information",
		"Model: NVIDIA B200\nNUMA Node: -1\n")
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:0f:00.0/numa_node", "-1\n")

	// GPU 1 via procfs on NUMA 1.
	writeSyntheticFile(t, root, "proc/driver/nvidia/gpus/0000:8a:00.0/information",
		"Model: NVIDIA B200\nNUMA Node: 1\n")

	// GPU 2 via sysfs-only (vendor 0x10de, class 0x030200 3D controller) on NUMA 3 (no IB on NUMA 3 -> fallback to mlx5_0).
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:c4:00.0/vendor", "0x10de\n")
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:c4:00.0/class", "0x030200\n")
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:c4:00.0/numa_node", "3\n")

	// Non-NVIDIA PCI device (Intel host bridge) must be ignored.
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:00:00.0/vendor", "0x8086\n")
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:00:00.0/class", "0x060000\n")
	writeSyntheticFile(t, root, "sys/bus/pci/devices/0000:00:00.0/numa_node", "0\n")

	// InfiniBand HCAs: mlx5_0 on NUMA 0, mlx5_1 on NUMA 1, mlx5_10 on NUMA 1.
	writeSyntheticFile(t, root, "sys/class/infiniband/mlx5_0/device/numa_node", "0\n")
	writeSyntheticFile(t, root, "sys/class/infiniband/mlx5_1/device/numa_node", "1\n")
	writeSyntheticFile(t, root, "sys/class/infiniband/mlx5_10/device/numa_node", "1\n")

	got, err := cudacluster.DiscoverHostTopology(root)
	if err != nil {
		t.Fatalf("DiscoverHostTopology: %v", err)
	}
	want := []cudacluster.DeviceTopology{
		{
			DeviceIndex:  0,
			PCIBusID:     "0000:0f:00.0",
			NUMANode:     0,
			NearestIBDev: "mlx5_0",
			IBNUMANode:   0,
		},
		{
			DeviceIndex:  1,
			PCIBusID:     "0000:8a:00.0",
			NUMANode:     1,
			NearestIBDev: "mlx5_1",
			IBNUMANode:   1,
		},
		{
			DeviceIndex:  2,
			PCIBusID:     "0000:c4:00.0",
			NUMANode:     3,
			NearestIBDev: "mlx5_0",
			IBNUMANode:   0,
		},
	}
	if len(got) != len(want) {
		t.Fatalf("len(DiscoverHostTopology) = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("topology[%d] = %+v, want %+v", i, got[i], want[i])
		}
		if i > 0 && got[i].PCIBusID <= got[i-1].PCIBusID {
			t.Fatalf("PCIBusID order invariant violated at %d: %q <= %q", i, got[i].PCIBusID, got[i-1].PCIBusID)
		}
		if got[i].NUMANode < 0 {
			t.Fatalf("negative normalized NUMANode at %d: %d", i, got[i].NUMANode)
		}
	}
}
