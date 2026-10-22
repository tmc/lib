package cudacluster_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudacluster"
)

func ExampleDiscoverHostTopology() {
	root, err := os.MkdirTemp("", "cudacluster-topo-*")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	defer os.RemoveAll(root)

	mustWrite := func(rel, data string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			panic(err)
		}
		if err := os.WriteFile(p, []byte(data), 0o644); err != nil {
			panic(err)
		}
	}

	mustWrite("proc/driver/nvidia/gpus/0000:17:00.0/information", "Model: H800\nNUMA Node: 0\n")
	mustWrite("proc/driver/nvidia/gpus/0000:b7:00.0/information", "Model: H800\nNUMA Node: 1\n")
	mustWrite("sys/class/infiniband/mlx5_0/device/numa_node", "0\n")
	mustWrite("sys/class/infiniband/mlx5_1/device/numa_node", "1\n")

	topo, err := cudacluster.DiscoverHostTopology(root)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	for _, d := range topo {
		fmt.Printf("gpu=%d pci=%s numa=%d ib=%s ib_numa=%d\n",
			d.DeviceIndex, d.PCIBusID, d.NUMANode, d.NearestIBDev, d.IBNUMANode)
	}

	// Output:
	// gpu=0 pci=0000:17:00.0 numa=0 ib=mlx5_0 ib_numa=0
	// gpu=1 pci=0000:b7:00.0 numa=1 ib=mlx5_1 ib_numa=1
}
