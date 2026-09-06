package fabricmanager_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/fabricmanager"
)

// Example connects to a running Fabric Manager instance on the local host and
// lists the fabric partitions it supports. It returns early when the library
// is absent or no instance is listening, which is every machine without an
// NVSwitch fabric.
//
// Every versioned struct carries its own size in the low 24 bits of Version
// and the struct revision in the high 8, which is what FM_*_VERSION expands
// to in the C header; the constants themselves are not exported, so the value
// is computed from the Go struct.
func Example() {
	if err := fabricmanager.LibInit(); err != nil {
		fmt.Println("Fabric Manager unavailable:", err)
		return
	}
	defer fabricmanager.LibShutdown()

	params := fabricmanager.ConnectParams{TimeoutMs: 1000}
	copy(params.AddressInfo[:], "127.0.0.1:6666")
	params.Version = uint32(unsafe.Sizeof(params)) | 1<<24

	h, err := fabricmanager.Connect(&params)
	if err != nil {
		fmt.Println("connect:", err)
		return
	}
	defer h.Disconnect()

	var list fabricmanager.FabricPartitionList
	list.Version = uint32(unsafe.Sizeof(list)) | 1<<24
	if err := h.GetSupportedFabricPartitions(&list); err != nil {
		fmt.Println("partitions:", err)
		return
	}
	for _, p := range list.PartitionInfo[:list.NumPartitions] {
		fmt.Printf("partition %d: %d GPUs\n", p.PartitionId, p.NumGpus)
	}
}
