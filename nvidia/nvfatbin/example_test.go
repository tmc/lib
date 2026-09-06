package nvfatbin_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvfatbin"
)

// Example packages PTX for two architectures into one fatbinary. It returns
// early when the nvFatbin library is not available. The bytes Get returns are
// what the driver package's ModuleLoadFatBinary expects; the driver picks the
// image matching the device it loads on.
func Example() {
	if _, _, err := nvfatbin.Version(); err != nil {
		fmt.Println("nvFatbin unavailable:", err)
		return
	}

	const ptx = `
.version 7.0
.target sm_70
.address_size 64
.visible .entry noop() { ret; }
`
	h, err := nvfatbin.Create([]string{"-cuda"})
	if err != nil {
		fmt.Println("create:", err)
		return
	}
	defer nvfatbin.Destroy(&h)

	for _, arch := range []string{"compute_70", "compute_80"} {
		if err := h.AddPTXString(ptx, arch, "noop.ptx", ""); err != nil {
			fmt.Println("add", arch+":", err)
			return
		}
	}
	fatbin, err := h.Get()
	if err != nil {
		fmt.Println("get:", err)
		return
	}
	fmt.Println("fatbinary of", len(fatbin), "bytes")
}
