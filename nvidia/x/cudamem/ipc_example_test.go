package cudamem_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func ExampleRoundUpPageAligned() {
	for _, req := range []int64{1, 4096, 4097, 65535} {
		aligned, err := cudamem.RoundUpPageAligned(req)
		if err != nil {
			fmt.Println("error:", err)
			return
		}
		fmt.Printf("%d -> %d\n", req, aligned)
	}

	var h cudamem.IPCHandle
	h[0] = 0xde
	h[63] = 0xad
	parsed, err := cudamem.ParseIPCHandleHex(h.Hex())
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("ipc zero=%v match=%v\n", parsed.IsZero(), parsed == h)

	// Output:
	// 1 -> 4096
	// 4096 -> 4096
	// 4097 -> 8192
	// 65535 -> 65536
	// ipc zero=false match=true
}
