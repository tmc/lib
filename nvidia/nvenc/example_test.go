package nvenc_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvenc"
)

// Example fills in the function table and lists the codecs the encoder
// supports. It returns early when the NVENC library is not available.
//
// This is the shape of every NVENC program: the shared library exports only
// two symbols, so the driver writes the rest of the entry points into a
// [FunctionList] whose Version says which API revision the caller was built
// against. [Bind] turns that table into callable Go functions. Opening a
// session then takes a current CUDA context from the driver package.
func Example() {
	var max uint32
	if _, err := nvenc.GetMaxSupportedVersion(&max); err != nil {
		fmt.Println("NVENC unavailable:", err)
		return
	}

	list := nvenc.FunctionList{Version: nvenc.FunctionListVer}
	if status, err := nvenc.CreateInstance(&list); err != nil {
		fmt.Println("create instance:", err)
		return
	} else if status != nvenc.Success {
		fmt.Println("create instance:", nvenc.Err(status))
		return
	}
	api, err := nvenc.Bind(&list)
	if err != nil {
		fmt.Println("bind:", err)
		return
	}

	// session is the encoder session opened against a CUDA context; see
	// NvEncOpenEncodeSession and NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS.
	var session unsafe.Pointer

	var count uint32
	if status := api.NvEncGetEncodeGUIDCount(session, &count); status != nvenc.Success {
		fmt.Println("guid count:", nvenc.Err(status))
		return
	}
	guids := make([]nvenc.GUID, count)
	if status := api.NvEncGetEncodeGUIDs(session, &guids[0], count, &count); status != nvenc.Success {
		fmt.Println("guids:", nvenc.Err(status))
		return
	}
	fmt.Printf("API %#x, %d encode GUIDs\n", max, count)
}
