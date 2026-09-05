package nvenc_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/nvenc"
)

func ExampleGUID() {
	params := nvenc.InitializeParams{
		Version:    nvenc.InitializeParamsVer,
		EncodeGUID: nvenc.CodecH264GUID,
		PresetGUID: nvenc.PresetP1GUID,
	}
	fmt.Println(params.EncodeGUID == nvenc.CodecH264GUID)
	// Output: true
}
