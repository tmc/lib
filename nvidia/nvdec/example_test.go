package nvdec_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvdec"
)

// Example asks the hardware what it can decode before anything is created.
// It returns early when the NVDEC library is not available.
//
// GetDecoderCaps needs a current CUDA context, so a real program calls it
// after the driver package's Init and NewContext. The capability query is the
// step that decides whether a clip can be decoded at all: BIsSupported is
// per codec, chroma format and bit depth, and NMaxWidth/NMaxHeight bound the
// dimensions CreateDecoder will accept.
func Example() {
	caps := nvdec.DecodeCaps{
		ECodecType:      nvdec.CodecH264,
		EChromaFormat:   nvdec.ChromaFormat420,
		NBitDepthMinus8: 0,
	}
	res, err := nvdec.GetDecoderCaps(&caps)
	if err != nil {
		fmt.Println("NVDEC unavailable:", err)
		return
	}
	if res != nvdec.Success {
		fmt.Println("caps:", res)
		return
	}
	if caps.BIsSupported == 0 {
		fmt.Println("H.264 4:2:0 8-bit is not supported here")
		return
	}
	fmt.Printf("H.264 4:2:0 8-bit up to %dx%d on %d NVDEC engines\n",
		caps.NMaxWidth, caps.NMaxHeight, caps.NNumNVDECs)
}
