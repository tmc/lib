// SPDX-License-Identifier: Apache-2.0

package mxfp4_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/mxfp4"
)

func ExamplePackOCPBlocks() {
	src := make([]float32, 32)
	src[0], src[1] = 6.0, -3.0
	packed, err := mxfp4.PackOCPBlocks(src)
	if err != nil {
		panic(err)
	}
	unpacked, err := mxfp4.UnpackOCPBlocks(packed, 32)
	if err != nil {
		panic(err)
	}
	fmt.Printf("bytes=%d scaleExp=%d firstTwo=[%.1f %.1f]\n",
		len(packed), packed[16], unpacked[0], unpacked[1])
	// Output:
	// bytes=17 scaleExp=127 firstTwo=[6.0 -3.0]
}
