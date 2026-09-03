// SPDX-License-Identifier: Apache-2.0

package cudaquant_test

import (
	"encoding/binary"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaquant"
)

func Example() {
	data := make([]byte, 34)
	binary.LittleEndian.PutUint16(data, 0x3c00) // scale 1
	data[2] = 42
	values, err := cudaquant.DequantizeQ8_0(data, 32)
	if err != nil {
		panic(err)
	}
	fmt.Printf("values=%d first=%.0f\n", len(values), values[0])
	// Output: values=32 first=42
}
