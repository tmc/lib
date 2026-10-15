// SPDX-License-Identifier: Apache-2.0

package cudakv_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExampleBlockAllocator() {
	allocator := cudakv.NewBlockAllocator(2, 2)
	if err := allocator.AllocSequence(7, nil, 3); err != nil {
		fmt.Println(err)
		return
	}
	if err := allocator.AppendToken(7); err != nil {
		fmt.Println(err)
		return
	}
	length, _ := allocator.SequenceLength(7)
	fmt.Println(allocator.Table(7), length, allocator.NumFreeBlocks())
	// Output: [1 0] 4 0
}
