package openxla_test

import (
	"fmt"
	"github.com/tmc/lib/openxla"
)

func ExampleBufferWrapper_IsOnCPU() {
	var buffer openxla.BufferWrapper
	_, err := buffer.IsOnCPU()
	fmt.Println(err)
	// Output: openxla: buffer is closed
}

func ExampleBufferWrapper_WaitReady() {
	var buffer openxla.BufferWrapper
	fmt.Println(buffer.WaitReady())
	// Output: openxla: buffer is closed
}

func ExampleClientWrapper_CreateViewOfDeviceBuffer() {
	// A live client and caller-owned device allocation are required.
	var client openxla.ClientWrapper
	_, err := client.CreateViewOfDeviceBuffer(nil, openxla.Buffer_Type_U8, nil, nil, nil, 0, nil)
	fmt.Println(err)
	// Output: openxla: client is closed
}
