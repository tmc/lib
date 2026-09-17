package cudakernel_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func ExampleNewSchema() {
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "in", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "alpha", Type: "f32"},
		cudakernel.Param{Name: "n", Type: "u32"},
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("params: %d, total bytes: %d\n", len(schema.Params), schema.TotalParamBytes)
	for _, p := range schema.Params {
		fmt.Printf("%s: offset %d\n", p.Name, p.Offset)
	}
	// Output:
	// params: 4, total bytes: 24
	// out: offset 0
	// in: offset 8
	// alpha: offset 16
	// n: offset 20
}

func ExampleSchema_Pack() {
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "ptr", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "scale", Type: "f32"},
		cudakernel.Param{Name: "count", Type: "u32"},
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Pack valid device pointer and scalar parameters.
	buf, err := schema.Pack(
		driver.DevicePtr(0x1000), // 8-byte aligned device pointer
		float32(2.5),
		uint32(1024),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("packed buffer length: %d\n", len(buf))
	// Output:
	// packed buffer length: 16
}

func ExampleNewArtifact() {
	schema, _ := cudakernel.NewSchema(
		cudakernel.Param{Name: "a", Type: "u32"},
	)
	ptx := []byte(".version 8.0\n.target sm_121a\n.entry test_k(.param .u32 a) { ret; }\n")
	art, err := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		ptx,
		"test_k",
		schema,
		"sm_121a",
		cudakernel.WithJIT(cudakernel.JITOptions{
			MaxRegisters:    32,
			ThreadsPerBlock: 128,
			OptLevel:        4,
		}),
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("kind:", art.Kind)
	fmt.Println("entry:", art.EntryPoint)
	fmt.Println("target:", art.Target)
	fmt.Println("requires link:", art.RequiresLink())
	// Output:
	// kind: PTX
	// entry: test_k
	// target: sm_121a
	// requires link: false
}
