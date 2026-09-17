package cudacache_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func ExampleCompilerCacheKey() {
	id := cudacache.CompilerIdentity{
		NVRTCVersion: "12.8",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_121a",
		Flags:        []string{"--std=c++20", "--extra-device-vectorization"},
		OutputKind:   nvrtcx.OutputCUBIN,
	}

	key := cudacache.CompilerCacheKey("kernel_source_hash_1234", id)
	fmt.Println(len(key))
	// Output:
	// 64
}

func ExampleCache() {
	tmpDir := "/tmp/cudacache-example"
	cache, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	schema, _ := cudakernel.NewSchema(cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true})
	art, err := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		[]byte(".version 8.0\n.target sm_121a\n.address_size 64\n.entry k(.param .u64 p) { ret; }\n"),
		"k",
		schema,
		"sm_121a",
	)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	ctx := context.Background()
	key := "example-key"
	if err := cache.PutArtifact(ctx, key, art); err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	retrieved, err := cache.GetArtifact(ctx, key)
	if err != nil {
		fmt.Printf("err: %v\n", err)
		return
	}

	fmt.Println(retrieved.EntryPoint)
	// Output:
	// k
}
