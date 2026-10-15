package cudacache_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"os"
)

func ExampleCompilerIdentity_jitOptions() {
	dir, err := os.MkdirTemp("", "jit-example-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	c, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		panic(err)
	}
	jit := cudakernel.JITOptions{OptLevelSet: true}
	id := cudacache.CompilerIdentity{TargetArch: "compute_80", OutputKind: nvrtcx.OutputPTX, JITOptions: jit}
	// Hand-authored bytes demonstrate storage, not compilation or execution.
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("descriptor fixture"), "k", cudakernel.Schema{}, id.TargetArch, cudakernel.WithJIT(jit))
	if err != nil {
		panic(err)
	}
	a := cudacache.NewNVRTCAdapter(c)
	err = a.Store(context.Background(), "fixture-source", id, art)
	fmt.Println(err)
	got, hit, err := a.Lookup(context.Background(), "fixture-source", id)
	fmt.Println(hit, got.JITOptions.OptLevelSet, err)
	// Output:
	// <nil>
	// true true <nil>
}
