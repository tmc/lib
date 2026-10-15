package cudacheck_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudacheck"
	"github.com/tmc/lib/nvidia/x/cudair"
	"os"
)

func ExampleWithArtifactCache() {
	dir, err := os.MkdirTemp("", "trace-cache-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	cache, err := cudacache.New(cudacache.Config{RootDir: dir})
	if err != nil {
		panic(err)
	}
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{1}, cudair.Float32)
	y := g.AddUnary("neg", x)
	y.Name = "Y"
	if err := g.SetOutputs(y); err != nil {
		panic(err)
	}
	// Configuration example only: no device or compiler is loaded here.
	result, err := cudacheck.Compare(context.Background(), g, cudacheck.WithInputs(map[string][]float32{"X": {1}}), cudacheck.WithArtifactCache(cudacache.NewNVRTCAdapter(cache), "compute_80"))
	fmt.Println(result.Status == cudacheck.StatusUnmeasured, err)
	// Output: true <nil>
}
