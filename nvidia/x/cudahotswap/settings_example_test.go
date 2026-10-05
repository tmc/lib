package cudahotswap

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// These host examples substitute a fake loader and lease. A CUDA consumer
// uses NewNVRTCLoader and NewService's default cudalock request leases.
func settingsExampleService() *Service {
	s, err := NewService(&serviceLoader{kernel: &serviceKernel{raw: 1}})
	if err != nil {
		panic(err)
	}
	s.acquire = func(context.Context, string) (io.Closer, error) {
		return io.NopCloser(strings.NewReader("")), nil
	}
	_, err = s.Register(context.Background(), "kernel", Source{Code: "source", Entry: "entry"}, "original")
	if err != nil {
		panic(err)
	}
	return s
}

func ExampleLaunchGeometry() {
	g := LaunchGeometry{Grid: cudashape.Grid3D{X: 2, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1)}
	fmt.Println(g.Grid.X, g.Block.X, g.SharedMem)
	// Output: 2 32 0
}

func ExampleService_SetCompileFlags() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	state, err := s.SetCompileFlags("kernel", []string{"--fmad=false"}, "precise")
	fmt.Println(state.Generation, state.ConfigRevision, err)
	// Output: 1 1 <nil>
}

func ExampleService_SetGeometry() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	g := LaunchGeometry{Grid: cudashape.Grid3D{X: 2, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1)}
	state, err := s.SetGeometry("kernel", &g, "two blocks")
	fmt.Println(state.Generation, state.ConfigRevision, err)
	// Output: 1 1 <nil>
}

func ExampleService_Status() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	state, err := s.Status("kernel")
	fmt.Println(state.Name, state.Kind, state.Generation, err)
	// Output: kernel kernel 1 <nil>
}
