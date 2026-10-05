package cudahotswap

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func graphExampleService() (*Service, *cudagraph.Exec) {
	s := settingsExampleService() // Host fake loader, lease and graph operations.
	s.graphs.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error { return nil }
	s.graphs.ops.launch = func(*cudagraph.Exec, *cuda.Stream) error { return nil }
	s.graphs.ops.wait = func(context.Context, *cuda.Stream) error { return nil }
	exec := new(cudagraph.Exec)
	_, err := s.RegisterGraph(context.Background(), "kernel", exec, new(cudagraph.Node), new(cuda.Stream), driver.Dims(1, 1, 1))
	if err != nil {
		panic(err)
	}
	return s, exec
}

func ExampleService_RegisterGraph() {
	s, _ := graphExampleService()
	defer s.Close(context.Background())
	fmt.Println(s.graphs.bindings[0].name)
	// Output: kernel
}

func ExampleService_LaunchGraph() {
	s, exec := graphExampleService()
	defer s.Close(context.Background())
	p, err := s.LaunchGraph(context.Background(), exec)
	if p != nil {
		err = p.Wait(context.Background())
	}
	fmt.Println(err)
	// Output: <nil>
}

func ExampleGraphLaunch() {
	var p GraphLaunch
	fmt.Println(p.Wait(context.Background()))
	// Output: <nil>
}

func ExampleGraphLaunch_Wait() {
	var p GraphLaunch
	fmt.Println(p.Wait(context.Background()))
	// Output: <nil>
}
