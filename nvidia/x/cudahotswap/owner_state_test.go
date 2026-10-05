package cudahotswap

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestOwnerStateRetainsConfiguration(t *testing.T) {
	s, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	t.Cleanup(func() { s.Close(context.Background()) })
	registerService(t, s)
	want, err := s.SetCompileFlags("kernel", []string{"--fmad=false"}, "first")
	if err != nil {
		t.Fatal(err)
	}
	pin, err := s.Pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	launch, err := s.Launch(context.Background(), "kernel", cudalaunch.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCompileFlags("kernel", nil, "second"); err != nil {
		t.Fatal(err)
	}
	if pin.State() != want || launch.State() != want {
		t.Fatal("owner snapshot followed newer configuration")
	}
	if err := launch.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := pin.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if pin.State() != want || launch.State() != want {
		t.Fatal("completed owner lost its snapshot")
	}
}

func TestGraphOwnerStatesDetached(t *testing.T) {
	s, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	t.Cleanup(func() { s.Close(context.Background()) })
	registerService(t, s)
	want, err := s.Status("kernel")
	if err != nil {
		t.Fatal(err)
	}
	exec, node, stream := new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream)
	s.graphs.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error { return nil }
	s.graphs.ops.launch = func(*cudagraph.Exec, *cuda.Stream) error { return nil }
	s.graphs.ops.wait = func(context.Context, *cuda.Stream) error { return nil }
	if _, err := s.RegisterGraph(context.Background(), "kernel", exec, node, stream, driver.Dims(1, 1, 1)); err != nil {
		t.Fatal(err)
	}
	launch, err := s.LaunchGraph(context.Background(), exec)
	if err != nil {
		t.Fatal(err)
	}
	got := launch.States()
	if !reflect.DeepEqual(got, []State{want}) {
		t.Fatalf("states = %+v, want %+v", got, want)
	}
	got[0].Label = "edited detached result"
	if err := launch.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(launch.States(), []State{want}) {
		t.Fatal("returned slice or completion changed retained snapshot")
	}
}

func TestNilOwnerStates(t *testing.T) {
	if (*Pin)(nil).State() != (State{}) || (*PendingLaunch)(nil).State() != (State{}) || (*GraphLaunch)(nil).States() != nil {
		t.Fatal("nil owner supplied a state")
	}
}

func ExamplePin_State() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	p, err := s.Pin("kernel")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer p.Release(context.Background())
	state := p.State()
	fmt.Println(state.Name, state.Generation)
	// Output: kernel 1
}

func ExamplePendingLaunch_State() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	p, err := s.Launch(context.Background(), "kernel", cudalaunch.Config{})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer p.Wait(context.Background())
	state := p.State()
	fmt.Println(state.Name, state.Generation)
	// Output: kernel 1
}

func ExampleGraphLaunch_States() {
	s, exec := graphExampleService()
	defer s.Close(context.Background())
	p, err := s.LaunchGraph(context.Background(), exec)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer p.Wait(context.Background())
	states := p.States()
	fmt.Println(len(states), states[0].Name, states[0].Generation)
	// Output: 1 kernel 1
}
