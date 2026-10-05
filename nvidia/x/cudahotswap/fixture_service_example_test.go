package cudahotswap

import (
	"context"
	"fmt"
	"io"
)

func ExampleDefaultReplayPolicy() {
	p := DefaultReplayPolicy()
	fmt.Println(p.Rounds, p.Exact)
	// Output: 5 true
}

func ExampleReplayPolicy() {
	p := ReplayPolicy{Rounds: 5, MaxAbs: 1e-5, RMS: 1e-6, MaxULP: 4}
	fmt.Println(p.Rounds, p.MaxULP)
	// Output: 5 4
}

func ExampleFixtureBuffer() {
	b := FixtureBuffer{Arg: 2, Bytes: 8, Stable: true}
	fmt.Println(b.Arg, b.Bytes, b.Stable)
	// Output: 2 8 true
}

func ExampleFixtureOutput() {
	o := FixtureOutput{Arg: 1, Elem: "f32"}
	fmt.Println(o.Arg, o.Elem)
	// Output: 1 f32
}

func ExampleFixtureConfig() {
	c := FixtureConfig{Policy: DefaultReplayPolicy(), Outputs: []FixtureOutput{{Arg: 1, Elem: "f32"}}}
	fmt.Println(c.Policy.Rounds, c.Outputs[0].Elem)
	// Output: 5 f32
}

func fixtureExampleService() (*Service, FixtureConfig) {
	// Fake loader, request lease and replay clock keep this example host-only.
	k := &serviceFixtureKernel{fixtureTestKernel: fixtureKernel("baseline", false)}
	l := &serviceLoader{kernel: k, compile: func(context.Context, Source) (*Compilation, error) {
		return &Compilation{Entry: "entry", PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry(.param .b64 x, .param .b64 y, .param .b64 w, .param .u32 n) { ret; }"}, nil
	}}
	s, err := NewService(l)
	if err != nil {
		panic(err)
	}
	s.acquire = func(context.Context, string) (io.Closer, error) { return new(serviceCloser), nil }
	s.newReplay = func(string, replaySpec, replayPolicy, []replayDeviceBinding) (replayBackend, error) {
		return &serviceFixtureBackend{fixtureTestBackend: fixtureBackend(), service: s}, nil
	}
	if _, err := s.Register(context.Background(), "kernel", Source{Code: "source", Entry: "entry"}, "original"); err != nil {
		panic(err)
	}
	p := fixtureSpec()
	c := FixtureConfig{Args: p.Args, Launch: p.Config, Policy: DefaultReplayPolicy(), Outputs: []FixtureOutput{{Arg: 1, Elem: "f32"}}}
	for _, b := range p.Buffers {
		c.Buffers = append(c.Buffers, FixtureBuffer{Arg: b.Arg, Bytes: b.Bytes, Stable: b.Stable})
	}
	return s, c
}

func ExampleService_RecordFixture() {
	s, config := fixtureExampleService()
	defer s.Close(context.Background())
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if f != nil {
		defer f.Close(context.Background())
	}
	fmt.Println(f.State().Name, err)
	// Output: kernel <nil>
}

func ExampleFixture() {
	var f Fixture
	fmt.Println(f.Close(context.Background()))
	// Output: <nil>
}

func ExampleFixture_State() {
	var f Fixture
	fmt.Printf("%q\n", f.State().Name)
	// Output: ""
}

func ExampleFixture_Close() {
	var f Fixture
	fmt.Println(f.Close(context.Background()))
	// Output: <nil>
}
