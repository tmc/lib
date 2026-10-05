//go:build linux || darwin

package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
)

// The host-only example has no CUDA compiler or device owner.
type stageExampleLoader struct{}

func (stageExampleLoader) Compile(context.Context, Source) (*Compilation, error) {
	return nil, errors.New("example has no CUDA compiler")
}

func (stageExampleLoader) Load(context.Context, *Compilation) (Kernel, error) {
	return nil, errors.New("example has no CUDA loader")
}

func ExampleStageSpec() {
	spec := StageSpec{Buffers: []StageBuffer{{Name: "output", Bytes: 1024}}, Functions: []StageFunction{{Name: "entry", Kernel: "writer"}}}
	fmt.Println(spec.Buffers[0].Name, spec.Functions[0].Kernel)
	// Output: output writer
}

func ExampleStageBuffer() {
	buffer := StageBuffer{Name: "output", Bytes: 1024}
	fmt.Println(buffer.Name, buffer.Bytes)
	// Output: output 1024
}

func ExampleStageFunction() {
	function := StageFunction{Name: "entry", Kernel: "writer"}
	fmt.Println(function.Name, function.Kernel)
	// Output: entry writer
}

func ExampleStageLaunch() {
	// Host-only stages have no device bindings. Native stages require every
	// declared buffer and a stream belonging to the actual owning device.
	launch := StageLaunch{}
	fmt.Println(len(launch.Buffers), launch.Stream == nil)
	// Output: 0 true
}

func exampleStageService() (*Service, *stage.Builder, func(), error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return nil, nil, nil, err
	}
	parent := filepath.Join(cache, "cudahotswap-stage-examples")
	if err := os.MkdirAll(parent, 0700); err != nil {
		return nil, nil, nil, err
	}
	dir, err := os.MkdirTemp(parent, "run-")
	if err != nil {
		return nil, nil, nil, err
	}
	builder, err := stage.NewBuilder(dir, "cc")
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, nil, nil, err
	}
	s, err := NewService(stageExampleLoader{})
	if err != nil {
		_ = os.RemoveAll(dir)
		return nil, nil, nil, err
	}
	cleanup := func() {
		if s.Close(context.Background()) == nil {
			_ = os.RemoveAll(dir)
		}
	}
	return s, builder, cleanup, nil
}

func ExampleService_RegisterStage() {
	s, builder, cleanup, err := exampleStageService()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cleanup()
	state, err := s.RegisterStage(context.Background(), "prepare", builder, stage.Source{Code: `int stage(struct stage_ctx *ctx) { return ctx->version == 1 ? 0 : 1; }`}, StageSpec{HostOnly: true}, "original")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(state.Name, state.Kind)
	// Output: prepare stage
}

func ExampleService_RunStage() {
	s, builder, cleanup, err := exampleStageService()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cleanup()
	ctx := context.Background()
	_, err = s.RegisterStage(ctx, "prepare", builder, stage.Source{Code: `int stage(struct stage_ctx *ctx) { return 0; }`}, StageSpec{HostOnly: true}, "original")
	if err != nil {
		fmt.Println(err)
		return
	}
	pending, err := s.RunStage(ctx, "prepare", StageLaunch{})
	if pending != nil {
		err = errors.Join(err, pending.Wait(ctx))
	}
	fmt.Println(err == nil)
	// Output: true
}

func exampleStageChecker(s *Service) Checker {
	return func(ctx context.Context, state State) error {
		pending, err := s.RunStage(ctx, state.Name, StageLaunch{})
		if pending != nil {
			err = errors.Join(err, pending.Wait(ctx))
		}
		return err
	}
}

func ExampleService_PushStage() {
	s, builder, cleanup, err := exampleStageService()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cleanup()
	ctx := context.Background()
	source := stage.Source{Code: `int stage(struct stage_ctx *ctx) { return 0; }`}
	spec := StageSpec{HostOnly: true}
	if _, err = s.RegisterStage(ctx, "prepare", builder, source, spec, "original"); err != nil {
		fmt.Println(err)
		return
	}
	if _, err = s.SetCheck("prepare", CheckConfig{Checker: exampleStageChecker(s)}, "application check"); err != nil {
		fmt.Println(err)
		return
	}
	result, err := s.PushStage(ctx, "prepare", source, spec, "variant")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Published)
	// Output: true
}

func ExampleService_RevertStage() {
	s, builder, cleanup, err := exampleStageService()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cleanup()
	ctx := context.Background()
	source := stage.Source{Code: `int stage(struct stage_ctx *ctx) { return 0; }`}
	spec := StageSpec{HostOnly: true}
	original, err := s.RegisterStage(ctx, "prepare", builder, source, spec, "original")
	if err != nil {
		fmt.Println(err)
		return
	}
	if _, err = s.SetCheck("prepare", CheckConfig{Checker: exampleStageChecker(s)}, "application check"); err != nil {
		fmt.Println(err)
		return
	}
	if _, err = s.PushStage(ctx, "prepare", source, spec, "variant"); err != nil {
		fmt.Println(err)
		return
	}
	selected, err := s.RevertStage(ctx, "prepare", "restore original")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(selected.Version == original.Version)
	// Output: true
}

func ExampleService_RecordStageFixture() {
	s, builder, cleanup, err := exampleStageService()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer cleanup()
	ctx := context.Background()
	if _, err = s.RegisterStage(ctx, "prepare", builder, stage.Source{Code: `int stage(struct stage_ctx *ctx) { return 0; }`}, StageSpec{HostOnly: true}, "original"); err != nil {
		fmt.Println(err)
		return
	}
	// Replay requires an actual device and complete declared device buffers.
	// A host-only stage uses an application Checker instead.
	_, err = s.RecordStageFixture(ctx, "prepare", FixtureConfig{})
	fmt.Println(err != nil)
	// Output: true
}

func ExamplePendingStage_Wait() {
	var owner PendingStage
	fmt.Println(owner.Wait(context.Background()) != nil)
	// Output: true
}
