package httpapi

import (
	"context"
	"errors"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/stage"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

// StageService is the optional host-stage extension of Service. An application
// without this extension refuses stage calls; existing kernel calls still work.
type StageService interface {
	RegisterStage(context.Context, string, *stage.Builder, stage.Source, cudahotswap.StageSpec, string) (cudahotswap.State, error)
	PushStage(context.Context, string, stage.Source, cudahotswap.StageSpec, string) (*cudahotswap.PushResult, error)
	RevertStage(context.Context, string, string) (cudahotswap.State, error)
	RunStage(context.Context, string, cudahotswap.StageLaunch) (*cudahotswap.PendingStage, error)
	RecordStageFixture(context.Context, string, cudahotswap.FixtureConfig) (*cudahotswap.Fixture, error)
}

// StageBinding fixes one application-owned stage's compilation and context.
// Builder, its generation directories and all named owners must remain live
// until Handler.Close succeeds. BufferNames is in Schema.Buffers order. Stream
// names an application stream. HostOnly requires no stream or device bindings.
// CUDA and Flags are fixed by the application, never by the HTTP caller.
type StageBinding struct {
	Builder     *stage.Builder
	Schema      cudahotswap.StageSpec
	BufferNames []string
	Stream      string
	CUDA        bool
	Flags       []string
}

func copyStageBindings(bindings map[string]StageBinding, resources Resources) (map[string]StageBinding, error) {
	if len(bindings) > 128 {
		return nil, errors.New("httpapi: stage binding limit exceeded")
	}
	copy := make(map[string]StageBinding, len(bindings))
	for name, b := range bindings {
		if !validName(name) || b.Builder == nil || len(b.Flags) > 64 {
			return nil, errors.New("httpapi: invalid stage builder binding")
		}
		flagBytes := 0
		for _, flag := range b.Flags {
			flagBytes += len(flag)
			if flag == "" || strings.ContainsRune(flag, 0) || flagBytes > 16<<10 {
				return nil, errors.New("httpapi: invalid stage compiler flags")
			}
		}
		if b.Schema.HostOnly {
			if b.CUDA || len(b.Schema.Buffers) != 0 || len(b.Schema.Functions) != 0 || len(b.BufferNames) != 0 || b.Stream != "" {
				return nil, errors.New("httpapi: host-only stage has device bindings")
			}
		} else {
			if len(b.Schema.Buffers) == 0 || len(b.Schema.Buffers) > 64 || len(b.Schema.Functions) == 0 || len(b.Schema.Functions) > 64 || len(b.BufferNames) != len(b.Schema.Buffers) || resources.Streams[b.Stream] == nil {
				return nil, errors.New("httpapi: incomplete stage context binding")
			}
			seen := make(map[string]bool)
			total := 0
			for i, spec := range b.Schema.Buffers {
				buffer, ok := resources.Buffers[b.BufferNames[i]]
				if !validName(spec.Name) || seen[spec.Name] || !ok || spec.Bytes <= 0 || spec.Bytes > (64<<20)-total || buffer.Len() != spec.Bytes {
					return nil, errors.New("httpapi: invalid stage buffer extent or name")
				}
				seen[spec.Name] = true
				total += spec.Bytes
			}
			for _, f := range b.Schema.Functions {
				if !validName(f.Name) || seen[f.Name] || !validName(f.Kernel) {
					return nil, errors.New("httpapi: invalid stage function binding")
				}
				seen[f.Name] = true
			}
		}
		b.Schema.Buffers = append([]cudahotswap.StageBuffer(nil), b.Schema.Buffers...)
		b.Schema.Functions = append([]cudahotswap.StageFunction(nil), b.Schema.Functions...)
		b.BufferNames = append([]string(nil), b.BufferNames...)
		b.Flags = append([]string(nil), b.Flags...)
		copy[name] = b
	}
	return copy, nil
}

func isStageMethod(method string) bool {
	switch method {
	case "register_stage", "push_stage", "revert_stage", "run_stage", "record_stage_fixture":
		return true
	}
	return false
}

func (h *Handler) callStage(ctx context.Context, method string, q Request) (response Response, err error) {
	if !validName(q.Name) || !validName(q.Label) {
		return response, bad("stage name and caller label required")
	}
	service, ok := h.service.(StageService)
	if !ok {
		return response, bad("application does not support stages")
	}
	binding, ok := h.resources.Stages[q.Name]
	if !ok {
		return response, bad("unknown application stage binding")
	}
	if q.Geometry != nil || len(q.Flags) != 0 || q.Source.Code != "" || q.Source.Filename != "" || q.Source.Entry != "" || len(q.Source.Flags) != 0 {
		return response, bad("stage compilation and geometry are application-owned")
	}
	state := func(s cudahotswap.State, e error) (Response, error) { response.State = &s; return response, e }
	source := stage.Source{Code: q.StageCode, CUDA: binding.CUDA, Flags: append([]string(nil), binding.Flags...)}
	schema := binding.Schema
	// Each call receives its own slices, not the handler's retained schema.
	schema.Buffers = append([]cudahotswap.StageBuffer(nil), schema.Buffers...)
	schema.Functions = append([]cudahotswap.StageFunction(nil), schema.Functions...)
	switch method {
	case "register_stage", "push_stage":
		if source.Code == "" || len(source.Code) > 1<<20 || len(q.Args) != 0 || len(q.Buffers) != 0 || q.Stream != "" {
			return response, bad("stage source required without client launch bindings")
		}
		if method == "register_stage" {
			return state(service.RegisterStage(ctx, q.Name, binding.Builder, source, schema, q.Label))
		}
		response.Push, err = service.PushStage(ctx, q.Name, source, schema, q.Label)
		return response, err
	case "revert_stage":
		if q.StageCode != "" || len(q.Args) != 0 || len(q.Buffers) != 0 || q.Stream != "" {
			return response, bad("stage revert takes no source or launch bindings")
		}
		return state(service.RevertStage(ctx, q.Name, q.Label))
	}
	if q.StageCode != "" || q.Stream != binding.Stream || len(q.Args) != len(binding.BufferNames) {
		return response, bad("stage context differs from application binding")
	}
	launch := cudahotswap.StageLaunch{Stream: h.resources.Streams[binding.Stream]}
	for i, name := range binding.BufferNames {
		arg := q.Args[i]
		if arg.Type != "buffer" || arg.Name != name || len(arg.Value) != 0 {
			return response, bad("stage argument differs from named buffer order")
		}
		launch.Buffers = append(launch.Buffers, h.resources.Buffers[name])
	}
	var config cudahotswap.FixtureConfig
	if method == "record_stage_fixture" {
		if schema.HostOnly || len(q.Buffers) != len(schema.Buffers) {
			return response, bad("native stage fixture requires complete named buffers")
		}
		config = cudahotswap.FixtureConfig{Args: make([]any, len(schema.Buffers)), Launch: cudalaunch.Config{Stream: launch.Stream}, Outputs: q.Outputs, Policy: cudahotswap.DefaultReplayPolicy()}
		if q.Policy != nil {
			config.Policy = *q.Policy
		}
		for i, b := range q.Buffers {
			if b.Arg != i || b.Name != binding.BufferNames[i] || b.Bytes != schema.Buffers[i].Bytes {
				return response, bad("stage fixture differs from declared buffer order or extent")
			}
			config.Buffers = append(config.Buffers, cudahotswap.FixtureBuffer{Arg: i, Bytes: b.Bytes, Stable: b.Stable, Buffer: launch.Buffers[i]})
		}
	} else if len(q.Buffers) != 0 || len(q.Outputs) != 0 || q.Policy != nil {
		return response, bad("stage run takes no fixture settings")
	}
	id, err := h.reserve()
	if err != nil {
		return response, err
	}
	owner := retained{}
	if method == "run_stage" {
		p, e := service.RunStage(ctx, q.Name, launch)
		err = e
		if p != nil {
			owner.wait = p.Wait
			response.Handle = id
		}
	} else {
		p, e := service.RecordStageFixture(ctx, q.Name, config)
		err = e
		if p != nil {
			owner.fixture = p
			s := p.State()
			owner.state = &s
			response.State = &s
			response.Handle = id
		}
	}
	h.adopt(id, owner)
	return response, err
}
