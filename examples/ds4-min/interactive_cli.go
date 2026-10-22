package main

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4chat"
	"github.com/tmc/lib/examples/ds4-min/sampler"
)

// interactiveBackend transfers the already allocated request state into Run.
// The outer sessionBackend closer continues to own the immutable model.
type interactiveBackend struct {
	dsv4.Backend
	state dsv4.State
}

func (b *interactiveBackend) NewState(context int) (dsv4.State, error) {
	if b.state == nil {
		return b.Backend.NewState(context)
	}
	s := b.state
	b.state = nil
	return s, nil
}

func runInteractive(cfg config, tok *tokenizer, input io.Reader, output, log io.Writer) (err error) {
	b, closer, err := openSessionBackend(cfg)
	if err != nil {
		return err
	}
	defer func() { err = finishBackendCleanup(err, closer) }()
	if err := configurePrefillChunk(b, cfg.prefillChunk); err != nil {
		return err
	}
	backend, err := transferInteractiveState(b)
	if err != nil {
		return err
	}
	defer func() {
		if backend.state != nil {
			if closeErr := backend.state.Close(); closeErr != nil {
				requests := append(backendErrorOwners(err), backend.state)
				err = &backendCleanupOwner{cause: errors.Join(err, closeErr), request: &backendCleanupGroup{requests}}
			} else {
				backend.state = nil
			}
		}
	}()
	return runInteractiveState(context.Background(), cfg, tok, backend, input, output, log)
}

func transferInteractiveState(b sessionBackend) (*interactiveBackend, error) {
	var owner *stateSessionBackend
	switch b := b.(type) {
	case *stateSessionBackend:
		owner = b
	case *batchStateSessionBackend:
		owner = b.stateSessionBackend
	default:
		return nil, errors.New("interactive chat requires independent state")
	}
	backend := &interactiveBackend{Backend: owner.backend, state: owner.state}
	// The runner now owns the request state. The deferred closer owns weights.
	owner.state = nil
	return backend, nil
}

func runInteractiveState(ctx context.Context, cfg config, tok *tokenizer, backend dsv4.Backend, input io.Reader, output, log io.Writer) error {
	stops := dsv4.NewStopSet()
	for id := range tok.tokens {
		if tok.IsStop(id) {
			stops[id] = struct{}{}
		}
	}
	sampling := interactiveSampling{params: cfg.sampling, seed: cfg.seed, randomSeed: interactiveClockSeed, sampler: sampler.New(len(tok.tokens)), log: log}
	maxTokenBytes := 1
	for _, piece := range tok.tokens {
		maxTokenBytes = max(maxTokenBytes, len(piece))
	}
	commands := interactiveCommands{mode: cfg.thinkMode, context: cfg.context, output: output, maxTokenBytes: maxTokenBytes}
	var vectors *steeringVectors
	if cfg.dirSteeringAttn != 0 || cfg.dirSteeringFFN != 0 {
		var err error
		vectors, err = loadSteeringVectors(cfg.dirSteeringFile)
		if err != nil {
			return err
		}
	}
	return dsv4chat.Run(ctx, input, output, dsv4chat.Config{
		Backend: backend, Context: cfg.context, Stops: stops, Tokens: cfg.tokens, Prompt: "> ",
		Render: func(messages []dsv4chat.Message) ([]int, error) {
			return tok.renderInteractive(cfg.system, messages, commands.mode, commands.context)
		},
		BeforeTurn: sampling.begin,
		Sample:     func(logits []float32) (int, error) { return sampling.sample(logits), nil },
		Decode:     func(id int) (string, error) { data, err := tok.Decode([]int{id}); return string(data), err },
		Command:    commands.command, Errors: log,
		ContextChanged:  func(size int) { commands.context = size; fmt.Fprintf(output, "Context: %d.\n", size) },
		IsStop:          func(id int) bool { return tok.IsStopForMode(id, commands.effectiveMode()) },
		AssistantPrefix: func() []int { return tok.interactivePrefix(commands.effectiveMode()) },
		ConfigureState: func(state dsv4.State) error {
			if vectors == nil {
				return nil
			}
			loader, ok := state.(steeringLoader)
			if !ok {
				return fmt.Errorf("directional steering: %w", dsv4.ErrUnimplemented)
			}
			ffn := float32(cfg.dirSteeringFFN)
			if commands.steering != nil {
				ffn = commands.steering.steeringFFN()
			}
			return loader.loadSteering(vectors, float32(cfg.dirSteeringAttn), ffn)
		},
		StateChanged: func(state dsv4.State) { commands.steering, _ = state.(steeringControl) },
	})
}

// renderInteractive retains generated token IDs verbatim. In particular, the
// assistant's printable text is not a reversible representation of its tokens.
func (t *tokenizer) renderInteractive(system string, turns []dsv4chat.Message, mode ThinkMode, context int) ([]int, error) {
	if len(turns) == 0 || len(turns)%2 != 1 {
		return nil, errors.New("interactive history must end with a user message")
	}
	if mode == ThinkMax && context > 0 && context < maxThinkContext {
		mode = ThinkHigh
	}
	ids := []int{t.bos}
	if mode == ThinkMax {
		part, err := t.Encode(maxThinkPrefix)
		if err != nil {
			return nil, err
		}
		ids = append(ids, part...)
	}
	part, err := t.Encode(system)
	if err != nil {
		return nil, err
	}
	ids = append(ids, part...)
	for i, turn := range turns {
		if i%2 == 0 {
			if turn.Role != "user" {
				return nil, errors.New("expected user message")
			}
			ids = append(ids, t.user)
			part, err := t.Encode(turn.Text)
			if err != nil {
				return nil, err
			}
			ids = append(ids, part...)
			prefix := t.interactivePrefix(mode)
			if i+1 < len(turns) && len(turns[i+1].Prefix) > 0 {
				prefix = turns[i+1].Prefix
			}
			ids = append(ids, prefix...)
		} else {
			if turn.Role != "assistant" {
				return nil, errors.New("expected assistant message")
			}
			ids = append(ids, turn.Tokens...)
			ids = append(ids, t.eos)
		}
	}
	return ids, nil
}

func (t *tokenizer) interactivePrefix(mode ThinkMode) []int {
	if mode == ThinkNone {
		return []int{t.assistant, t.thinkingEnd}
	}
	return []int{t.assistant, t.thinking}
}
