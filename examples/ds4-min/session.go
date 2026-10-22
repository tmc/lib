package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"unicode/utf8"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

var (
	errEmptyPrompt     = errors.New("empty prompt")
	errContextOverflow = errors.New("prompt exceeds context")
)

type sessionBackend interface {
	Reset() error
	Eval(token, position int) error
	Logits() []float32
}

type persistentSessionBackend interface {
	sessionBackend
	Save(io.Writer) error
	Load(io.Reader) error
}

type prefillChunkSetter interface {
	SetPrefillChunk(int) error
}

func configurePrefillChunk(backend sessionBackend, chunk int) error {
	if chunk == 2048 {
		return nil
	}
	setter, ok := backend.(prefillChunkSetter)
	if !ok {
		return fmt.Errorf("prefill-chunk: backend does not support batching: %w", dsv4.ErrUnimplemented)
	}
	if err := setter.SetPrefillChunk(chunk); err != nil {
		return fmt.Errorf("prefill-chunk: %w", err)
	}
	return nil
}

// A batchEvaluator evaluates a whole run of tokens starting at position. A
// backend implements it when the run costs less than the sum of its Eval
// calls. The result must be indistinguishable from calling Eval for each token
// in order: the same timeline advance, and the same Logits afterwards as after
// evaluating the final token of the run. Logits for the earlier positions of a
// run are never observed, so a backend may skip work only those logits need.
type batchEvaluator interface {
	EvalBatch(tokens []int, position int) error
}

type session struct {
	backend sessionBackend
	context int
	stop    stopSet
	tokens  []int
	logits  []float32
}

const (
	localSessionHeaderBytes = 32
	localSessionVersion     = 1
	maxLocalSessionBytes    = 1 << 30
	maxLocalSessionLogits   = 1 << 20
)

var localSessionMagic = [8]byte{'D', 'S', '4', 'M', 'S', 'S', 'N', 0}

func newSession(backend sessionBackend, context int, stop stopSet) (*session, error) {
	if backend == nil {
		return nil, errors.New("missing session backend")
	}
	if context <= 1 {
		return nil, errors.New("context must exceed one token")
	}
	if len(stop) == 0 {
		return nil, errors.New("empty stop-token set")
	}
	return &session{backend: backend, context: context, stop: stop}, nil
}

func (s *session) Sync(prompt []int) error {
	if len(prompt) == 0 {
		return errEmptyPrompt
	}
	if len(prompt) >= s.context {
		return fmt.Errorf("%w: length %d, context %d", errContextOverflow, len(prompt), s.context)
	}
	common := commonPrefix(s.tokens, prompt)
	if common != len(s.tokens) {
		if err := s.backend.Reset(); err != nil {
			return fmt.Errorf("reset session: %w", err)
		}
		s.tokens = s.tokens[:0]
		s.logits = nil
		common = 0
	}
	if run := prompt[common:]; len(run) != 0 {
		if err := s.evalRun(run, common); err != nil {
			return err
		}
		s.tokens = append(s.tokens, run...)
	}
	if common != len(prompt) || s.logits == nil {
		s.logits = append(s.logits[:0], s.backend.Logits()...)
	}
	return nil
}

// Reset discards the evaluated tokens, so the next Sync evaluates its whole
// prompt.
func (s *session) Reset() error {
	if err := s.backend.Reset(); err != nil {
		return fmt.Errorf("reset session: %w", err)
	}
	s.tokens = s.tokens[:0]
	s.logits = nil
	return nil
}

func (s *session) Save(w io.Writer) error {
	if s == nil || s.backend == nil {
		return dsv4.ErrClosed
	}
	if w == nil {
		return errors.New("nil session writer")
	}
	p, ok := s.backend.(persistentSessionBackend)
	if !ok {
		return fmt.Errorf("save session: %w", dsv4.ErrUnimplemented)
	}
	var state bytes.Buffer
	if err := p.Save(&state); err != nil {
		return fmt.Errorf("save backend state: %w", err)
	}
	if state.Len() > maxLocalSessionBytes || len(s.tokens) >= s.context || len(s.logits) > maxLocalSessionLogits {
		return errors.New("save session: state exceeds limit")
	}
	var header [localSessionHeaderBytes]byte
	copy(header[:8], localSessionMagic[:])
	binary.LittleEndian.PutUint32(header[8:12], localSessionVersion)
	binary.LittleEndian.PutUint32(header[12:16], uint32(s.context))
	binary.LittleEndian.PutUint32(header[16:20], uint32(len(s.tokens)))
	binary.LittleEndian.PutUint32(header[20:24], uint32(len(s.logits)))
	binary.LittleEndian.PutUint64(header[24:32], uint64(state.Len()))
	if err := writeSessionFull(w, header[:]); err != nil {
		return fmt.Errorf("write session header: %w", err)
	}
	var word [4]byte
	for _, token := range s.tokens {
		binary.LittleEndian.PutUint32(word[:], uint32(int32(token)))
		if err := writeSessionFull(w, word[:]); err != nil {
			return fmt.Errorf("write session token: %w", err)
		}
	}
	for _, logit := range s.logits {
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(logit))
		if err := writeSessionFull(w, word[:]); err != nil {
			return fmt.Errorf("write session logit: %w", err)
		}
	}
	if err := writeSessionFull(w, state.Bytes()); err != nil {
		return fmt.Errorf("write backend state: %w", err)
	}
	return nil
}

func (s *session) Load(r io.Reader) error {
	if s == nil || s.backend == nil {
		return dsv4.ErrClosed
	}
	if r == nil {
		return errors.New("nil session reader")
	}
	p, ok := s.backend.(persistentSessionBackend)
	if !ok {
		return fmt.Errorf("load session: %w", dsv4.ErrUnimplemented)
	}
	data, err := io.ReadAll(io.LimitReader(r, maxLocalSessionBytes+1))
	if err != nil {
		return fmt.Errorf("read session: %w", err)
	}
	if len(data) > maxLocalSessionBytes || len(data) < localSessionHeaderBytes ||
		!bytes.Equal(data[:8], localSessionMagic[:]) || binary.LittleEndian.Uint32(data[8:12]) != localSessionVersion {
		return errors.New("read session: invalid envelope")
	}
	if binary.LittleEndian.Uint32(data[12:16]) != uint32(s.context) {
		return errors.New("read session: context does not match")
	}
	tokenCount := binary.LittleEndian.Uint32(data[16:20])
	logitCount := binary.LittleEndian.Uint32(data[20:24])
	stateBytes := binary.LittleEndian.Uint64(data[24:32])
	if tokenCount >= uint32(s.context) || logitCount > maxLocalSessionLogits || stateBytes > uint64(maxLocalSessionBytes) {
		return errors.New("read session: invalid sizes")
	}
	words := uint64(tokenCount) + uint64(logitCount)
	if words > (uint64(len(data)-localSessionHeaderBytes))/4 {
		return errors.New("read session: truncated envelope")
	}
	offset := localSessionHeaderBytes
	tokens := make([]int, tokenCount)
	for i := range tokens {
		tokens[i] = int(int32(binary.LittleEndian.Uint32(data[offset : offset+4])))
		offset += 4
	}
	logits := make([]float32, logitCount)
	for i := range logits {
		logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4
	}
	if stateBytes != uint64(len(data)-offset) {
		return errors.New("read session: state length mismatch")
	}
	if err := p.Load(bytes.NewReader(data[offset:])); err != nil {
		return fmt.Errorf("load backend state: %w", err)
	}
	s.tokens, s.logits = tokens, logits
	return nil
}

func writeSessionFull(w io.Writer, data []byte) error {
	for len(data) > 0 {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n <= 0 || n > len(data) {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}

func (s *session) Accept(token int) (stop bool, err error) {
	if s.stop.Contains(token) {
		return true, nil
	}
	if len(s.tokens) >= s.context {
		return false, fmt.Errorf("%w: generation has no free token", errContextOverflow)
	}
	if err := s.backend.Eval(token, len(s.tokens)); err != nil {
		return false, fmt.Errorf("evaluate generated token: %w", err)
	}
	s.tokens = append(s.tokens, token)
	s.logits = append(s.logits[:0], s.backend.Logits()...)
	return false, nil
}

// evalRun evaluates tokens starting at position, preferring the backend's
// batched prefill when it has one.
func (s *session) evalRun(tokens []int, position int) error {
	if b, ok := s.backend.(batchEvaluator); ok {
		if err := b.EvalBatch(tokens, position); err != nil {
			return fmt.Errorf("evaluate tokens %d through %d: %w", position, position+len(tokens)-1, err)
		}
		return nil
	}
	for i, token := range tokens {
		if err := s.backend.Eval(token, position+i); err != nil {
			return fmt.Errorf("evaluate token %d: %w", position+i, err)
		}
	}
	return nil
}

func commonPrefix(a, b []int) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

type textAssembler struct {
	pending []byte
}

func (a *textAssembler) Add(fragment []byte) (string, error) {
	a.pending = append(a.pending, fragment...)
	if !utf8.FullRune(a.pending) {
		return "", nil
	}
	if !utf8.Valid(a.pending) {
		a.pending = a.pending[:0]
		return "", errors.New("invalid UTF-8 output")
	}
	out := string(a.pending)
	a.pending = a.pending[:0]
	return out, nil
}

func (a *textAssembler) Finish() error {
	if len(a.pending) != 0 {
		return errors.New("incomplete UTF-8 output")
	}
	return nil
}
