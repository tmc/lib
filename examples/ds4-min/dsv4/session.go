package dsv4

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

var (
	errEmptyPrompt     = errors.New("empty prompt")
	errContextOverflow = errors.New("prompt exceeds context")
	errSessionFormat   = errors.New("invalid session snapshot")
)

const (
	sessionMagic       = "DS4SESS1"
	sessionVersion     = uint32(1)
	maxSessionBytes    = int64(2 << 30)
	maxSessionLogits   = 1 << 27
	sessionHeaderBytes = 32
)

// StopSet contains token IDs that terminate generation. Its zero value is
// unusable; construct one with NewStopSet.
type StopSet map[int]struct{}

// NewStopSet copies ids into a set owned by the returned value.
func NewStopSet(ids ...int) StopSet {
	set := make(StopSet, len(ids))
	for _, id := range ids {
		set[id] = struct{}{}
	}
	return set
}

func (s StopSet) contains(token int) bool {
	_, ok := s[token]
	return ok
}

// Session owns the token timeline for one request and delegates mutable model
// history to its independent State. A Session must not be shared concurrently.
type Session struct {
	backend Backend
	state   State
	context int
	stop    StopSet
	tokens  []int
	logits  []float32
	broken  bool
	closed  bool
}

// NewSession creates an independent state from backend and binds it to a new
// request timeline.
func NewSession(backend Backend, context int, stop StopSet) (*Session, error) {
	if backend == nil {
		return nil, errors.New("missing backend")
	}
	if context <= 1 {
		return nil, errors.New("context must exceed one token")
	}
	if len(stop) == 0 {
		return nil, errors.New("empty stop-token set")
	}
	state, err := backend.NewState(context)
	if err != nil {
		return nil, fmt.Errorf("new state: %w", err)
	}
	if state == nil {
		return nil, errors.New("backend returned nil state")
	}
	return &Session{backend: backend, state: state, context: context, stop: stop}, nil
}

// Sync evaluates prompt, reusing the common prefix already held by the
// session. The prompt is copied so callers may reuse their input buffer.
func (s *Session) Sync(prompt []int) error {
	if s.closed || s.state == nil {
		return ErrClosed
	}
	if s.broken {
		return ErrDesynchronized
	}
	if len(prompt) == 0 {
		return errEmptyPrompt
	}
	if len(prompt) >= s.context {
		return fmt.Errorf("%w: length %d, context %d", errContextOverflow, len(prompt), s.context)
	}
	common := commonPrefix(s.tokens, prompt)
	if common != len(s.tokens) {
		if err := s.Reset(); err != nil {
			return err
		}
		// Reset has made the backend diverge from the old timeline. Clear the
		// timeline before evaluating the replacement so an error cannot leave
		// stale metadata advertising a state that no longer exists.
		s.tokens = s.tokens[:0]
		s.logits = nil
		common = 0
	}
	if run := prompt[common:]; len(run) != 0 {
		if err := s.evalRun(run, common); err != nil {
			s.tokens = nil
			s.logits = nil
			s.broken = true
			return fmt.Errorf("sync session: %w: %w", ErrDesynchronized, err)
		}
		s.tokens = append(s.tokens, run...)
	}
	if common != len(prompt) || s.logits == nil {
		s.logits = append(s.logits[:0], s.backend.Logits(s.state)...)
	}
	return nil
}

// Reset discards the evaluated token timeline and mutable state.
func (s *Session) Reset() error {
	if s.closed || s.state == nil {
		return ErrClosed
	}
	if s.broken {
		return ErrDesynchronized
	}
	if err := s.state.Reset(); err != nil {
		s.broken = true
		return fmt.Errorf("reset state: %w", err)
	}
	s.tokens = s.tokens[:0]
	s.logits = nil
	return nil
}

// Accept evaluates one generated token. It returns true when token is a stop
// token; stop tokens are not added to the session state.
func (s *Session) Accept(token int) (bool, error) {
	if s.closed || s.state == nil {
		return false, ErrClosed
	}
	if s.broken {
		return false, ErrDesynchronized
	}
	if s.stop.contains(token) {
		return true, nil
	}
	if len(s.tokens) >= s.context {
		return false, fmt.Errorf("%w: generation has no free token", errContextOverflow)
	}
	if err := s.backend.Eval(s.state, token, len(s.tokens)); err != nil {
		s.tokens = nil
		s.logits = nil
		s.broken = true
		return false, fmt.Errorf("evaluate generated token: %w: %w", ErrDesynchronized, err)
	}
	s.tokens = append(s.tokens, token)
	s.logits = append(s.logits[:0], s.backend.Logits(s.state)...)
	return false, nil
}

// Close releases the request-owned State. The Backend remains usable for
// other sessions. The first call makes the session unusable. If release fails,
// Close retains ownership and subsequent calls retry it. Successful release is
// idempotent; methods that use state return ErrClosed after the first call.
func (s *Session) Close() error {
	s.closed = true
	s.tokens = nil
	s.logits = nil
	s.broken = false
	if s.state == nil {
		return nil
	}
	if err := s.state.Close(); err != nil {
		return err
	}
	s.state = nil
	return nil
}

// Logits returns a copy of the most recent logits.
func (s *Session) Logits() []float32 { return append([]float32(nil), s.logits...) }

// Tokens returns a copy of the accepted token timeline.
func (s *Session) Tokens() []int { return append([]int(nil), s.tokens...) }

// Save writes a versioned session envelope containing the token timeline,
// latest logits, and backend-owned State payload. The format is ds4-min's
// private format; it is not compatible with upstream ds4 files. The backend
// payload remains opaque to this package.
func (s *Session) Save(w io.Writer) error {
	if s.closed || s.state == nil {
		return ErrClosed
	}
	if s.broken {
		return ErrDesynchronized
	}
	if w == nil {
		return errors.New("nil session writer")
	}
	var state bytes.Buffer
	if err := s.state.Save(&state); err != nil {
		return fmt.Errorf("save backend state: %w", err)
	}
	if int64(state.Len()) > maxSessionBytes {
		return fmt.Errorf("save backend state: %w", errSessionFormat)
	}
	if len(s.tokens) >= s.context {
		return fmt.Errorf("save tokens: %w", errContextOverflow)
	}
	if len(s.logits) > maxSessionLogits {
		return fmt.Errorf("save logits: %w", errSessionFormat)
	}
	var header [sessionHeaderBytes]byte
	copy(header[:8], sessionMagic)
	binary.LittleEndian.PutUint32(header[8:12], sessionVersion)
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
	if _, err := io.Copy(w, &state); err != nil {
		return fmt.Errorf("write backend state: %w", err)
	}
	return nil
}

func writeSessionFull(w io.Writer, data []byte) error {
	for len(data) != 0 {
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

// Load replaces the session with a snapshot produced by Save. The context
// size must match. Parsing is completed before the backend is called, and a
// backend is required by State's contract to leave itself unchanged when its
// Load fails. Extra trailing bytes are rejected.
func (s *Session) Load(r io.Reader) error {
	if s.closed || s.state == nil {
		return ErrClosed
	}
	if s.broken {
		return ErrDesynchronized
	}
	if r == nil {
		return errors.New("nil session reader")
	}
	data, err := io.ReadAll(io.LimitReader(r, maxSessionBytes+1))
	if err != nil {
		return fmt.Errorf("read session: %w", err)
	}
	if int64(len(data)) > maxSessionBytes {
		return fmt.Errorf("read session: %w", errSessionFormat)
	}
	if len(data) < sessionHeaderBytes || string(data[:8]) != sessionMagic || binary.LittleEndian.Uint32(data[8:12]) != sessionVersion {
		return fmt.Errorf("read session: %w", errSessionFormat)
	}
	contextSize := binary.LittleEndian.Uint32(data[12:16])
	if contextSize != uint32(s.context) {
		return fmt.Errorf("read session: context %d does not match %d", contextSize, s.context)
	}
	tokenCount := binary.LittleEndian.Uint32(data[16:20])
	logitCount := binary.LittleEndian.Uint32(data[20:24])
	stateBytes := binary.LittleEndian.Uint64(data[24:32])
	if tokenCount >= uint32(s.context) || logitCount > maxSessionLogits || stateBytes > uint64(maxSessionBytes) {
		return fmt.Errorf("read session: %w", errSessionFormat)
	}
	words := uint64(tokenCount) + uint64(logitCount)
	if words > (uint64(len(data)-sessionHeaderBytes))/4 {
		return fmt.Errorf("read session: %w", errSessionFormat)
	}
	offset := sessionHeaderBytes
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
		return fmt.Errorf("read session: %w", errSessionFormat)
	}
	if err := s.state.Load(bytes.NewReader(data[offset:])); err != nil {
		return fmt.Errorf("load backend state: %w", err)
	}
	s.tokens = tokens
	s.logits = logits
	return nil
}

func (s *Session) evalRun(tokens []int, position int) error {
	if b, ok := s.backend.(batchSessionEvaluator); ok {
		if err := b.EvalBatch(s.state, tokens, position); err != nil {
			return fmt.Errorf("evaluate tokens %d through %d: %w", position, position+len(tokens)-1, err)
		}
		return nil
	}
	for i, token := range tokens {
		if err := s.backend.Eval(s.state, token, position+i); err != nil {
			return fmt.Errorf("evaluate token %d: %w", position+i, err)
		}
	}
	return nil
}

// batchSessionEvaluator is intentionally private to the session adapter. The
// public BatchEvaluator contract may be reused by backend packages without
// making this request timeline depend on their optional capability.
type batchSessionEvaluator interface {
	EvalBatch(State, []int, int) error
}

func commonPrefix(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
