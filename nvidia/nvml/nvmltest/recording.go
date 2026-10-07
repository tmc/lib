package nvmltest

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"

	"github.com/tmc/lib/nvidia/nvml"
)

// Version is the recording schema version.
const Version = 1
const maxBytes = 16 << 20
const maxCalls = 4096

// Queries is the selected consumer interface. The compile-time assertion
// against nvml.Device detects signature drift without mirroring every raw call.
type Queries interface {
	GpmSupported() (bool, error)
	ProcessesUtilization(uint64) ([]nvml.ProcessUtilizationInfoV1, error)
	FabricInfo() (nvml.GpuFabricInfoV, error)
	FieldValuesFor(...uint32) ([]nvml.FieldValue, error)
	GpuInstanceList(uint32) ([]nvml.GpuInstance, error)
}

var _ Queries = nvml.Device(0)

// Provenance identifies the source of responses. Measured recordings require
// GPU, driver and source identities; MIG records the observed mode, including
// unavailable. Synthetic scripts must set Measured to false.
type Provenance struct {
	GPU      string `json:"gpu"`
	Driver   string `json:"driver"`
	MIG      string `json:"mig"`
	Source   string `json:"source"`
	Measured bool   `json:"measured"`
}

// Failure preserves NVML error classification and wrapped/joined causes.
// Kind is result, symbol, wrap, join or other. Other errors preserve text.
type Failure struct {
	Kind    string      `json:"kind"`
	Message string      `json:"message"`
	Code    nvml.Result `json:"code,omitempty"`
	Symbol  string      `json:"symbol,omitempty"`
	Causes  []*Failure  `json:"causes,omitempty"`
}

func (f *Failure) Error() string { return f.Message }

// Unwrap reconstructs the underlying classified errors.
func (f *Failure) Unwrap() []error {
	if f.Kind == "result" {
		return []error{f.Code}
	}
	if f.Kind == "symbol" {
		var cause error
		if len(f.Causes) > 0 {
			cause = f.Causes[0]
		}
		return []error{&nvml.SymbolError{Symbol: f.Symbol, Err: cause}}
	}
	causes := make([]error, len(f.Causes))
	for i, c := range f.Causes {
		causes[i] = c
	}
	return causes
}

// Call is one ordered invocation. Args and Result are typed JSON values for
// Method. Per-entry status fields remain inside Result.
type Call struct {
	Method string          `json:"method"`
	Args   json.RawMessage `json:"args"`
	Result json.RawMessage `json:"result"`
	Error  *Failure        `json:"error,omitempty"`
}

// Recording is a versioned ordered script with source provenance.
type Recording struct {
	Version    int        `json:"version"`
	Provenance Provenance `json:"provenance"`
	Calls      []Call     `json:"calls"`
}

// Read reads one recording, rejecting unknown schema fields, trailing input
// and recordings larger than 16 MiB or 4096 calls.
func Read(r io.Reader) (Recording, error) {
	b, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return Recording{}, fmt.Errorf("read recording: %w", err)
	}
	if len(b) > maxBytes {
		return Recording{}, errors.New("recording exceeds byte limit")
	}
	var rec Recording
	if err := decode(b, &rec); err != nil {
		return Recording{}, fmt.Errorf("decode recording: %w", err)
	}
	if err := validate(rec); err != nil {
		return Recording{}, err
	}
	return rec, nil
}

// Write validates and writes one recording. It does not close w.
func Write(w io.Writer, rec Recording) error {
	if err := validate(rec); err != nil {
		return err
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if len(b)+1 > maxBytes {
		return errors.New("recording exceeds byte limit")
	}
	b = append(b, '\n')
	for len(b) > 0 {
		n, err := w.Write(b)
		if n > 0 {
			b = b[n:]
		}
		if err != nil {
			return fmt.Errorf("write recording: %w", err)
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func validate(rec Recording) error {
	if rec.Version != Version {
		return fmt.Errorf("unsupported recording version %d", rec.Version)
	}
	if rec.Provenance.Measured && (rec.Provenance.GPU == "" || rec.Provenance.Driver == "" || rec.Provenance.MIG == "" || rec.Provenance.Source == "") {
		return errors.New("measured recording lacks provenance")
	}
	if len(rec.Calls) > maxCalls {
		return errors.New("recording exceeds call limit")
	}
	for i, c := range rec.Calls {
		if bytes.Equal(bytes.TrimSpace(c.Args), []byte("null")) {
			return fmt.Errorf("call %d: null arguments", i)
		}
		if (c.Method == "GpmSupported" || c.Method == "FabricInfo") && bytes.Equal(bytes.TrimSpace(c.Result), []byte("null")) {
			return fmt.Errorf("call %d: null result", i)
		}
		var args, result any
		switch c.Method {
		case "GpmSupported":
			args = &struct{}{}
			result = new(bool)
		case "FabricInfo":
			args = &struct{}{}
			result = new(nvml.GpuFabricInfoV)
		case "ProcessesUtilization":
			args = &struct{ LastSeen uint64 }{}
			result = new([]nvml.ProcessUtilizationInfoV1)
		case "FieldValuesFor":
			args = &struct{ IDs []uint32 }{}
			result = new([]nvml.FieldValue)
		case "GpuInstanceList":
			args = &struct{ ProfileID uint32 }{}
			result = new([]nvml.GpuInstance)
		default:
			return fmt.Errorf("call %d: unknown method %q", i, c.Method)
		}
		for _, v := range []struct {
			b   []byte
			dst any
		}{{c.Args, args}, {c.Result, result}} {
			if err := decode(v.b, v.dst); err != nil {
				return fmt.Errorf("call %d: malformed value: %w", i, err)
			}
		}
		// Arguments must name every parameter, including a zero cursor and
		// nil ID slice. An omitted parameter is not a recorded invocation.
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(c.Args, &fields); err != nil {
			return fmt.Errorf("call %d: malformed arguments: %w", i, err)
		}
		for j := 0; j < reflect.TypeOf(args).Elem().NumField(); j++ {
			field := reflect.TypeOf(args).Elem().Field(j)
			value, ok := fields[field.Name]
			if !ok || field.Type.Kind() != reflect.Slice && bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
				return fmt.Errorf("call %d: missing or null argument %s", i, field.Name)
			}
		}
		if err := validateFailure(c.Error, 0); err != nil {
			return fmt.Errorf("call %d: %w", i, err)
		}
	}
	return nil
}

func validateFailure(f *Failure, depth int) error {
	if f == nil {
		return nil
	}
	if depth > 32 || f.Message == "" {
		return errors.New("invalid error record")
	}
	switch f.Kind {
	case "result":
		if f.Code == nvml.Success || len(f.Causes) != 0 {
			return errors.New("invalid result error")
		}
	case "symbol":
		if f.Symbol == "" || len(f.Causes) > 1 {
			return errors.New("invalid symbol error")
		}
	case "wrap":
		if len(f.Causes) != 1 {
			return errors.New("invalid wrapped error")
		}
	case "join":
		if len(f.Causes) == 0 {
			return errors.New("invalid joined error")
		}
	case "other":
		if len(f.Causes) != 0 {
			return errors.New("invalid opaque error")
		}
	default:
		return errors.New("unknown error kind")
	}
	for _, c := range f.Causes {
		if c == nil {
			return errors.New("nil error cause")
		}
		if err := validateFailure(c, depth+1); err != nil {
			return err
		}
	}
	return nil
}

func failure(err error, depth int) (*Failure, error) {
	remaining := 4096
	return captureFailure(err, depth, &remaining)
}

func captureFailure(err error, depth int, remaining *int) (*Failure, error) {
	if err == nil {
		return nil, nil
	}
	if depth > 32 || *remaining == 0 {
		return nil, errors.New("error record exceeds depth or cause limit")
	}
	*remaining--
	v := reflect.ValueOf(err)
	if v.Kind() == reflect.Pointer && v.IsNil() {
		return nil, errors.New("nil error cause")
	}
	f := &Failure{Kind: "other", Message: err.Error()}
	var causes []error
	switch e := err.(type) {
	case *Failure:
		*f = *e
		f.Causes = nil
		for _, cause := range e.Causes {
			causes = append(causes, cause)
		}
	case nvml.Result:
		f.Kind = "result"
		f.Code = e
	case *nvml.CallError:
		f.Kind = "wrap"
		causes = []error{e.Result}
	case *nvml.SymbolError:
		f.Kind = "symbol"
		f.Symbol = e.Symbol
		if e.Err != nil {
			causes = []error{e.Err}
		}
	case interface{ Unwrap() []error }:
		f.Kind = "join"
		for _, e := range e.Unwrap() {
			if e != nil {
				causes = append(causes, e)
			}
		}
	case interface{ Unwrap() error }:
		if e.Unwrap() != nil {
			f.Kind = "wrap"
			causes = []error{e.Unwrap()}
		}
	}
	if len(causes) > *remaining {
		return nil, errors.New("error record exceeds cause limit")
	}
	for _, cause := range causes {
		c, err := captureFailure(cause, depth+1, remaining)
		if err != nil {
			return nil, err
		}
		f.Causes = append(f.Causes, c)
	}
	return f, nil
}

func canonical(v []byte) ([]byte, error) {
	d := json.NewDecoder(bytes.NewReader(v))
	d.UseNumber()
	var value any
	if err := d.Decode(&value); err != nil {
		return nil, err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return nil, errors.New("trailing JSON")
	}
	return json.Marshal(value)
}

// Replay is a race-safe script. Calls must arrive in recording order with
// matching arguments. A mismatch does not consume the next scripted call.
type Replay struct {
	mu   sync.Mutex
	rec  Recording
	next int
}

// NewReplay validates and deep-copies a recording.
func NewReplay(rec Recording) (*Replay, error) {
	var b bytes.Buffer
	if err := Write(&b, rec); err != nil {
		return nil, err
	}
	copy, err := Read(&b)
	if err != nil {
		return nil, err
	}
	return &Replay{rec: copy}, nil
}

// Done reports whether all scripted calls were consumed.
func (r *Replay) Done() bool { r.mu.Lock(); defer r.mu.Unlock(); return r.next == len(r.rec.Calls) }

func (r *Replay) query(method string, args, result any) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.next >= len(r.rec.Calls) {
		return fmt.Errorf("unexpected %s after script end", method)
	}
	c := r.rec.Calls[r.next]
	b, err := json.Marshal(args)
	if err != nil {
		return err
	}
	want, err := canonical(c.Args)
	if err != nil {
		return err
	}
	got, err := canonical(b)
	if err != nil {
		return err
	}
	if c.Method != method || !bytes.Equal(want, got) {
		return fmt.Errorf("call %d: method or arguments mismatch", r.next)
	}
	if err := json.Unmarshal(c.Result, result); err != nil {
		return err
	}
	r.next++
	if c.Error != nil {
		return c.Error
	}
	return nil
}

// GpmSupported replays the next support query.
func (r *Replay) GpmSupported() (v bool, err error) {
	err = r.query("GpmSupported", struct{}{}, &v)
	return
}

// FabricInfo replays the next fabric query.
func (r *Replay) FabricInfo() (v nvml.GpuFabricInfoV, err error) {
	err = r.query("FabricInfo", struct{}{}, &v)
	return
}

// ProcessesUtilization replays a query with the exact timestamp cursor.
func (r *Replay) ProcessesUtilization(lastSeen uint64) (v []nvml.ProcessUtilizationInfoV1, err error) {
	err = r.query("ProcessesUtilization", struct{ LastSeen uint64 }{lastSeen}, &v)
	return
}

// FieldValuesFor replays a query with fields in the requested order.
func (r *Replay) FieldValuesFor(ids ...uint32) (v []nvml.FieldValue, err error) {
	err = r.query("FieldValuesFor", struct{ IDs []uint32 }{ids}, &v)
	return
}

// GpuInstanceList replays a query for profileID.
func (r *Replay) GpuInstanceList(profileID uint32) (v []nvml.GpuInstance, err error) {
	err = r.query("GpuInstanceList", struct{ ProfileID uint32 }{profileID}, &v)
	return
}

var _ Queries = (*Replay)(nil)

// Recorder serializes queries and retains their responses in call order.
// It does not initialize or shut down the caller's NVML instance.
// The zero value is not usable; call NewRecorder.
type Recorder struct {
	mu     sync.Mutex
	device Queries
	rec    Recording
	size   int
}

// NewRecorder records device queries with the given provenance.
func NewRecorder(device Queries, p Provenance) (*Recorder, error) {
	if device == nil || reflect.ValueOf(device).Kind() == reflect.Pointer && reflect.ValueOf(device).IsNil() {
		return nil, errors.New("nil query device")
	}
	rec := Recording{Version: Version, Provenance: p, Calls: []Call{}}
	if err := validate(rec); err != nil {
		return nil, err
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return nil, err
	}
	if len(b)+1 > maxBytes {
		return nil, errors.New("recording envelope exceeds byte limit")
	}
	return &Recorder{device: device, rec: rec, size: len(b) + 1}, nil
}

// Recording returns a deep copy of the calls recorded so far.
func (r *Recorder) Recording() (Recording, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	var b bytes.Buffer
	if err := Write(&b, r.rec); err != nil {
		return Recording{}, err
	}
	return Read(&b)
}

func (r *Recorder) query(method string, args, result any, call func() (any, error)) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.rec.Calls) >= maxCalls || r.size >= maxBytes {
		return errors.New("recording budget exhausted")
	}
	a, err := json.Marshal(args)
	if err != nil {
		return err
	}
	separator := 0
	if len(r.rec.Calls) != 0 {
		separator = 1
	}
	minimum, err := json.Marshal(Call{Method: method, Args: a, Result: json.RawMessage("0")})
	if err != nil {
		return err
	}
	if len(minimum)+separator > maxBytes-r.size {
		return errors.New("recording arguments exceed byte budget")
	}
	v, callErr := call()
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	f, err := failure(callErr, 0)
	if err != nil {
		return fmt.Errorf("record query error: %w", err)
	}
	c := Call{Method: method, Args: a, Result: b, Error: f}
	if err := validate(Recording{Version: Version, Calls: []Call{c}}); err != nil {
		return err
	}
	encoded, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if len(encoded)+separator > maxBytes-r.size {
		return errors.New("recorded response exceeds byte budget")
	}
	r.size += len(encoded) + separator
	r.rec.Calls = append(r.rec.Calls, c)
	if err := json.Unmarshal(b, result); err != nil {
		return err
	}
	return callErr
}

// GpmSupported records the underlying support query.
func (r *Recorder) GpmSupported() (v bool, err error) {
	err = r.query("GpmSupported", struct{}{}, &v, func() (any, error) { return r.device.GpmSupported() })
	return
}

// FabricInfo records the underlying fabric query.
func (r *Recorder) FabricInfo() (v nvml.GpuFabricInfoV, err error) {
	err = r.query("FabricInfo", struct{}{}, &v, func() (any, error) { return r.device.FabricInfo() })
	return
}

// ProcessesUtilization records the query and timestamp cursor.
func (r *Recorder) ProcessesUtilization(lastSeen uint64) (v []nvml.ProcessUtilizationInfoV1, err error) {
	err = r.query("ProcessesUtilization", struct{ LastSeen uint64 }{lastSeen}, &v, func() (any, error) { return r.device.ProcessesUtilization(lastSeen) })
	return
}

// FieldValuesFor records the query and ordered field IDs.
func (r *Recorder) FieldValuesFor(ids ...uint32) (v []nvml.FieldValue, err error) {
	err = r.query("FieldValuesFor", struct{ IDs []uint32 }{ids}, &v, func() (any, error) { return r.device.FieldValuesFor(ids...) })
	return
}

// GpuInstanceList records the profile query.
func (r *Recorder) GpuInstanceList(profileID uint32) (v []nvml.GpuInstance, err error) {
	err = r.query("GpuInstanceList", struct{ ProfileID uint32 }{profileID}, &v, func() (any, error) { return r.device.GpuInstanceList(profileID) })
	return
}

var _ Queries = (*Recorder)(nil)
