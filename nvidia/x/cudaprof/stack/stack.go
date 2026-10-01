package stack

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sync"
)

const Version = 1
const JournalName = "language-stacks-v1.jsonl"
const JournalEnv = "CUDAPROF_STACK_JOURNAL"

// Frame is one symbolized language frame.
type Frame struct {
	Function string `json:"function"`
	File     string `json:"file,omitempty"`
	Line     uint32 `json:"line,omitempty"`
}

// Record retains a stack and its exact external correlation reference.
// Process identifies a process generation, rather than a reusable PID alone.
type Record struct {
	Type           string  `json:"type,omitempty"`
	StackID        uint64  `json:"stack_id,omitempty"`
	Attempted      uint64  `json:"attempted,omitempty"`
	Recorded       uint64  `json:"recorded,omitempty"`
	Dropped        uint64  `json:"dropped,omitempty"`
	TruncatedCount uint64  `json:"truncated_count,omitempty"`
	Final          bool    `json:"final,omitempty"`
	Version        int     `json:"version"`
	Process        string  `json:"process"`
	Session        string  `json:"session,omitempty"`
	Generation     uint64  `json:"generation,omitempty"`
	PID            int     `json:"pid"`
	Language       string  `json:"language"`
	Attribution    string  `json:"attribution"`
	ExternalKind   uint32  `json:"external_kind"`
	ExternalID     uint64  `json:"external_id"`
	Frames         []Frame `json:"frames"`
	Truncated      bool    `json:"truncated,omitempty"`
	Status         string  `json:"status"`
}

// Limits bounds journal bytes, records and frames. Zero fields use defaults.
type Limits struct {
	Bytes   int64
	Records int
	Frames  int
}

func (l Limits) defaults() Limits {
	if l.Bytes <= 0 {
		l.Bytes = 32 << 20
	}
	if l.Records <= 0 {
		l.Records = 100000
	}
	if l.Frames <= 0 {
		l.Frames = 128
	}
	return l
}

func validate(r Record, limit int) error {
	if r.Version != Version {
		return fmt.Errorf("unsupported stack version %d", r.Version)
	}
	if r.Process == "" || r.PID <= 0 {
		return fmt.Errorf("missing stack process identity")
	}
	if r.Language != "go" && r.Language != "python" {
		return fmt.Errorf("unsupported stack language %q", r.Language)
	}
	if r.Attribution != "scope_entry" && r.Attribution != "submission" {
		return fmt.Errorf("unsupported stack attribution %q", r.Attribution)
	}
	if r.ExternalID == 0 {
		return fmt.Errorf("missing stack external ID")
	}
	if r.Status != "attached" && r.Status != "push_failed" && r.Status != "unavailable" {
		return fmt.Errorf("unsupported stack status %q", r.Status)
	}
	if len(r.Frames) > limit {
		return fmt.Errorf("stack frame limit exceeded")
	}
	if r.Status == "attached" && len(r.Frames) == 0 {
		return fmt.Errorf("attached stack has no frames")
	}
	for _, f := range r.Frames {
		if f.Function == "" {
			return fmt.Errorf("stack frame has no function")
		}
	}
	return nil
}

// Read reads a bounded JSONL journal. It rejects unknown fields, duplicate
// attached references, incomplete lines and inconsistent process identities.
func Read(rd io.Reader, limits Limits) ([]Record, error) {
	l := limits.defaults()
	s := bufio.NewScanner(io.LimitReader(rd, l.Bytes+1))
	s.Buffer(make([]byte, 4096), 1<<20)
	s.Split(splitLines)
	var lines int
	var out []Record
	var size int64
	seen := make(map[string]bool)
	processes := make(map[string]int)
	identities := make(map[string]Record)
	definitions := make(map[string]Record)
	coverageSeen := make(map[string]bool)
	attached := make(map[string]uint64)
	failed := make(map[string]uint64)
	var references int
	for s.Scan() {
		size += int64(len(s.Bytes()) + 1)
		if size > l.Bytes {
			return nil, fmt.Errorf("stack journal byte limit exceeded")
		}
		lines++
		if lines > 2*l.Records+1 {
			return nil, fmt.Errorf("stack journal record limit exceeded")
		}
		if err := checkJSON(s.Bytes()); err != nil {
			return nil, err
		}
		var r Record
		d := json.NewDecoder(bytes.NewReader(s.Bytes()))
		d.DisallowUnknownFields()
		if err := d.Decode(&r); err != nil {
			return nil, fmt.Errorf("read stack record %d: %w", len(out)+1, err)
		}
		var extra any
		if d.Decode(&extra) != io.EOF {
			return nil, fmt.Errorf("trailing stack record data")
		}
		if r.Type == "coverage" {
			identity, ok := identities[r.Process]
			if !r.Final || r.PID <= 0 || (r.Language != "go" && r.Language != "python") || (ok && (r.PID != identity.PID || r.Session != identity.Session || r.Generation != identity.Generation || r.Language != identity.Language)) {
				return nil, fmt.Errorf("invalid stack coverage identity")
			}
			if r.Version != Version || r.Process == "" || r.Recorded > r.Attempted || r.Dropped != r.Attempted-r.Recorded || coverageSeen[r.Process] || r.Recorded != attached[r.Process] || r.Dropped < failed[r.Process] {
				return nil, fmt.Errorf("invalid stack coverage")
			}
			coverageSeen[r.Process] = true
			out = append(out, r)
			continue
		}
		if coverageSeen[r.Process] {
			return nil, fmt.Errorf("stack record after final coverage")
		}
		if identity, ok := identities[r.Process]; ok && (r.PID != identity.PID || r.Session != identity.Session || r.Generation != identity.Generation || r.Language != identity.Language) {
			return nil, fmt.Errorf("mixed stack process identity")
		}
		identities[r.Process] = r
		keyDefinition := fmt.Sprintf("%s/%d", r.Process, r.StackID)
		if r.Type == "definition" {
			if r.Version != Version || r.Process == "" || r.PID <= 0 {
				return nil, fmt.Errorf("invalid stack definition identity")
			}
			if r.StackID == 0 || len(r.Frames) == 0 || len(r.Frames) > l.Frames || definitions[keyDefinition].StackID != 0 {
				return nil, fmt.Errorf("invalid stack definition")
			}
			definitions[keyDefinition] = r
			continue
		}
		if r.Type == "reference" {
			if len(r.Frames) != 0 || definitions[keyDefinition].StackID == 0 {
				return nil, fmt.Errorf("missing stack definition")
			}
			def := definitions[keyDefinition]
			if def.Session != r.Session || def.Generation != r.Generation || def.PID != r.PID || def.Language != r.Language {
				return nil, fmt.Errorf("stack definition identity mismatch")
			}
			r.Frames = def.Frames
		}
		if r.Type != "" && r.Type != "reference" {
			return nil, fmt.Errorf("unsupported stack record type")
		}
		references++
		if references > l.Records {
			return nil, fmt.Errorf("stack journal record limit exceeded")
		}
		if err := validate(r, l.Frames); err != nil {
			return nil, err
		}
		if pid, ok := processes[r.Process]; ok && pid != r.PID {
			return nil, fmt.Errorf("inconsistent stack process identity")
		}
		processes[r.Process] = r.PID
		key := fmt.Sprintf("%s/%d/%d", r.Process, r.ExternalKind, r.ExternalID)
		if seen[key] {
			return nil, fmt.Errorf("duplicate stack reference")
		}
		seen[key] = true
		if r.Status == "attached" {
			attached[r.Process]++
		} else {
			failed[r.Process]++
		}
		out = append(out, r)
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("read stack journal: %w", err)
	}
	return out, nil
}

// Journal serializes bounded stack records. It reports exhaustion to the caller
// so a producer can retain its drop count instead of silently omitting stacks.
type Journal struct {
	mu          sync.Mutex
	w           io.Writer
	limits      Limits
	bytes       int64
	records     int
	definitions map[string]uint64
}

// NewJournal returns a journal writing JSONL to w.
func NewJournal(w io.Writer, limits Limits) *Journal {
	return &Journal{w: w, limits: limits.defaults()}
}

// Record writes one record or returns an error without writing on exhaustion.
func (j *Journal) Record(r Record) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.w == nil {
		return fmt.Errorf("write stack journal: missing writer")
	}
	if err := validate(r, j.limits.Frames); err != nil {
		return err
	}
	if j.definitions == nil {
		j.definitions = make(map[string]uint64)
	}
	key := fmt.Sprintf("%s/%s/%d/%d/%s/%#v", r.Process, r.Session, r.Generation, r.PID, r.Language, r.Frames)
	id := j.definitions[key]
	var prefix []byte
	if id == 0 && len(r.Frames) > 0 {
		id = uint64(len(j.definitions) + 1)
		def := Record{Type: "definition", Version: Version, Process: r.Process, PID: r.PID, Session: r.Session, Generation: r.Generation, Language: r.Language, StackID: id, Frames: r.Frames}
		prefix, _ = json.Marshal(def)
		prefix = append(prefix, '\n')
	}
	if len(r.Frames) > 0 {
		r.Type = "reference"
		r.StackID = id
		r.Frames = nil
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(prefix, append(b, '\n')...)
	if j.records >= j.limits.Records || j.bytes+int64(len(b)) > j.limits.Bytes {
		return fmt.Errorf("stack journal limit exceeded")
	}
	n, err := j.w.Write(b)
	j.bytes += int64(n)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	if err != nil {
		return fmt.Errorf("write stack journal: %w", err)
	}
	j.definitions[key] = id
	j.records++
	return nil
}

// WriteCoverage writes a final coverage receipt. Counts refer to attempted
// attribution records, including failed pushes and journal exhaustion.
func WriteCoverage(w io.Writer, r Record) error {
	r.Type = "coverage"
	r.Version = Version
	r.Final = true
	r.Frames = nil
	if r.Process == "" || r.Recorded > r.Attempted || r.Dropped > r.Attempted-r.Recorded {
		return fmt.Errorf("invalid stack coverage")
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

func splitLines(data []byte, eof bool) (int, []byte, error) {
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		return i + 1, data[:i], nil
	}
	if eof && len(data) > 0 {
		return 0, nil, fmt.Errorf("incomplete stack journal line")
	}
	return 0, nil, nil
}

func checkJSON(b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	var value func() error
	value = func() error {
		t, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := t.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]bool)
			for d.More() {
				t, err := d.Token()
				if err != nil {
					return err
				}
				key, ok := t.(string)
				if !ok || seen[key] {
					return fmt.Errorf("duplicate stack JSON field")
				}
				seen[key] = true
				if err := value(); err != nil {
					return err
				}
			}
		case '[':
			for d.More() {
				if err := value(); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("invalid stack JSON delimiter")
		}
		_, err = d.Token()
		return err
	}
	if err := value(); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing stack JSON data")
	}
	return nil
}
