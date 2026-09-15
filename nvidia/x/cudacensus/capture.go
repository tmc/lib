package cudacensus

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Launch is one kernel launch, holding every field a capture records about it.
// There is no byte count and no argument metadata because the capture records
// none; see the package documentation.
type Launch struct {
	Name        string // mangled device symbol, as captured
	StartNs     int64  // device clock
	EndNs       int64  // device clock
	Grid        string // e.g. "43x1x1"
	Block       string // e.g. "256x1x1"
	Registers   int64
	SharedMem   int64 // static shared memory, bytes
	DeviceID    int64
	StreamID    int64
	GraphID     int64 // zero when the launch was not replayed from a graph
	Correlation int64
}

// DurationNs reports the device execution span.
func (l Launch) DurationNs() int64 { return l.EndNs - l.StartNs }

// Transfer is a memcpy or memset. Unlike a kernel, a transfer records the
// bytes it moved.
type Transfer struct {
	Kind      string // "memcpy" or "memset"
	Direction string // "HtoD", "DtoH", ...; empty for memset
	Bytes     uint64
	StartNs   int64 // device clock
	EndNs     int64 // device clock
	StreamID  int64
}

// Span is an application annotation: a phase such as "prefill", or one token.
// Its timestamps are on the device clock, converted by ReadDir.
type Span struct {
	Name    string
	StartNs int64
	EndNs   int64
	Seq     int64
	Labels  map[string]string
}

// Capture is one decoded capture directory. Launches are sorted by start time.
type Capture struct {
	// ClockOffsetNs is device minus host, added to every span timestamp so
	// spans and launches share one clock.
	ClockOffsetNs int64

	Launches  []Launch
	Transfers []Transfer
	Spans     []Span
}

// jsonRecord is the union of the record kinds this package reads. Fields the
// capture omits stay zero.
type jsonRecord struct {
	Kind string `json:"kind"`

	// clock_sync
	UnixNs  int64 `json:"unix_ns"`
	CuptiNs int64 `json:"cupti_ns"`

	// kernel, memcpy, memset, span
	StartNs int64 `json:"start_ns"`
	EndNs   int64 `json:"end_ns"`

	// kernel
	RawSymbol   string `json:"raw_symbol"`
	Grid        string `json:"grid"`
	Block       string `json:"block"`
	Registers   int64  `json:"registers"`
	SharedMem   int64  `json:"shared_mem"`
	DeviceID    int64  `json:"device_id"`
	StreamID    int64  `json:"stream_id"`
	GraphID     int64  `json:"graph_id"`
	Correlation int64  `json:"correlation_id"`

	// memcpy, memset
	Bytes uint64 `json:"bytes"`
	Dir   string `json:"dir"`

	// span
	Name    string            `json:"name"`
	EvalSeq int64             `json:"eval_seq"`
	Labels  map[string]string `json:"labels"`
}

// ReadDir decodes the capture directory at dir.
//
// Device records are read from events.*.jsonl and application spans from
// events.app.*.jsonl. Spans are recorded on the host clock; ReadDir converts
// them to the device clock using the capture's clock_sync record, so a span
// bound can be compared against a launch timestamp. A capture with spans but
// no clock_sync record is an error, because the two clocks cannot then be
// related and any window taken over them would be silently wrong.
func ReadDir(dir string) (*Capture, error) {
	appFiles, err := filepath.Glob(filepath.Join(dir, "events.app.*.jsonl"))
	if err != nil {
		return nil, err
	}
	allFiles, err := filepath.Glob(filepath.Join(dir, "events.*.jsonl"))
	if err != nil {
		return nil, err
	}
	app := make(map[string]bool, len(appFiles))
	for _, f := range appFiles {
		app[f] = true
	}
	var devFiles []string
	for _, f := range allFiles {
		if !app[f] {
			devFiles = append(devFiles, f)
		}
	}
	if len(devFiles) == 0 && len(appFiles) == 0 {
		return nil, fmt.Errorf("cudacensus: no events.*.jsonl in %s", dir)
	}

	c := new(Capture)
	haveOffset := false
	for _, f := range devFiles {
		if err := c.readFile(f, &haveOffset); err != nil {
			return nil, err
		}
	}
	var spans []Span
	for _, f := range appFiles {
		s, err := readSpans(f)
		if err != nil {
			return nil, err
		}
		spans = append(spans, s...)
	}
	if len(spans) > 0 && !haveOffset {
		return nil, fmt.Errorf("cudacensus: %s has spans but no clock_sync record", dir)
	}
	for i := range spans {
		spans[i].StartNs += c.ClockOffsetNs
		spans[i].EndNs += c.ClockOffsetNs
	}
	c.Spans = spans

	sort.Slice(c.Launches, func(i, j int) bool { return c.Launches[i].StartNs < c.Launches[j].StartNs })
	sort.Slice(c.Transfers, func(i, j int) bool { return c.Transfers[i].StartNs < c.Transfers[j].StartNs })
	sort.SliceStable(c.Spans, func(i, j int) bool { return c.Spans[i].StartNs < c.Spans[j].StartNs })
	return c, nil
}

func (c *Capture) readFile(path string, haveOffset *bool) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return eachRecord(f, path, func(r jsonRecord) error {
		switch r.Kind {
		case "clock_sync":
			c.ClockOffsetNs = r.CuptiNs - r.UnixNs
			*haveOffset = true
		case "kernel":
			c.Launches = append(c.Launches, Launch{
				Name: r.RawSymbol, StartNs: r.StartNs, EndNs: r.EndNs,
				Grid: r.Grid, Block: r.Block,
				Registers: r.Registers, SharedMem: r.SharedMem,
				DeviceID: r.DeviceID, StreamID: r.StreamID,
				GraphID: r.GraphID, Correlation: r.Correlation,
			})
		case "memcpy", "memset":
			c.Transfers = append(c.Transfers, Transfer{
				Kind: r.Kind, Direction: r.Dir, Bytes: r.Bytes,
				StartNs: r.StartNs, EndNs: r.EndNs, StreamID: r.StreamID,
			})
		}
		return nil
	})
}

func readSpans(path string) ([]Span, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var spans []Span
	err = eachRecord(f, path, func(r jsonRecord) error {
		if r.Kind != "span" {
			return nil
		}
		spans = append(spans, Span{Name: r.Name, StartNs: r.StartNs, EndNs: r.EndNs, Seq: r.EvalSeq, Labels: r.Labels})
		return nil
	})
	return spans, err
}

// eachRecord decodes one JSON record per line. Blank lines are skipped; a
// malformed line is an error naming the file and line, because a capture that
// decodes partway is a capture whose totals are wrong by an unknown amount.
func eachRecord(r io.Reader, path string, fn func(jsonRecord) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec jsonRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return fmt.Errorf("cudacensus: %s:%d: %w", path, n, err)
		}
		if err := fn(rec); err != nil {
			return err
		}
	}
	return sc.Err()
}

// Span returns the first span with the given name. Workloads name their
// phases differently ("decode" in one capture, "generation" in another), so
// the name is the caller's to supply.
func (c *Capture) Span(name string) (Span, bool) {
	for _, s := range c.Spans {
		if s.Name == name {
			return s, true
		}
	}
	return Span{}, false
}

// SpansNamed returns every span with the given name, in start order.
func (c *Capture) SpansNamed(name string) []Span {
	var out []Span
	for _, s := range c.Spans {
		if s.Name == name {
			out = append(out, s)
		}
	}
	return out
}

// Window returns the launches that started at or after startNs and before
// endNs. Both bounds are on the device clock, as Span reports them.
func (c *Capture) Window(startNs, endNs int64) []Launch {
	lo := sort.Search(len(c.Launches), func(i int) bool { return c.Launches[i].StartNs >= startNs })
	hi := sort.Search(len(c.Launches), func(i int) bool { return c.Launches[i].StartNs >= endNs })
	return c.Launches[lo:hi]
}
