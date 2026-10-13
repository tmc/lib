package stack

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/buildinfo"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Stage U and UP controls. None of them starts a child process or decodes a
// trace: D1 admits no decoder. Stage 1 and 2 controls need a reviewed
// decoder and a separate process grant and are not in this file.

var (
	goTraceTestHeader   = []byte("go 1.26 trace\x00\x00\x00")
	goTraceTestProducer = GoProducer{GoVersion: "go1.27.2", GOOS: "darwin", GOARCH: "arm64"}
)

// Absolute, clean, nonexistent paths: any file system step would turn them
// into ErrGoTraceScratch or ErrGoTraceCustody.
const (
	goTraceNoTool    = "/nonexistent-c6d1/trace"
	goTraceNoScratch = "/nonexistent-c6d1/scratch"
)

func goTraceTestInput(n int) []byte {
	b := make([]byte, n)
	copy(b, goTraceTestHeader)
	return b
}

// goTraceVector is the illustrative record of the design's fixed vector. It
// admits nothing.
func goTraceVector() goTraceAdmission {
	return goTraceAdmission{
		ID:            "66228f0e314dee80e8a43286a0e94e0e37206a125e558dcc7d47c06962e77510",
		ToolSHA256:    strings.Repeat("a", 64),
		ToolBytes:     1234,
		ToolPath:      "cmd/trace",
		ToolGoVersion: "go1.27.2",
		ToolGOOS:      "darwin",
		ToolGOARCH:    "arm64",
		RecipeSHA256:  strings.Repeat("b", 64),
		Header:        26,
		Producer:      GoProducer{GoVersion: "go1.27.2", GOOS: "darwin", GOARCH: "arm64"},
		Receipt:       strings.Repeat("c", 64),
		Review:        strings.Repeat("d", 64),
	}
}

func withID(r goTraceAdmission) goTraceAdmission {
	r.ID = goTraceAdmissionID(encodeGoTraceAdmission(r))
	return r
}

// countingEncoder returns an encoder that counts its calls.
func countingEncoder(n *int) func(goTraceAdmission) []byte {
	return func(r goTraceAdmission) []byte {
		*n++
		return encodeGoTraceAdmission(r)
	}
}

// goTraceNoChild fails the test if fn starts a child.
func goTraceNoChild(t *testing.T, fn func()) {
	t.Helper()
	before := goTraceStarts.Load()
	fn()
	if n := goTraceStarts.Load() - before; n != 0 {
		t.Errorf("started %d children", n)
	}
}

// A2v: the encoding reproduces the fixed vector and covers every field.
func TestGoTraceAdmissionVector(t *testing.T) {
	r := goTraceVector()
	e := encodeGoTraceAdmission(r)
	if len(e) != 410 {
		t.Errorf("len(E) = %d, want 410", len(e))
	}
	prefix, _ := hex.DecodeString("730000001d6375646170726f662e676f74726163652e61646d697373696f6e2e76327300000040" + "61")
	for _, want := range [][]byte{prefix} {
		if !bytes.HasPrefix(e, want) {
			t.Errorf("E prefix = %x", e[:len(want)])
		}
	}
	for _, field := range []string{"6900000000000004d2", "7300000009636d642f7472616365", "69000000000000001a"} {
		want, _ := hex.DecodeString(field)
		if !bytes.Contains(e, want) {
			t.Errorf("E lacks %s", field)
		}
	}
	if got := goTraceAdmissionID(e); got != r.ID {
		t.Errorf("ID = %s, want %s", got, r.ID)
	}
	mutations := []func(*goTraceAdmission){
		func(r *goTraceAdmission) { r.ToolSHA256 = strings.Repeat("e", 64) },
		func(r *goTraceAdmission) { r.ToolBytes++ },
		func(r *goTraceAdmission) { r.ToolPath = "cmd/trace2" },
		func(r *goTraceAdmission) { r.ToolGoVersion = "go1.27.1" },
		func(r *goTraceAdmission) { r.ToolGOOS = "linux" },
		func(r *goTraceAdmission) { r.ToolGOARCH = "amd64" },
		func(r *goTraceAdmission) { r.RecipeSHA256 = strings.Repeat("e", 64) },
		func(r *goTraceAdmission) { r.Header = 25 },
		func(r *goTraceAdmission) { r.Producer.GoVersion = "go1.27.1" },
		func(r *goTraceAdmission) { r.Producer.GOOS = "linux" },
		func(r *goTraceAdmission) { r.Producer.GOARCH = "amd64" },
		func(r *goTraceAdmission) { r.Receipt = strings.Repeat("e", 64) },
		func(r *goTraceAdmission) { r.Review = strings.Repeat("e", 64) },
	}
	seen := map[string]bool{r.ID: true}
	for i, m := range mutations {
		c := goTraceVector()
		m(&c)
		id := goTraceAdmissionID(encodeGoTraceAdmission(c))
		if seen[id] {
			t.Errorf("mutation %d left the ID unchanged", i)
		}
		seen[id] = true
	}
	other := goTraceVector()
	other.ToolSHA256 = strings.Repeat("f", 64)
	table := []goTraceAdmission{goTraceVector(), withID(other)}
	if err := validateGoTraceAdmissions(table, encodeGoTraceAdmission); err != nil {
		t.Errorf("valid table refused: %v", err)
	}
}

// A2s: syntactic defects refuse the table before anything is encoded.
func TestGoTraceAdmissionSyntax(t *testing.T) {
	type mutation struct {
		name string
		f    func(*goTraceAdmission)
	}
	var rows []mutation
	labels := map[string]func(*goTraceAdmission) *string{
		"ToolPath":           func(r *goTraceAdmission) *string { return &r.ToolPath },
		"ToolGoVersion":      func(r *goTraceAdmission) *string { return &r.ToolGoVersion },
		"ToolGOOS":           func(r *goTraceAdmission) *string { return &r.ToolGOOS },
		"ToolGOARCH":         func(r *goTraceAdmission) *string { return &r.ToolGOARCH },
		"Producer.GoVersion": func(r *goTraceAdmission) *string { return &r.Producer.GoVersion },
		"Producer.GOOS":      func(r *goTraceAdmission) *string { return &r.Producer.GOOS },
		"Producer.GOARCH":    func(r *goTraceAdmission) *string { return &r.Producer.GOARCH },
	}
	for name, field := range labels {
		for _, v := range []string{"a\x00b", "", strings.Repeat("a", 65), "a b", "é"} {
			rows = append(rows, mutation{fmt.Sprintf("%s=%q", name, v), func(r *goTraceAdmission) { *field(r) = v }})
		}
	}
	digests := map[string]func(*goTraceAdmission) *string{
		"ToolSHA256":   func(r *goTraceAdmission) *string { return &r.ToolSHA256 },
		"RecipeSHA256": func(r *goTraceAdmission) *string { return &r.RecipeSHA256 },
		"Receipt":      func(r *goTraceAdmission) *string { return &r.Receipt },
		"Review":       func(r *goTraceAdmission) *string { return &r.Review },
	}
	for name, field := range digests {
		for _, v := range []string{strings.Repeat("a", 63), strings.Repeat("A", 64)} {
			rows = append(rows, mutation{fmt.Sprintf("%s=%q", name, v), func(r *goTraceAdmission) { *field(r) = v }})
		}
	}
	rows = append(rows,
		mutation{"ToolBytes=0", func(r *goTraceAdmission) { r.ToolBytes = 0 }},
		mutation{"ToolBytes=64MiB+1", func(r *goTraceAdmission) { r.ToolBytes = 64<<20 + 1 }},
		mutation{"Header=0", func(r *goTraceAdmission) { r.Header = 0 }},
		mutation{"Header=256", func(r *goTraceAdmission) { r.Header = 256 }},
		mutation{"ToolGoVersion=1.27", func(r *goTraceAdmission) { r.ToolGoVersion = "1.27" }},
		mutation{"Producer.GoVersion=1.27", func(r *goTraceAdmission) { r.Producer.GoVersion = "1.27" }},
	)
	if len(rows) != 7*5+4*2+6 {
		t.Fatalf("rows = %d", len(rows))
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			r := goTraceVector()
			row.f(&r)
			r = withID(r) // a correct ID must not rescue an invalid record
			var calls int
			err := validateGoTraceAdmissions([]goTraceAdmission{goTraceVector(), r}, countingEncoder(&calls))
			if err == nil || calls != 0 {
				t.Errorf("err = %v, encoder calls = %d", err, calls)
			}
		})
	}
}

// A2d: duplicate keys refuse the table before anything is encoded.
func TestGoTraceAdmissionDuplicate(t *testing.T) {
	a := goTraceVector()
	b := goTraceVector()
	b.ToolPath = "cmd/trace2"
	b = withID(b)
	var calls int
	if err := validateGoTraceAdmissions([]goTraceAdmission{a, b}, countingEncoder(&calls)); err == nil || calls != 0 {
		t.Errorf("err = %v, encoder calls = %d", err, calls)
	}
}

// A2i: a well-formed but edited ID is recomputed and refused.
func TestGoTraceAdmissionID(t *testing.T) {
	r := goTraceVector()
	id := []byte(r.ID)
	id[10] = "0123456789abcdef"[(strings.IndexByte("0123456789abcdef", id[10])+1)%16]
	r.ID = string(id)
	var calls int
	if err := validateGoTraceAdmissions([]goTraceAdmission{r}, countingEncoder(&calls)); err == nil || calls != 1 {
		t.Errorf("err = %v, encoder calls = %d", err, calls)
	}
}

// A2z: D1 admits nothing.
func TestGoTraceAdmissionEmpty(t *testing.T) {
	if len(goTraceAdmissions) != 0 {
		t.Fatalf("goTraceAdmissions has %d entries; adoption needs its own reviewed grant", len(goTraceAdmissions))
	}
	if err := validateGoTraceAdmissions(goTraceAdmissions, encodeGoTraceAdmission); err != nil {
		t.Fatal(err)
	}
}

// A2r: an invalid table refuses every call without file system access.
func TestGoTraceAdmissionRuntime(t *testing.T) {
	match := goTraceVector()
	match.Producer = goTraceTestProducer
	match = withID(match)
	bad := match
	bad.Header = 0
	edited := match
	edited.ID = strings.Repeat("0", 64)
	d := &GoTraceDecoder{Tool: goTraceNoTool, Scratch: goTraceNoScratch}
	for name, table := range map[string][]goTraceAdmission{
		"invalid record":   {match, withID(bad)},
		"edited id":        {edited},
		"duplicate record": {match, match},
	} {
		goTraceNoChild(t, func() {
			r, _, err := d.decode(context.Background(), goTraceTestInput(64), goTraceTestProducer, table, false, nil)
			if r != nil || !errors.Is(err, ErrGoTraceUnadmitted) || !strings.Contains(err.Error(), "invalid admission table") {
				t.Errorf("%s: %v", name, err)
			}
		})
	}
}

// A1, A7, C7a, L2, X1, F4 and PR: the pure prechecks P0-P7 in order, through
// both public entry points, with no file system access and no child.
func TestGoTracePrecheck(t *testing.T) {
	sel := func() *GoTraceDecoder { return &GoTraceDecoder{Tool: goTraceNoTool, Scratch: goTraceNoScratch} }
	with := func(f func(*GoTraceDecoder)) *GoTraceDecoder { d := sel(); f(d); return d }
	header25 := append([]byte("go 1.25 trace\x00\x00\x00"), 1, 2, 3)
	n9 := []byte("go 1.27 trace\x00not trace records")
	oversized := goTraceTestInput(32<<20 + 1)
	oversizedBad := make([]byte, 32<<20+1)
	for _, tt := range []struct {
		name     string
		d        *GoTraceDecoder
		cancel   bool
		data     []byte
		producer GoProducer
		want     error
	}{
		{"A1 empty table", sel(), false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceUnadmitted},
		{"A7 header 25", sel(), false, header25, goTraceTestProducer, ErrGoTraceUnadmitted},
		{"C7a empty tool", with(func(d *GoTraceDecoder) { d.Tool = "" }), false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceDecoderUnselected},
		{"C7a empty scratch", with(func(d *GoTraceDecoder) { d.Scratch = "" }), false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceDecoderUnselected},
		{"C7a relative tool", with(func(d *GoTraceDecoder) { d.Tool = "trace" }), false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceDecoderUnselected},
		{"C7a unclean tool", with(func(d *GoTraceDecoder) { d.Tool = "/x/../trace" }), false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceDecoderUnselected},
		{"C7a relative scratch", with(func(d *GoTraceDecoder) { d.Scratch = "scratch" }), false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceScratch},
		{"C7a unclean scratch", with(func(d *GoTraceDecoder) { d.Scratch = "/x/../scratch" }), false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceScratch},
		{"L2 input over limit", sel(), false, oversized, goTraceTestProducer, ErrGoTraceLimit},
		{"L2 input at limit", sel(), false, goTraceTestInput(32 << 20), goTraceTestProducer, ErrGoTraceUnadmitted},
		{"X1 cancelled", new(GoTraceDecoder), true, nil, GoProducer{}, context.Canceled},
		{"F4 historical header", sel(), false, n9, goTraceTestProducer, ErrGoTraceHeader},
		{"short input", sel(), false, goTraceTestHeader[:15], goTraceTestProducer, ErrGoTraceHeader},
		{"one-digit header", sel(), false, []byte("go 1.9 trace\x00\x00\x00\x00\x00"), goTraceTestProducer, ErrGoTraceHeader},
		{"PR limits before ctx", with(func(d *GoTraceDecoder) { d.Limits.Timeout = -1 }), true, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceLimits},
		{"PR ctx before tool", with(func(d *GoTraceDecoder) { d.Tool = "" }), true, goTraceTestInput(64), goTraceTestProducer, context.Canceled},
		{"PR tool before scratch", &GoTraceDecoder{Scratch: "scratch"}, false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceDecoderUnselected},
		{"PR relative tool before relative scratch", &GoTraceDecoder{Tool: "trace", Scratch: "scratch"}, false, goTraceTestInput(64), goTraceTestProducer, ErrGoTraceDecoderUnselected},
		{"PR scratch before input", with(func(d *GoTraceDecoder) { d.Scratch = "scratch" }), false, oversized, goTraceTestProducer, ErrGoTraceScratch},
		{"PR input before header", sel(), false, oversizedBad, goTraceTestProducer, ErrGoTraceLimit},
		{"PR header before producer", sel(), false, n9, GoProducer{}, ErrGoTraceHeader},
		{"PR producer incomplete", sel(), false, goTraceTestInput(64), GoProducer{GoVersion: "go1.27.2", GOOS: "darwin"}, ErrGoTraceUnadmitted},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tt.cancel {
				cancel()
			}
			goTraceNoChild(t, func() {
				r, err := tt.d.Validate(ctx, tt.data, tt.producer)
				if r != nil || !errors.Is(err, tt.want) || errors.Is(err, ErrGoTraceCustody) {
					t.Errorf("Validate = %v, want %v", err, tt.want)
				}
				g, err := tt.d.Analyze(ctx, tt.data, tt.producer, Record{Process: "p", PID: 1}, Limits{})
				if g != nil || !errors.Is(err, tt.want) {
					t.Errorf("Analyze = %v, want %v", err, tt.want)
				}
			})
		})
	}
}

// N0: a nil decoder is unselected, not a panic.
func TestGoTraceNilDecoder(t *testing.T) {
	var d *GoTraceDecoder
	goTraceNoChild(t, func() {
		if _, err := d.Validate(context.Background(), goTraceTestInput(64), goTraceTestProducer); !errors.Is(err, ErrGoTraceDecoderUnselected) {
			t.Errorf("Validate = %v", err)
		}
		if _, err := d.Analyze(context.Background(), goTraceTestInput(64), goTraceTestProducer, Record{}, Limits{}); !errors.Is(err, ErrGoTraceDecoderUnselected) {
			t.Errorf("Analyze = %v", err)
		}
	})
}

func goTraceLimitFields(l *GoTraceLimits) []*int64 {
	return []*int64{&l.InputBytes, &l.OutputBytes, &l.StderrBytes, &l.ToolBytes, (*int64)(&l.Timeout), (*int64)(&l.WaitDelay)}
}

// L1 and L1p: limits are checked first; valid limits pass on to selection.
func TestGoTraceLimitsPublic(t *testing.T) {
	ceilings := goTraceLimitFields(&goTraceCeilings)
	for i := range ceilings {
		for _, delta := range []int64{-1, 1} {
			var d GoTraceDecoder
			v := int64(-1)
			if delta > 0 {
				v = *ceilings[i] + 1
			}
			*goTraceLimitFields(&d.Limits)[i] = v
			goTraceNoChild(t, func() {
				if _, err := d.Validate(context.Background(), nil, GoProducer{}); !errors.Is(err, ErrGoTraceLimits) {
					t.Errorf("field %d = %d: %v", i, v, err)
				}
			})
		}
	}
	for _, l := range []GoTraceLimits{{}, goTraceCeilings} {
		d := GoTraceDecoder{Limits: l}
		if _, err := d.Validate(context.Background(), nil, GoProducer{}); !errors.Is(err, ErrGoTraceDecoderUnselected) {
			t.Errorf("limits %+v: %v", l, err)
		}
	}
}

// L1n: resolution defaults, keeps the ceiling and refuses everything else.
func TestGoTraceLimitsResolve(t *testing.T) {
	ceilings := goTraceLimitFields(&goTraceCeilings)
	for i, ceiling := range ceilings {
		for _, tt := range []struct {
			in, want int64
			ok       bool
		}{
			{0, *ceiling, true},
			{*ceiling, *ceiling, true},
			{*ceiling + 1, 0, false},
			{-1, 0, false},
			{math.MaxInt64, 0, false},
		} {
			var l GoTraceLimits
			*goTraceLimitFields(&l)[i] = tt.in
			got, err := resolveGoTraceLimits(l)
			if tt.ok != (err == nil) || tt.ok && *goTraceLimitFields(&got)[i] != tt.want {
				t.Errorf("field %d = %d: %+v, %v", i, tt.in, got, err)
			}
			if !tt.ok && !errors.Is(err, ErrGoTraceLimits) {
				t.Errorf("field %d = %d: %v", i, tt.in, err)
			}
		}
	}
	if goTraceCeilings.Timeout != 30*time.Second || goTraceCeilings.WaitDelay != 2*time.Second || goTraceCeilings.InputBytes != 32<<20 || goTraceCeilings.OutputBytes != 32<<20 || goTraceCeilings.StderrBytes != 1<<20 || goTraceCeilings.ToolBytes != 64<<20 {
		t.Errorf("ceilings = %+v", goTraceCeilings)
	}
}

func TestGoTraceHeaderParse(t *testing.T) {
	for _, tt := range []struct {
		in   string
		want int
		ok   bool
	}{
		{"go 1.26 trace\x00\x00\x00", 26, true},
		{"go 1.99 trace\x00\x00\x00rest", 99, true},
		{"go 1.26 trace\x00\x00", 0, false},
		{"go 1.05 trace\x00\x00\x00", 0, false},
		{"go 1.9 trace\x00\x00\x00\x00", 0, false},
		{"go 1.27 trace\x00not trace records", 0, false},
		{"go 1.2x trace\x00\x00\x00", 0, false},
		{"go 2.26 trace\x00\x00\x00", 0, false},
	} {
		got, ok := parseGoTraceHeader([]byte(tt.in))
		if got != tt.want || ok != tt.ok {
			t.Errorf("%q = %d, %v", tt.in, got, ok)
		}
	}
}

// goTraceScratchRecord is a valid in-test record matching the synthetic
// header-26 input and the test producer. It is never in goTraceAdmissions.
func goTraceScratchRecord() []goTraceAdmission {
	r := goTraceVector()
	r.Producer = goTraceTestProducer
	return []goTraceAdmission{withID(r)}
}

func goTraceListing(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

// C7b (UP): past an admitted tuple, scratch policy is checked before Tool is
// opened, and nothing is created inside or beside a refused scratch.
func TestGoTraceScratchPolicy(t *testing.T) {
	table := goTraceScratchRecord()
	for _, tt := range []struct {
		name  string
		setup func(parent string) string
	}{
		{"mode 0770", func(parent string) string {
			s := filepath.Join(parent, "s")
			goTraceMust(t, os.Mkdir(s, 0o700))
			goTraceMust(t, os.Chmod(s, 0o770))
			return s
		}},
		{"nonexistent", func(parent string) string { return filepath.Join(parent, "missing") }},
		{"symlink to 0700", func(parent string) string {
			target := filepath.Join(parent, "target")
			goTraceMust(t, os.Mkdir(target, 0o700))
			link := filepath.Join(parent, "link")
			goTraceMust(t, os.Symlink(target, link))
			return link
		}},
		{"regular file", func(parent string) string {
			f := filepath.Join(parent, "file")
			goTraceMust(t, os.WriteFile(f, nil, 0o600))
			return f
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			parent := t.TempDir()
			scratch := tt.setup(parent)
			before := goTraceListing(t, parent)
			d := &GoTraceDecoder{Tool: goTraceNoTool, Scratch: scratch}
			goTraceNoChild(t, func() {
				r, _, err := d.decode(context.Background(), goTraceTestInput(64), goTraceTestProducer, table, false, nil)
				if r != nil || !errors.Is(err, ErrGoTraceScratch) || errors.Is(err, ErrGoTraceCustody) {
					t.Errorf("err = %v", err)
				}
			})
			if after := goTraceListing(t, parent); !reflect.DeepEqual(before, after) {
				t.Errorf("parent %v -> %v", before, after)
			}
			for _, dir := range []string{scratch, filepath.Join(parent, "target")} {
				if st, err := os.Stat(dir); err == nil && st.IsDir() {
					if names := goTraceListing(t, dir); len(names) != 0 {
						t.Errorf("%s holds %v", dir, names)
					}
				}
			}
		})
	}
}

// C7b positive arm and K1u (UP): a valid scratch reaches the Tool step, whose
// nonexistent path is a custody failure; the per-call directory is removed,
// and an injected cleanup failure is joined, names the directory and leaves
// it in place.
func TestGoTraceScratchCleanup(t *testing.T) {
	table := goTraceScratchRecord()
	scratch := t.TempDir()
	goTraceMust(t, os.Chmod(scratch, 0o700))
	d := &GoTraceDecoder{Tool: goTraceNoTool, Scratch: scratch}
	goTraceNoChild(t, func() {
		_, _, err := d.decode(context.Background(), goTraceTestInput(64), goTraceTestProducer, table, false, nil)
		if !errors.Is(err, ErrGoTraceCustody) || errors.Is(err, ErrGoTraceCleanup) {
			t.Errorf("err = %v", err)
		}
	})
	if names := goTraceListing(t, scratch); len(names) != 0 {
		t.Errorf("scratch holds %v after removed cleanup", names)
	}

	var removed []string
	h := &goTraceHooks{removeAll: func(dir string) error {
		removed = append(removed, dir)
		return errors.New("injected cleanup failure")
	}}
	goTraceNoChild(t, func() {
		_, _, err := d.decode(context.Background(), goTraceTestInput(64), goTraceTestProducer, table, false, h)
		if !errors.Is(err, ErrGoTraceCustody) || !errors.Is(err, ErrGoTraceCleanup) || len(removed) != 1 || !strings.Contains(err.Error(), removed[0]) {
			t.Errorf("err = %v, removed %v", err, removed)
		}
	})
	if len(removed) == 1 {
		if st, err := os.Lstat(removed[0]); err != nil || !st.IsDir() || filepath.Dir(removed[0]) != scratch {
			t.Errorf("per-call directory %s: %v", removed[0], err)
		}
		err := os.RemoveAll(removed[0])
		t.Logf("gotrace-receipt: test_cleanup path=%s err=%v", removed[0], err)
	}
}

func goTraceMust(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// The closed structural allowlist, as a pure classifier. Raw-runner helper
// rows (X5*) and real-decoder rows (F2) are stage 1.
func TestGoTraceStructuralDiagnostic(t *testing.T) {
	match := []string{
		"expected batch event, got event 127",
		"invalid batch size 70000, maximum is 65536",
		"incomplete generation found; trace likely truncated",
		"invalid generation number 0",
		"broken trace: missing end-of-generation event, or generations are interleaved",
		"no frequency event found",
		"no clock snapshot event found",
	}
	for _, field := range []string{"gen", "M ID", "timestamp", "size"} {
		for _, cause := range []string{"EOF", "unexpected EOF", "binary: varint overflows a 64-bit integer"} {
			match = append(match, "error reading batch "+field+": "+cause)
		}
	}
	for _, line := range match {
		if got, ok := structuralDiagnostic([]byte(line + "\n")); !ok || got != line {
			t.Errorf("%q not matched", line)
		}
	}
	for _, stderr := range []string{
		"error reading batch gen|M ID: EOF\n",
		"error reading batch (gen|M ID|timestamp|size): EOF\n",
		"error reading batch gen: read runtime.trace: input/output error\n",
		"error reading batch gen: EOF extra\n",
		"failed to read full batch: read 3 but wanted 9\n",
		"copying batch data: EOF\n",
		"failed to read trace file: x\n",
		"unknown or unsupported trace version go 1.99\n",
		"unrecognized decoder diagnostic\n",
		"expected batch event, got event 127\nsecond line\n",
		"expected batch event, got event 127",
		"expected batch event, got event 127\n\n",
		"expected batch event, got event 127\r\n",
		"xexpected batch event, got event 127\n",
		"invalid batch size 1, maximum is 65537\n",
		"\n",
		"",
	} {
		if line, ok := structuralDiagnostic([]byte(stderr)); ok {
			t.Errorf("%q matched as %q", stderr, line)
		}
	}
	if len(goTraceStructural) != 8 {
		t.Errorf("allowlist has %d entries, want 8", len(goTraceStructural))
	}
}

// The overflow writer: exactly the limit is accepted, one more byte cancels
// once with errOutputLimit, and the other stream then fails too.
func TestGoTraceOutputWriter(t *testing.T) {
	ctx, cancel := context.WithCancelCause(context.Background())
	defer cancel(nil)
	var cancels int
	o := &goTraceOverflow{cancel: func(err error) { cancels++; cancel(err) }}
	stdout := &goTraceWriter{stream: "stdout", limit: 4, o: o}
	stderr := &goTraceWriter{stream: "stderr", limit: 4, keep: true, o: o}
	if n, err := stdout.Write([]byte("abcd")); n != 4 || err != nil {
		t.Fatalf("at limit: %d, %v", n, err)
	}
	if n, err := stderr.Write([]byte("ab")); n != 2 || err != nil {
		t.Fatalf("stderr: %d, %v", n, err)
	}
	if _, err := stdout.Write([]byte("e")); err != errOutputLimit {
		t.Fatalf("over limit: %v", err)
	}
	if _, err := stderr.Write([]byte("c")); err != errOutputLimit {
		t.Fatalf("after overflow: %v", err)
	}
	if cancels != 1 || context.Cause(ctx) != errOutputLimit || o.stream != "stdout" {
		t.Errorf("cancels = %d, cause = %v, stream = %q", cancels, context.Cause(ctx), o.stream)
	}
	if stdout.n != 4 || stdout.buf.Len() != 0 || stderr.buf.String() != "ab" {
		t.Errorf("stdout %d/%d, stderr %q", stdout.n, stdout.buf.Len(), stderr.buf.String())
	}
	big := &goTraceWriter{stream: "stdout", limit: math.MaxInt64, n: math.MaxInt64 - 1, o: &goTraceOverflow{cancel: func(error) {}}}
	if _, err := big.Write([]byte("ab")); err != errOutputLimit {
		t.Errorf("near MaxInt64: %v", err)
	}
}

// Classification precedence for the outcomes that need no exit status: the
// first cancellation cause wins, and exit 0 with stderr is not success.
func TestGoTraceClassify(t *testing.T) {
	cancelled := func(cause error) context.Context {
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(cause)
		return ctx
	}
	live := context.Background()
	for _, tt := range []struct {
		name       string
		ctx, child context.Context
		run        goTraceRun
		want       []error
		not        error
	}{
		{"overflow", live, cancelled(errOutputLimit), goTraceRun{overflow: "stdout"}, []error{ErrGoTraceLimit, errOutputLimit}, nil},
		{"overflow outranks caller", cancelled(context.Canceled), cancelled(errOutputLimit), goTraceRun{overflow: "stderr"}, []error{ErrGoTraceLimit, errOutputLimit}, context.Canceled},
		{"caller outranks timer", cancelled(context.Canceled), cancelled(errChildTime), goTraceRun{}, []error{context.Canceled}, ErrGoTraceLimit},
		{"timer", live, cancelled(errChildTime), goTraceRun{overflow: "stdout"}, []error{ErrGoTraceLimit, errChildTime}, errOutputLimit},
		{"wait delay", live, live, goTraceRun{waitErr: fmt.Errorf("wrapped: %w", exec.ErrWaitDelay)}, []error{ErrGoTraceReap}, nil},
		{"other wait error", live, live, goTraceRun{waitErr: errors.New("copy failed")}, []error{ErrGoTraceReap}, nil},
		{"exit 0 with stderr", live, live, goTraceRun{stderr: []byte("warning\n"), stderrBytes: 8}, []error{ErrGoTraceDecoderFailed}, ErrGoTraceMalformed},
	} {
		err := classifyGoTraceRun(tt.ctx, tt.child, &tt.run)
		for _, want := range tt.want {
			if !errors.Is(err, want) {
				t.Errorf("%s: %v, want %v", tt.name, err, want)
			}
		}
		if tt.not != nil && errors.Is(err, tt.not) {
			t.Errorf("%s: %v must not be %v", tt.name, err, tt.not)
		}
	}
	if err := classifyGoTraceRun(live, live, &goTraceRun{}); err != nil {
		t.Errorf("clean exit: %v", err)
	}
}

// Q6 (stack part): no implicit decoder dispatch remains in the package.
func TestGoTraceSourceGate(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var checked int
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		checked++
		for _, word := range []string{"exec.LookPath", `"tool", "trace"`, "UserHomeDir", "AnalyzeGoTrace", `goTool, "version"`} {
			if strings.Contains(string(data), word) {
				t.Errorf("%s contains %s", name, word)
			}
		}
	}
	if checked < 3 {
		t.Fatalf("checked %d non-test files", checked)
	}
}

// FD controls (UP). copyGoTraceTool and writeGoTraceInput are called
// directly, with the test binary as Tool and a synthetic in-test admission.
// They start no child, run no decoder and produce no trace; goTraceStarts
// must stay 0.

var (
	errGoTraceFDCloseInjected = errors.New("injected close failure")
	errGoTraceFDCloseWrite    = errors.New("injected write failure")
)

// goTraceFDCloseNoStart fails the test unless goTraceStarts is 0 before and
// after it.
func goTraceFDCloseNoStart(t *testing.T) {
	t.Helper()
	if n := goTraceStarts.Load(); n != 0 {
		t.Fatalf("goTraceStarts = %d before the test", n)
	}
	t.Cleanup(func() {
		if n := goTraceStarts.Load(); n != 0 {
			t.Errorf("goTraceStarts = %d", n)
		}
	})
}

// goTraceFDCloseTool returns the test binary and a synthetic record that
// matches its buildinfo, size and digest. The record is never in
// goTraceAdmissions.
func goTraceFDCloseTool(t *testing.T) (string, goTraceAdmission) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	if int64(len(data)) > goTraceCeilings.ToolBytes {
		t.Fatalf("test binary is %d bytes, above the tool ceiling", len(data))
	}
	bi, err := buildinfo.ReadFile(exe)
	if err != nil {
		t.Fatalf("test binary buildinfo: %v", err)
	}
	sum := sha256.Sum256(data)
	return exe, goTraceAdmission{
		ToolSHA256:    hex.EncodeToString(sum[:]),
		ToolBytes:     int64(len(data)),
		ToolPath:      bi.Path,
		ToolGoVersion: bi.GoVersion,
		ToolGOOS:      buildSetting(bi, "GOOS"),
		ToolGOARCH:    buildSetting(bi, "GOARCH"),
		Producer:      goTraceTestProducer,
	}
}

// A goTraceFDCloseCase is one copyGoTraceTool path: the candidate record,
// where ctx is cancelled ("", "buildinfo" or "chunk"), whether the private
// copy is created, and the primary error (nil for success).
type goTraceFDCloseCase struct {
	name     string
	record   goTraceAdmission
	cancelAt string
	copied   bool
	primary  error
}

func goTraceFDCloseCases(admitted goTraceAdmission) []goTraceFDCloseCase {
	unadmitted := admitted
	unadmitted.ToolPath = "example.com/not-the-test-binary"
	mismatched := admitted
	mismatched.ToolSHA256 = strings.Repeat("0", 64)
	return []goTraceFDCloseCase{
		{"admitted", admitted, "", true, nil},
		{"buildinfo unadmitted", unadmitted, "", false, ErrGoTraceUnadmitted},
		{"cancel after buildinfo", admitted, "buildinfo", false, context.Canceled},
		{"cancel at copy chunk", admitted, "chunk", true, context.Canceled},
		{"sha mismatch", mismatched, "", true, ErrGoTraceUnadmitted},
	}
}

// A goTraceFDCloseResult is one observed call: the results, the closes per
// descriptor name and per descriptor, and whether the private copy exists
// afterwards.
type goTraceFDCloseResult struct {
	rec    goTraceAdmission
	bi     *buildinfo.BuildInfo
	err    error
	closes map[string]int
	files  map[*os.File]int
	copied bool
}

func newGoTraceFDCloseResult() *goTraceFDCloseResult {
	return &goTraceFDCloseResult{closes: make(map[string]int), files: make(map[*os.File]int)}
}

// goTraceFDCloseHook closes f, counts the close, and returns an error
// wrapping errGoTraceFDCloseInjected when name is in fail.
func goTraceFDCloseHook(res *goTraceFDCloseResult, fail []string) func(string, *os.File) error {
	return func(name string, f *os.File) error {
		err := f.Close()
		res.closes[name]++
		res.files[f]++
		for _, n := range fail {
			if n == name {
				return fmt.Errorf("%s: %w", name, errGoTraceFDCloseInjected)
			}
		}
		return err
	}
}

// goTraceFDCloseCopy calls copyGoTraceTool once for c, failing the closes
// named in fail.
func goTraceFDCloseCopy(t *testing.T, tool string, c goTraceFDCloseCase, fail ...string) *goTraceFDCloseResult {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	res := newGoTraceFDCloseResult()
	h := &goTraceHooks{closeFile: goTraceFDCloseHook(res, fail)}
	switch c.cancelAt {
	case "buildinfo":
		h.afterBuildinfo = cancel
	case "chunk":
		h.copyChunk = cancel
	}
	exe := filepath.Join(t.TempDir(), "trace")
	res.rec, res.bi, res.err = copyGoTraceTool(ctx, tool, exe, []goTraceAdmission{c.record}, goTraceCeilings, h)
	_, err := os.Lstat(exe)
	res.copied = err == nil
	return res
}

// goTraceFDClosePrimaryFirst checks that err is a join whose first error is
// the primary one and whose later errors carry the injected close error.
func goTraceFDClosePrimaryFirst(t *testing.T, err, primary error) {
	t.Helper()
	j, ok := err.(interface{ Unwrap() []error })
	if !ok {
		t.Fatalf("err %v is not a join", err)
	}
	errs := j.Unwrap()
	if len(errs) < 2 || !errors.Is(errs[0], primary) || errors.Is(errs[0], errGoTraceFDCloseInjected) || !errors.Is(errors.Join(errs[1:]...), errGoTraceFDCloseInjected) {
		t.Errorf("join %v: want %v first, then the close error", errs, primary)
	}
}

// FD1: on every path the tool descriptor is closed exactly once, and the
// private copy is closed exactly once when it was created.
func TestGoTraceFDCloseCount(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, admitted := goTraceFDCloseTool(t)
	for _, c := range goTraceFDCloseCases(admitted) {
		t.Run(c.name, func(t *testing.T) {
			res := goTraceFDCloseCopy(t, tool, c)
			if c.primary == nil {
				if res.err != nil || res.rec != c.record || res.bi == nil {
					t.Fatalf("copyGoTraceTool = %+v, buildinfo %v, %v", res.rec, res.bi != nil, res.err)
				}
			} else if !errors.Is(res.err, c.primary) {
				t.Fatalf("err = %v, want %v", res.err, c.primary)
			}
			if res.copied != c.copied {
				t.Errorf("copy created = %v, want %v", res.copied, c.copied)
			}
			want := map[string]int{"tool": 1}
			if c.copied {
				want["copy"] = 1
			}
			if !reflect.DeepEqual(res.closes, want) {
				t.Errorf("closes = %v, want %v", res.closes, want)
			}
			if len(res.files) != len(want) {
				t.Errorf("%d descriptors closed, want %d", len(res.files), len(want))
			}
			for f, n := range res.files {
				if n != 1 {
					t.Errorf("%s closed %d times", f.Name(), n)
				}
			}
		})
	}
}

// FD2: a tool close error turns success into a custody failure with a zero
// record and nil buildinfo; on an early return it is joined after the
// primary error.
func TestGoTraceFDCloseToolError(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, admitted := goTraceFDCloseTool(t)
	for _, c := range goTraceFDCloseCases(admitted) {
		t.Run(c.name, func(t *testing.T) {
			res := goTraceFDCloseCopy(t, tool, c, "tool")
			if res.rec != (goTraceAdmission{}) || res.bi != nil {
				t.Errorf("result %+v, buildinfo %v, want zero", res.rec, res.bi != nil)
			}
			if !errors.Is(res.err, ErrGoTraceCustody) || !errors.Is(res.err, errGoTraceFDCloseInjected) || !strings.Contains(fmt.Sprint(res.err), "close tool") {
				t.Fatalf("err = %v, want the tool close error", res.err)
			}
			if c.primary == nil {
				if errors.Is(res.err, ErrGoTraceUnadmitted) {
					t.Errorf("err = %v, want custody only", res.err)
				}
				return
			}
			goTraceFDClosePrimaryFirst(t, res.err, c.primary)
		})
	}
}

// FD3: a copy close error is reported even when the copy already failed,
// and the copy is still closed once.
func TestGoTraceFDCloseCopyError(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, admitted := goTraceFDCloseTool(t)
	var ran int
	for _, c := range goTraceFDCloseCases(admitted) {
		if !c.copied {
			continue
		}
		ran++
		t.Run(c.name, func(t *testing.T) {
			res := goTraceFDCloseCopy(t, tool, c, "copy")
			if res.closes["copy"] != 1 || res.rec != (goTraceAdmission{}) || res.bi != nil {
				t.Errorf("copy closes %d, result %+v", res.closes["copy"], res.rec)
			}
			if !errors.Is(res.err, ErrGoTraceCustody) || !errors.Is(res.err, errGoTraceFDCloseInjected) || !strings.Contains(fmt.Sprint(res.err), "close copy") {
				t.Fatalf("err = %v, want the copy close error", res.err)
			}
			if c.cancelAt == "chunk" {
				goTraceFDClosePrimaryFirst(t, res.err, context.Canceled)
			}
		})
	}
	if ran != 3 {
		t.Fatalf("ran %d copy cases, want 3", ran)
	}
}

// FD4: writeGoTraceInput closes the input once and reports both its write
// and its close error.
func TestGoTraceFDCloseInputError(t *testing.T) {
	goTraceFDCloseNoStart(t)
	data := goTraceTestInput(64)
	for _, tt := range []struct {
		name       string
		writeFails bool
		closeFails bool
	}{
		{"write and close", true, true},
		{"close only", false, true},
		{"write only", true, false},
		{"clean", false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res := newGoTraceFDCloseResult()
			var fail []string
			if tt.closeFails {
				fail = []string{"input"}
			}
			h := &goTraceHooks{closeFile: goTraceFDCloseHook(res, fail)}
			if tt.writeFails {
				h.writeInput = func(*os.File, []byte) (int, error) { return 0, errGoTraceFDCloseWrite }
			}
			path := filepath.Join(t.TempDir(), "runtime.trace")
			err := writeGoTraceInput(path, data, h)
			if res.closes["input"] != 1 || len(res.files) != 1 {
				t.Errorf("closes = %v over %d descriptors, want input once", res.closes, len(res.files))
			}
			if errors.Is(err, errGoTraceFDCloseWrite) != tt.writeFails || errors.Is(err, errGoTraceFDCloseInjected) != tt.closeFails {
				t.Fatalf("err = %v, want write error %v, close error %v", err, tt.writeFails, tt.closeFails)
			}
			if err != nil {
				if !errors.Is(err, ErrGoTraceCustody) {
					t.Errorf("err = %v, want custody", err)
				}
				return
			}
			if got, rerr := os.ReadFile(path); rerr != nil || !bytes.Equal(got, data) {
				t.Errorf("input = %d bytes, %v", len(got), rerr)
			}
		})
	}
}

// FD5: a joined close error keeps the primary error's class. Every sentinel
// the primary-only error matches is still matched and no other is gained
// except ErrGoTraceCustody; in particular ErrGoTraceMalformed, the only
// class the recorder bridge maps to rejected, is never introduced.
func TestGoTraceFDCloseClass(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, admitted := goTraceFDCloseTool(t)
	type pair struct {
		name    string
		primary error
		joined  error
	}
	var pairs []pair
	for _, c := range goTraceFDCloseCases(admitted) {
		if c.primary == nil {
			continue
		}
		fail := "tool"
		if c.cancelAt == "chunk" {
			fail = "copy"
		}
		pairs = append(pairs, pair{c.name, goTraceFDCloseCopy(t, tool, c).err, goTraceFDCloseCopy(t, tool, c, fail).err})
	}
	input := func(fail ...string) error {
		h := &goTraceHooks{
			closeFile:  goTraceFDCloseHook(newGoTraceFDCloseResult(), fail),
			writeInput: func(*os.File, []byte) (int, error) { return 0, errGoTraceFDCloseWrite },
		}
		return writeGoTraceInput(filepath.Join(t.TempDir(), "runtime.trace"), goTraceTestInput(64), h)
	}
	pairs = append(pairs, pair{"input write", input(), input("input")})
	if len(pairs) != 5 {
		t.Fatalf("%d pairs, want 5", len(pairs))
	}
	sentinels := []error{
		ErrGoTraceDecoderUnselected, ErrGoTraceScratch, ErrGoTraceLimits, ErrGoTraceLimit,
		ErrGoTraceHeader, ErrGoTraceUnadmitted, ErrGoTraceStart, ErrGoTraceSignal,
		ErrGoTraceReap, ErrGoTraceDecoderFailed, ErrGoTraceMalformed, ErrGoTraceCleanup,
		context.Canceled, context.DeadlineExceeded,
	}
	for _, p := range pairs {
		if p.primary == nil || !errors.Is(p.joined, errGoTraceFDCloseInjected) || !errors.Is(p.joined, ErrGoTraceCustody) {
			t.Errorf("%s: precondition: primary %v; joined %v must carry the custody close error", p.name, p.primary, p.joined)
			continue
		}
		for _, s := range sentinels {
			if j, q := errors.Is(p.joined, s), errors.Is(p.primary, s); j != q {
				t.Errorf("%s: errors.Is(err, %v) = %v with the close error, %v without", p.name, s, j, q)
			}
		}
	}
}

// Custody and context controls (UP). Like the FD controls they use a copy of
// the test binary as Tool and a synthetic in-test admission; nothing is
// executed. Every decode path ends in an input-write hook that fails, so even
// a mutant that ignores cancellation stops before runGoTraceTool.

var errGoTraceBarrier = errors.New("input write barrier")

// goTraceCustodyTool copies the test binary to a private file and returns its
// path and the synthetic record that admits those bytes.
func goTraceCustodyTool(t *testing.T) (string, goTraceAdmission) {
	t.Helper()
	exe, rec := goTraceFDCloseTool(t)
	data, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	tool := filepath.Join(t.TempDir(), "trace")
	goTraceMust(t, os.WriteFile(tool, data, 0o700))
	return tool, rec
}

// goTraceCustodyRecord is goTraceCustodyTool with the record completed so
// that it is valid in a table and matches the synthetic header-26 input and
// the test producer.
func goTraceCustodyRecord(t *testing.T) (string, goTraceAdmission) {
	t.Helper()
	tool, rec := goTraceCustodyTool(t)
	rec.Header = 26
	rec.RecipeSHA256 = strings.Repeat("b", 64)
	rec.Receipt = strings.Repeat("c", 64)
	rec.Review = strings.Repeat("d", 64)
	return tool, rec
}

// goTraceCustodyTable returns a valid one-record table holding rec.
func goTraceCustodyTable(t *testing.T, rec goTraceAdmission) []goTraceAdmission {
	t.Helper()
	table := []goTraceAdmission{withID(rec)}
	if err := validateGoTraceAdmissions(table, encodeGoTraceAdmission); err != nil {
		t.Fatalf("synthetic record: %v", err)
	}
	return table
}

// C1: caller cancellation reaches the tool copy through decode and decodeIn.
// Cancelling after buildinfo stops before the copy starts; cancelling at the
// first chunk stops the copy there. Neither reaches the post-copy hook or the
// input write.
func TestGoTraceContextPropagation(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, rec := goTraceCustodyRecord(t)
	table := goTraceCustodyTable(t, rec)
	for _, at := range []string{"buildinfo", "chunk"} {
		t.Run(at, func(t *testing.T) {
			scratch := t.TempDir()
			goTraceMust(t, os.Chmod(scratch, 0o700))
			d := &GoTraceDecoder{Tool: tool, Scratch: scratch}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var chunks, copied, written int
			h := &goTraceHooks{
				copyChunk: func() {
					chunks++
					if at == "chunk" {
						cancel()
					}
				},
				afterCopy: func(string) { copied++ },
				writeInput: func(*os.File, []byte) (int, error) {
					written++
					return 0, errGoTraceBarrier
				},
			}
			if at == "buildinfo" {
				h.afterBuildinfo = cancel
			}
			r, _, err := d.decode(ctx, goTraceTestInput(64), goTraceTestProducer, table, false, h)
			if r != nil || !errors.Is(err, context.Canceled) || errors.Is(err, errGoTraceBarrier) {
				t.Fatalf("decode = %v, %v; want context.Canceled", r, err)
			}
			wantChunks := 0
			if at == "chunk" {
				wantChunks = 1
			}
			if chunks != wantChunks || copied != 0 || written != 0 {
				t.Errorf("chunks %d (want %d), post-copy %d, input writes %d; want the copy to stop at cancellation", chunks, wantChunks, copied, written)
			}
			if names := goTraceListing(t, scratch); len(names) != 0 {
				t.Errorf("scratch holds %v", names)
			}
		})
	}
}

// C2: a change to Tool during the copy is refused as a custody failure: the
// descriptor's size, mtime or mode changes, or the path is replaced by
// identical bytes in a new inode, or removed.
func TestGoTraceToolChangedDuringCopy(t *testing.T) {
	goTraceFDCloseNoStart(t)
	for _, tt := range []struct {
		name   string
		change func(t *testing.T, tool string)
		cause  error
	}{
		{"size", func(t *testing.T, tool string) {
			// Restore the mtime so only the size differs.
			st, err := os.Stat(tool)
			goTraceMust(t, err)
			f, err := os.OpenFile(tool, os.O_WRONLY|os.O_APPEND, 0)
			goTraceMust(t, err)
			_, err = f.Write([]byte{0})
			goTraceMust(t, errors.Join(err, f.Close()))
			goTraceMust(t, os.Chtimes(tool, st.ModTime(), st.ModTime()))
		}, nil},
		{"mtime", func(t *testing.T, tool string) {
			st, err := os.Stat(tool)
			goTraceMust(t, err)
			old := st.ModTime().Add(-time.Hour)
			goTraceMust(t, os.Chtimes(tool, old, old))
		}, nil},
		{"mode", func(t *testing.T, tool string) {
			goTraceMust(t, os.Chmod(tool, 0o500))
		}, nil},
		{"replaced inode", func(t *testing.T, tool string) {
			data, err := os.ReadFile(tool)
			goTraceMust(t, err)
			next := tool + ".next"
			goTraceMust(t, os.WriteFile(next, data, 0o700))
			goTraceMust(t, os.Rename(next, tool))
		}, nil},
		{"removed", func(t *testing.T, tool string) {
			goTraceMust(t, os.Remove(tool))
		}, fs.ErrNotExist},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tool, rec := goTraceCustodyTool(t)
			var chunks int
			h := &goTraceHooks{copyChunk: func() {
				if chunks++; chunks == 1 {
					tt.change(t, tool)
				}
			}}
			exe := filepath.Join(t.TempDir(), "trace")
			got, bi, err := copyGoTraceTool(context.Background(), tool, exe, []goTraceAdmission{rec}, goTraceCeilings, h)
			if got != (goTraceAdmission{}) || bi != nil || !errors.Is(err, ErrGoTraceCustody) {
				t.Fatalf("copyGoTraceTool = %+v, buildinfo %v, %v; want custody failure", got, bi != nil, err)
			}
			if chunks < 2 {
				t.Errorf("copy ran %d chunks; the change must happen mid-copy", chunks)
			}
			if tt.cause != nil && !errors.Is(err, tt.cause) {
				t.Errorf("err = %v, want cause %v", err, tt.cause)
			}
		})
	}
}

// C3: Tool replaced or removed between Lstat and Open is refused.
func TestGoTraceToolChangedWhileOpening(t *testing.T) {
	goTraceFDCloseNoStart(t)
	for _, tt := range []struct {
		name  string
		swap  func(t *testing.T, tool string)
		cause error
	}{
		{"replaced inode", func(t *testing.T, tool string) {
			data, err := os.ReadFile(tool)
			goTraceMust(t, err)
			next := tool + ".next"
			goTraceMust(t, os.WriteFile(next, data, 0o700))
			goTraceMust(t, os.Rename(next, tool))
		}, nil},
		{"removed", func(t *testing.T, tool string) {
			goTraceMust(t, os.Remove(tool))
		}, fs.ErrNotExist},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tool, rec := goTraceCustodyTool(t)
			h := &goTraceHooks{afterLstat: func() { tt.swap(t, tool) }}
			exe := filepath.Join(t.TempDir(), "trace")
			got, bi, err := copyGoTraceTool(context.Background(), tool, exe, []goTraceAdmission{rec}, goTraceCeilings, h)
			if got != (goTraceAdmission{}) || bi != nil || !errors.Is(err, ErrGoTraceCustody) {
				t.Fatalf("copyGoTraceTool = %+v, buildinfo %v, %v; want custody failure", got, bi != nil, err)
			}
			if tt.cause != nil && !errors.Is(err, tt.cause) {
				t.Errorf("err = %v, want cause %v", err, tt.cause)
			}
			if _, err := os.Lstat(exe); err == nil {
				t.Errorf("private copy %s created", exe)
			}
		})
	}
}

// C4: a short write with no error is a custody failure, and the input is
// still closed once.
func TestGoTraceInputShortWrite(t *testing.T) {
	goTraceFDCloseNoStart(t)
	res := newGoTraceFDCloseResult()
	h := &goTraceHooks{
		closeFile:  goTraceFDCloseHook(res, nil),
		writeInput: func(f *os.File, b []byte) (int, error) { return f.Write(b[:len(b)-1]) },
	}
	err := writeGoTraceInput(filepath.Join(t.TempDir(), "runtime.trace"), goTraceTestInput(64), h)
	if !errors.Is(err, ErrGoTraceCustody) || !errors.Is(err, io.ErrShortWrite) {
		t.Errorf("err = %v, want custody short write", err)
	}
	if res.closes["input"] != 1 || len(res.files) != 1 {
		t.Errorf("closes = %v, want input once", res.closes)
	}
}

// C5: other copy faults. A tool above Limits.ToolBytes, an existing private
// copy, and a cancelled copy whose copy and tool closes both fail: the
// cancellation stays first and both close errors are kept.
func TestGoTraceCopyFaults(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, rec := goTraceCustodyTool(t)
	t.Run("tool over limit", func(t *testing.T) {
		l := goTraceCeilings
		l.ToolBytes = rec.ToolBytes - 1
		exe := filepath.Join(t.TempDir(), "trace")
		got, _, err := copyGoTraceTool(context.Background(), tool, exe, []goTraceAdmission{rec}, l, new(goTraceHooks))
		if got != (goTraceAdmission{}) || !errors.Is(err, ErrGoTraceLimit) {
			t.Errorf("err = %v, want limit", err)
		}
		if _, err := os.Lstat(exe); err == nil {
			t.Errorf("private copy %s created", exe)
		}
	})
	t.Run("copy exists", func(t *testing.T) {
		exe := filepath.Join(t.TempDir(), "trace")
		goTraceMust(t, os.WriteFile(exe, []byte("x"), 0o700))
		got, _, err := copyGoTraceTool(context.Background(), tool, exe, []goTraceAdmission{rec}, goTraceCeilings, new(goTraceHooks))
		if got != (goTraceAdmission{}) || !errors.Is(err, ErrGoTraceCustody) || !errors.Is(err, fs.ErrExist) {
			t.Errorf("err = %v, want custody exist", err)
		}
		if data, err := os.ReadFile(exe); err != nil || string(data) != "x" {
			t.Errorf("existing copy = %d bytes, %v", len(data), err)
		}
	})
	t.Run("cancel with both closes failing", func(t *testing.T) {
		res := goTraceFDCloseCopy(t, tool, goTraceFDCloseCase{"chunk", rec, "chunk", true, context.Canceled}, "copy", "tool")
		if res.rec != (goTraceAdmission{}) || res.closes["copy"] != 1 || res.closes["tool"] != 1 {
			t.Errorf("result %+v, closes %v", res.rec, res.closes)
		}
		first := res.err
		for {
			j, ok := first.(interface{ Unwrap() []error })
			if !ok {
				break
			}
			first = j.Unwrap()[0]
		}
		if !errors.Is(first, context.Canceled) || errors.Is(first, errGoTraceFDCloseInjected) {
			t.Errorf("first joined error = %v, want the cancellation", first)
		}
		msg := fmt.Sprint(res.err)
		if !strings.Contains(msg, "close copy") || !strings.Contains(msg, "close tool") {
			t.Errorf("err = %v, want both close errors", res.err)
		}
	})
}

// C6: the private copy is re-hashed at rest. A change to it after the copy
// is closed is refused before the input is written.
func TestGoTraceCopyTampered(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, rec := goTraceCustodyRecord(t)
	table := goTraceCustodyTable(t, rec)
	for _, tt := range []struct {
		name   string
		tamper func(t *testing.T, exe string)
	}{
		{"byte flipped", func(t *testing.T, exe string) {
			f, err := os.OpenFile(exe, os.O_RDWR, 0)
			goTraceMust(t, err)
			b := make([]byte, 1)
			off := rec.ToolBytes / 2
			_, err = f.ReadAt(b, off)
			goTraceMust(t, err)
			b[0] ^= 0xff
			_, err = f.WriteAt(b, off)
			goTraceMust(t, errors.Join(err, f.Close()))
		}},
		{"appended", func(t *testing.T, exe string) {
			f, err := os.OpenFile(exe, os.O_WRONLY|os.O_APPEND, 0)
			goTraceMust(t, err)
			_, err = f.Write([]byte{0})
			goTraceMust(t, errors.Join(err, f.Close()))
		}},
		{"mode", func(t *testing.T, exe string) {
			goTraceMust(t, os.Chmod(exe, 0o755))
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			scratch := t.TempDir()
			goTraceMust(t, os.Chmod(scratch, 0o700))
			d := &GoTraceDecoder{Tool: tool, Scratch: scratch}
			var copied, written int
			h := &goTraceHooks{
				afterCopy: func(exe string) {
					copied++
					tt.tamper(t, exe)
				},
				writeInput: func(*os.File, []byte) (int, error) {
					written++
					return 0, errGoTraceBarrier
				},
			}
			r, _, err := d.decode(context.Background(), goTraceTestInput(64), goTraceTestProducer, table, false, h)
			if r != nil || !errors.Is(err, ErrGoTraceCustody) || errors.Is(err, errGoTraceBarrier) {
				t.Fatalf("decode = %v, %v; want custody failure before the input write", r, err)
			}
			if copied != 1 || written != 0 {
				t.Errorf("post-copy %d, input writes %d; want 1, 0", copied, written)
			}
			if names := goTraceListing(t, scratch); len(names) != 0 {
				t.Errorf("scratch holds %v", names)
			}
		})
	}
}

// A8: a valid, non-empty table admits nothing when no record matches the
// input header and producer exactly. Each record would admit Tool if the
// tuple were ignored; the refusal comes before Tool is opened.
func TestGoTraceAdmissionTupleMismatch(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, rec := goTraceCustodyRecord(t)
	for _, tt := range []struct {
		name   string
		change func(r *goTraceAdmission)
	}{
		{"header", func(r *goTraceAdmission) { r.Header = 25 }},
		{"producer go version", func(r *goTraceAdmission) { r.Producer.GoVersion = "go1.27.1" }},
		{"producer goos", func(r *goTraceAdmission) { r.Producer.GOOS = "linux" }},
		{"producer goarch", func(r *goTraceAdmission) { r.Producer.GOARCH = "amd64" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := rec
			tt.change(&r)
			table := goTraceCustodyTable(t, r)
			scratch := t.TempDir()
			goTraceMust(t, os.Chmod(scratch, 0o700))
			d := &GoTraceDecoder{Tool: tool, Scratch: scratch}
			var chunks, written int
			h := &goTraceHooks{
				copyChunk: func() { chunks++ },
				writeInput: func(*os.File, []byte) (int, error) {
					written++
					return 0, errGoTraceBarrier
				},
			}
			got, _, err := d.decode(context.Background(), goTraceTestInput(64), goTraceTestProducer, table, false, h)
			if got != nil || !errors.Is(err, ErrGoTraceUnadmitted) || errors.Is(err, ErrGoTraceCustody) {
				t.Fatalf("decode = %v, %v; want unadmitted", got, err)
			}
			if chunks != 0 || written != 0 {
				t.Errorf("copy chunks %d, input writes %d; want 0, 0", chunks, written)
			}
			if names := goTraceListing(t, scratch); len(names) != 0 {
				t.Errorf("scratch holds %v", names)
			}
		})
	}
}

// C7: Tool replaced mid-copy by a symlink to the original file is refused:
// the path no longer names a regular file, though it resolves to the same
// inode.
func TestGoTraceToolSymlinkedDuringCopy(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, rec := goTraceCustodyTool(t)
	var chunks int
	h := &goTraceHooks{copyChunk: func() {
		if chunks++; chunks == 1 {
			orig := tool + ".orig"
			goTraceMust(t, os.Rename(tool, orig))
			goTraceMust(t, os.Symlink(orig, tool))
		}
	}}
	exe := filepath.Join(t.TempDir(), "trace")
	got, bi, err := copyGoTraceTool(context.Background(), tool, exe, []goTraceAdmission{rec}, goTraceCeilings, h)
	if got != (goTraceAdmission{}) || bi != nil || !errors.Is(err, ErrGoTraceCustody) {
		t.Fatalf("copyGoTraceTool = %+v, buildinfo %v, %v; want custody failure", got, bi != nil, err)
	}
	if chunks < 2 {
		t.Errorf("copy ran %d chunks; the swap must happen mid-copy", chunks)
	}
	if !strings.Contains(err.Error(), "changed during copy") {
		t.Errorf("err = %v, want changed during copy", err)
	}
}

// C8: cancellation observed at a chunk boundary stops the copy before that
// chunk is read: the private copy holds only the chunks before it.
func TestGoTraceCopyCancelStopsRead(t *testing.T) {
	goTraceFDCloseNoStart(t)
	tool, rec := goTraceCustodyTool(t)
	if rec.ToolBytes <= 1<<20 {
		t.Fatalf("tool is %d bytes; need more than one chunk", rec.ToolBytes)
	}
	for _, at := range []int{1, 2} {
		t.Run(fmt.Sprint("chunk ", at), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var chunks int
			h := &goTraceHooks{copyChunk: func() {
				if chunks++; chunks == at {
					cancel()
				}
			}}
			exe := filepath.Join(t.TempDir(), "trace")
			got, _, err := copyGoTraceTool(ctx, tool, exe, []goTraceAdmission{rec}, goTraceCeilings, h)
			if got != (goTraceAdmission{}) || !errors.Is(err, context.Canceled) {
				t.Fatalf("copyGoTraceTool = %+v, %v; want context.Canceled", got, err)
			}
			st, err := os.Lstat(exe)
			goTraceMust(t, err)
			if want := int64(at-1) << 20; chunks != at || st.Size() != want {
				t.Errorf("chunks %d, copy %d bytes; want %d, %d", chunks, st.Size(), at, want)
			}
		})
	}
}
