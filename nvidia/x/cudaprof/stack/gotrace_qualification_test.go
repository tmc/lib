package stack

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

// Stage-2 controls of the admitted C6D1 tuple through the public Validate
// and Analyze. They run the admitted decoder on the authentic stage-1 trace,
// read from the files named by CUDAPROF_GO_TRACE_DECODER and
// CUDAPROF_GO_TRACE_INPUT and checked against fixed digests; no trace byte
// is kept in the source. They skip when either is unset, or fail when
// CUDAPROF_GO_TRACE_REQUIRED=1.

// The authentic trace: one public cudaprof.StartHostProfile run of a
// go1.27.2 darwin/arm64 test binary, retained by the stage-1 producer.
const (
	goTraceAuthenticSHA256 = "05a86dab1b8a7242fb1e04f55fd2f41535d73e6ea36a423b3b27791767164e90"
	goTraceAuthenticBytes  = 22370
)

// goTraceAuthentic returns the admitted decoder's path and the authentic
// trace, after checking both against their pinned digests.
func goTraceAuthentic(t *testing.T) (string, []byte) {
	t.Helper()
	tool, input := os.Getenv("CUDAPROF_GO_TRACE_DECODER"), os.Getenv("CUDAPROF_GO_TRACE_INPUT")
	if tool == "" || input == "" {
		if os.Getenv("CUDAPROF_GO_TRACE_REQUIRED") == "1" {
			t.Fatal("CUDAPROF_GO_TRACE_DECODER and CUDAPROF_GO_TRACE_INPUT are required")
		}
		t.Skip("CUDAPROF_GO_TRACE_DECODER or CUDAPROF_GO_TRACE_INPUT not set")
	}
	rec := goTraceC6D1
	exe, err := os.ReadFile(tool)
	goTraceMust(t, err)
	if sum := sha256.Sum256(exe); hex.EncodeToString(sum[:]) != rec.ToolSHA256 || int64(len(exe)) != rec.ToolBytes {
		t.Fatalf("decoder %s: %d bytes, sha256 %x; want the admitted decoder", tool, len(exe), sum)
	}
	data, err := os.ReadFile(input)
	goTraceMust(t, err)
	if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != goTraceAuthenticSHA256 || len(data) != goTraceAuthenticBytes {
		t.Fatalf("trace %s: %d bytes, sha256 %x; want the authentic trace", input, len(data), sum)
	}
	return tool, data
}

// goTraceAuthenticDecoder returns a public decoder over tool with a fresh
// 0700 scratch that must be empty again when the test ends.
func goTraceAuthenticDecoder(t *testing.T, tool string) *GoTraceDecoder {
	t.Helper()
	s := t.TempDir()
	goTraceMust(t, os.Chmod(s, 0o700))
	t.Cleanup(func() {
		if names := goTraceListing(t, s); len(names) != 0 {
			t.Errorf("scratch holds %v", names)
		}
	})
	return &GoTraceDecoder{Tool: tool, Scratch: s}
}

// goTraceAuthenticIdentity is a stand-in process identity; the stack
// package never needs the producer's own.
var goTraceAuthenticIdentity = Record{Process: "c6d1-stage2", PID: 1}

var goTraceClasses = []struct {
	name string
	err  error
}{
	{"unselected", ErrGoTraceDecoderUnselected}, {"scratch", ErrGoTraceScratch},
	{"limits", ErrGoTraceLimits}, {"limit", ErrGoTraceLimit}, {"header", ErrGoTraceHeader},
	{"unadmitted", ErrGoTraceUnadmitted}, {"custody", ErrGoTraceCustody}, {"start", ErrGoTraceStart},
	{"signal", ErrGoTraceSignal}, {"reap", ErrGoTraceReap}, {"failed", ErrGoTraceDecoderFailed},
	{"malformed", ErrGoTraceMalformed}, {"unproven", ErrGoTraceUnproven}, {"cleanup", ErrGoTraceCleanup},
	{"canceled", context.Canceled},
}

// goTraceClass lists every sentinel err matches.
func goTraceClass(err error) []string {
	var c []string
	for _, k := range goTraceClasses {
		if errors.Is(err, k.err) {
			c = append(c, k.name)
		}
	}
	return c
}

// goTraceBatchEnds walks the v2 framing of data and returns the end offset
// of each batch up to and including the first end-of-generation marker
// (EvEventBatch 1, EvExperimentalBatch 49, EvEndOfGeneration 52).
func goTraceBatchEnds(t *testing.T, data []byte) []int {
	t.Helper()
	var ends []int
	off := 16
	for off < len(data) {
		switch data[off] {
		case 52:
			return append(ends, off+1)
		case 1, 49:
			p := off + 1
			if data[off] == 49 {
				p++
			}
			var size uint64
			for range 4 { // generation, M, timestamp, size
				v, n := binary.Uvarint(data[p:])
				if n <= 0 {
					t.Fatalf("bad batch header varint at %d", p)
				}
				size, p = v, p+n
			}
			off = p + int(size)
			ends = append(ends, off)
		default:
			t.Fatalf("unexpected batch type %d at %d", data[off], off)
		}
	}
	t.Fatal("no end-of-generation marker")
	return nil
}

// goTraceCheckReceipt checks a successful receipt against the admitted
// record and the input. Completeness is unknown for every decode.
func goTraceCheckReceipt(t *testing.T, r *GoTraceDecode, data []byte) {
	t.Helper()
	rec := goTraceAdmitted(t)
	sum := sha256.Sum256(data)
	want := GoTraceDecode{
		Admission:     rec.ID,
		ToolSHA256:    rec.ToolSHA256,
		ToolBytes:     rec.ToolBytes,
		ToolPath:      "cmd/trace",
		ToolGoVersion: "go1.27.2",
		ToolGOOS:      "darwin",
		ToolGOARCH:    "arm64",
		RecipeSHA256:  rec.RecipeSHA256,
		Receipt:       rec.Receipt,
		Review:        rec.Review,
		Header:        26,
		Producer:      rec.Producer,
		InputSHA256:   hex.EncodeToString(sum[:]),
		InputBytes:    int64(len(data)),
		OutputBytes:   r.OutputBytes,
		Completeness:  goTraceCompletenessUnknown,
		Cleanup:       "removed",
	}
	if *r != want || r.OutputBytes <= 0 {
		t.Errorf("receipt %+v, want %+v", *r, want)
	}
}

// S2P: the authentic trace through the public entry points. Decoding is
// format evidence only: the clock stays unavailable and the trace's
// completeness unknown.
func TestGoTraceQualificationPositive(t *testing.T) {
	tool, data := goTraceAuthentic(t)
	before := goTraceStarts.Load()
	r, err := goTraceAuthenticDecoder(t, tool).Validate(context.Background(), data, goTraceC6D1.Producer)
	if err != nil || r == nil {
		t.Fatalf("Validate = %v, %v", r, err)
	}
	goTraceCheckReceipt(t, r, data)

	g, err := goTraceAuthenticDecoder(t, tool).Analyze(context.Background(), data, goTraceC6D1.Producer, goTraceAuthenticIdentity, Limits{})
	if err != nil || g == nil {
		t.Fatalf("Analyze = %v, %v", g, err)
	}
	if n := goTraceStarts.Load() - before; n != 2 {
		t.Errorf("started %d children, want 2", n)
	}
	goTraceCheckReceipt(t, g.Decoder, data)
	if *g.Decoder != *r || g.InputSHA256 != r.InputSHA256 || g.Backend != "cmd/trace go1.27.2 admission "+r.Admission {
		t.Errorf("Analyze receipt %+v backend %q, Validate receipt %+v", *g.Decoder, g.Backend, *r)
	}
	t.Logf("admission %s: %d output bytes, %d samples, %d clock snapshots, completeness %q", r.Admission, r.OutputBytes, len(g.Samples), len(g.ClockSnapshots), r.Completeness)
	if len(g.Samples) != 116 || len(g.ClockSnapshots) != 2 {
		t.Errorf("samples %d, clock snapshots %d; want 116, 2", len(g.Samples), len(g.ClockSnapshots))
	}
	for i, s := range g.Samples {
		if s.Clock != "go_trace" || s.Status != "clock_unavailable" || s.Provenance != "cpu_sample" || len(s.Frames) == 0 {
			t.Errorf("sample %d: %+v", i, s)
		}
	}
	// Format is not clock or device placement: the Darwin go1.27.2 producer
	// stays clock unavailable.
	if err := QualifyGoMonotonic(g, "darwin", "go1.27.2"); err == nil || err.Error() != "unsupported Go trace producer clock" || g.ClockStatus != "unavailable" {
		t.Errorf("QualifyGoMonotonic = %v, status %s", err, g.ClockStatus)
	}
}

// S2C (M1): cuts and corruption of the authentic trace through the public
// entry points, which must agree. A header-only or pre-event cut decodes
// to nothing and is unproven. A cut at the generation boundary decodes
// with fewer samples and the same unknown completeness as the whole trace:
// the decoder cannot tell it is short. A mid-payload cut stays decoder
// failed and a framing diagnostic on the closed list stays malformed.
func TestGoTraceQualificationCuts(t *testing.T) {
	tool, data := goTraceAuthentic(t)
	ends := goTraceBatchEnds(t, data)
	eog := ends[len(ends)-1]
	if ends[0] != 69 || eog != 13936 {
		t.Fatalf("first batch ends at %d, generation 1 at %d; want 69, 13936", ends[0], eog)
	}
	flip := bytes.Clone(data)
	flip[16] = 0xff
	p := goTraceC6D1.Producer
	for _, tt := range []struct {
		name  string
		data  []byte
		class []string
		diag  string
	}{
		{"whole trace", data, nil, ""},
		{"header only", data[:16], []string{"unproven"}, "no event other than Sync"},
		{"cut after first batch", data[:ends[0]], []string{"unproven"}, "no event other than Sync"},
		{"cut at generation boundary", data[:eog], nil, ""},
		{"cut inside first batch payload", data[:ends[0]-1], []string{"failed"}, "failed to read full batch"},
		{"cut inside last batch payload", data[:eog-2], []string{"failed"}, "failed to read full batch"},
		{"cut before end of generation", data[:eog-1], []string{"malformed"}, "incomplete generation found; trace likely truncated"},
		{"batch type byte", flip, []string{"malformed"}, "expected batch event, got event 255"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			before := goTraceStarts.Load()
			r, verr := goTraceAuthenticDecoder(t, tool).Validate(context.Background(), tt.data, p)
			g, aerr := goTraceAuthenticDecoder(t, tool).Analyze(context.Background(), tt.data, p, goTraceAuthenticIdentity, Limits{})
			if n := goTraceStarts.Load() - before; n != 2 {
				t.Errorf("started %d children, want 2", n)
			}
			t.Logf("%d bytes: Validate %v (%v), Analyze %v", len(tt.data), goTraceClass(verr), verr, goTraceClass(aerr))
			if tt.class != nil {
				for api, err := range map[string]error{"Validate": verr, "Analyze": aerr} {
					if !reflect.DeepEqual(goTraceClass(err), tt.class) || !strings.Contains(err.Error(), tt.diag) {
						t.Errorf("%s = %v, want class %v with %q", api, err, tt.class, tt.diag)
					}
				}
				if r != nil || g != nil {
					t.Errorf("results %v, %v; want none", r, g)
				}
				return
			}
			if verr != nil || aerr != nil || r == nil || g == nil {
				t.Fatalf("Validate = %v, %v; Analyze = %v, %v", r, verr, g, aerr)
			}
			goTraceCheckReceipt(t, r, tt.data)
			goTraceCheckReceipt(t, g.Decoder, tt.data)
			if !strings.HasPrefix(r.Completeness, "unknown: ") {
				t.Errorf("completeness %q", r.Completeness)
			}
			t.Logf("decoded %d samples, %d clock snapshots", len(g.Samples), len(g.ClockSnapshots))
			if len(tt.data) < len(data) && (len(g.Samples) == 0 || len(g.Samples) >= 116 || len(g.ClockSnapshots) == 0) {
				t.Errorf("generation cut: %d samples, %d clock snapshots; want some, fewer than the whole trace", len(g.Samples), len(g.ClockSnapshots))
			}
		})
	}
}
