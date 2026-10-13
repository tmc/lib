package cudaflightrecorder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

// goTraceValidateTrace has the exact header layout of a Go 1.26 trace. It is
// not a decodable trace; validators in these tests never decode it.
var goTraceValidateTrace = []byte("go 1.26 trace\x00\x00\x00\x01\x02\x03")

func goTraceValidateBundle(data []byte) *Bundle {
	b := NewBundle(Manifest{Reason: "test", GoTraceBytes: int64(len(data)), ClockUncertaintyNs: -1, ProviderStatus: map[string]string{"go.trace": "retained"}, LossReasons: []string{"none"}})
	b.Files["go.trace"] = data
	b.Files["gpu.pprof"] = []byte("other")
	return b
}

func goTraceValidateSnapshot(b *Bundle) Bundle {
	m := b.Manifest
	m.ProviderStatus = maps.Clone(m.ProviderStatus)
	m.LossReasons = slices.Clone(m.LossReasons)
	files := make(map[string][]byte, len(b.Files))
	for name, data := range b.Files {
		files[name] = bytes.Clone(data)
	}
	return Bundle{Manifest: m, Files: files}
}

// goTraceValidateCall runs ValidateGoTrace and checks the invariants every row
// shares: the bundle is unchanged and a non-nil result carries the authority.
func goTraceValidateCall(t *testing.T, ctx context.Context, b *Bundle, v GoTraceValidator) (*GoTraceValidation, error) {
	t.Helper()
	before := goTraceValidateSnapshot(b)
	r, err := ValidateGoTrace(ctx, b, v)
	goTraceValidateCheck(t, before, b, r, err)
	return r, err
}

func goTraceValidateCheck(t *testing.T, before Bundle, b *Bundle, r *GoTraceValidation, err error) {
	t.Helper()
	if !reflect.DeepEqual(before, goTraceValidateSnapshot(b)) {
		t.Errorf("bundle modified")
	}
	if r == nil {
		return
	}
	if r.Authority != GoTraceValidatorAuthority {
		t.Errorf("Authority = %q", r.Authority)
	}
	if (err == nil) != (r.Outcome == "decoded") {
		t.Errorf("Outcome %q with error %v", r.Outcome, err)
	}
	if !strings.HasPrefix(r.Status, r.Outcome+": ") {
		t.Errorf("Status %q does not name Outcome %q", r.Status, r.Outcome)
	}
}

func goTraceValidateSHA(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// goTraceValidateBridge is the caller bridge from the exact stack adapter.
func goTraceValidateBridge(dec *stack.GoTraceDecoder, p stack.GoProducer) GoTraceValidator {
	return func(ctx context.Context, data []byte) (string, error) {
		r, err := dec.Validate(ctx, data, p)
		if errors.Is(err, stack.ErrGoTraceMalformed) {
			return "", fmt.Errorf("%w: %w", ErrArtifactInvalid, err)
		}
		if err != nil {
			return "", err
		}
		return "cmd/trace admission " + r.Admission, nil
	}
}

func goTraceValidateOK(context.Context, []byte) (string, error) { return "test decoder", nil }

// R1.
func TestValidateGoTraceSelection(t *testing.T) {
	b := goTraceValidateBundle(goTraceValidateTrace)
	r, err := goTraceValidateCall(t, context.Background(), b, nil)
	if r != nil || !errors.Is(err, ErrGoTraceValidatorUnselected) {
		t.Errorf("nil validator: %+v, %v", r, err)
	}
	r, err = ValidateGoTrace(context.Background(), nil, goTraceValidateOK)
	if r != nil || !errors.Is(err, ErrGoTraceNotRetained) {
		t.Errorf("nil bundle: %+v, %v", r, err)
	}
	empty := NewBundle(Manifest{})
	r, err = goTraceValidateCall(t, context.Background(), empty, goTraceValidateOK)
	if r != nil || !errors.Is(err, ErrGoTraceNotRetained) {
		t.Errorf("missing go.trace: %+v, %v", r, err)
	}
}

// R2.
func TestValidateGoTraceRecorderRejects(t *testing.T) {
	for _, tt := range []struct {
		name string
		data []byte
		want error
	}{
		{"placeholder", []byte("go_trace_v2\x00"), ErrPlaceholderRejected},
		{"bad magic", []byte("not a trace"), ErrArtifactInvalid},
		{"short", []byte("go"), ErrArtifactInvalid},
	} {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			v := func(context.Context, []byte) (string, error) { called = true; return "x", nil }
			r, err := goTraceValidateCall(t, context.Background(), goTraceValidateBundle(tt.data), v)
			if !errors.Is(err, tt.want) || r == nil || r.Outcome != "rejected" || called {
				t.Errorf("got %+v, %v, called %v", r, err, called)
			}
			if r != nil && (r.InputSHA256 != goTraceValidateSHA(tt.data) || r.InputBytes != int64(len(tt.data))) {
				t.Errorf("input identity %s %d", r.InputSHA256, r.InputBytes)
			}
		})
	}
}

// R3 uses the unexported clone seam to observe that nothing is copied before
// the cancellation and budget checks.
func TestValidateGoTraceBudgetAndCancel(t *testing.T) {
	big := make([]byte, 64<<20)
	copy(big, goTraceValidateTrace)
	over := make([]byte, 64<<20+1)
	copy(over, goTraceValidateTrace)
	for _, tt := range []struct {
		name       string
		data       []byte
		cancel     bool // before the call
		cancelInV  bool
		wantErr    error
		wantOut    string
		wantClones int
		wantCalled bool
	}{
		{"pre-cancelled 64 MiB", big, true, false, context.Canceled, "unavailable", 0, false},
		{"64 MiB plus one", over, false, false, ErrSnapshotBudgetExceeded, "unavailable", 0, false},
		{"64 MiB", big, false, false, nil, "decoded", 1, true},
		{"cancelled by validator", goTraceValidateTrace, false, true, context.Canceled, "unavailable", 1, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tt.cancel {
				cancel()
			}
			clones := 0
			clone := func(p []byte) []byte { clones++; return bytes.Clone(p) }
			called := false
			v := func(context.Context, []byte) (string, error) {
				called = true
				if tt.cancelInV {
					cancel()
				}
				return "test decoder", nil
			}
			b := goTraceValidateBundle(tt.data)
			before := goTraceValidateSnapshot(b)
			r, err := validateGoTraceBundle(ctx, b, v, clone)
			goTraceValidateCheck(t, before, b, r, err)
			if !errors.Is(err, tt.wantErr) || (tt.wantErr == nil) != (err == nil) {
				t.Errorf("err = %v, want %v", err, tt.wantErr)
			}
			if r == nil || r.Outcome != tt.wantOut {
				t.Errorf("result = %+v, want %s", r, tt.wantOut)
			}
			if clones != tt.wantClones || called != tt.wantCalled {
				t.Errorf("clones = %d, called = %v", clones, called)
			}
		})
	}
}

// R3m: an input mutation outranks a cancellation observed after the callback.
func TestValidateGoTraceMutationPrecedence(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	v := func(_ context.Context, data []byte) (string, error) {
		data[0] ^= 0xff
		cancel()
		return "test decoder", nil
	}
	b := goTraceValidateBundle(bytes.Clone(goTraceValidateTrace))
	before := goTraceValidateSnapshot(b)
	r, err := validateGoTraceBundle(ctx, b, v, bytes.Clone)
	goTraceValidateCheck(t, before, b, r, err)
	if !errors.Is(err, ErrGoTraceValidatorResult) || errors.Is(err, context.Canceled) || r == nil || r.Outcome != "unavailable" {
		t.Errorf("got %+v, %v", r, err)
	}
}

// R4.
func TestValidateGoTraceResult(t *testing.T) {
	for _, tt := range []struct {
		name string
		desc string
		ok   bool
	}{
		{"empty", "", false},
		{"257 bytes", strings.Repeat("d", 257), false},
		{"256 bytes", strings.Repeat("d", 256), true},
		{"newline", "a\nb", false},
		{"control", "a\x00b", false},
		{"invalid utf-8", "a\xffb", false},
		{"utf-8", "décodeur", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v := func(context.Context, []byte) (string, error) { return tt.desc, nil }
			r, err := goTraceValidateCall(t, context.Background(), goTraceValidateBundle(goTraceValidateTrace), v)
			if tt.ok {
				if err != nil || r.Outcome != "decoded" || r.Decoder != tt.desc {
					t.Errorf("got %+v, %v", r, err)
				}
				return
			}
			if !errors.Is(err, ErrGoTraceValidatorResult) || r == nil || r.Outcome != "unavailable" {
				t.Errorf("got %+v, %v", r, err)
			}
		})
	}
}

// R5.
func TestValidateGoTraceInputMutation(t *testing.T) {
	data := bytes.Clone(goTraceValidateTrace)
	b := goTraceValidateBundle(data)
	v := func(_ context.Context, p []byte) (string, error) {
		p[len(p)-1]++
		return "test decoder", nil
	}
	r, err := goTraceValidateCall(t, context.Background(), b, v)
	if !errors.Is(err, ErrGoTraceValidatorResult) || r == nil || r.Outcome != "unavailable" {
		t.Errorf("got %+v, %v", r, err)
	}
	if !bytes.Equal(b.Files["go.trace"], goTraceValidateTrace) {
		t.Error("validator reached the bundle bytes")
	}
}

// R6.
func TestValidateGoTraceErrors(t *testing.T) {
	for _, tt := range []struct {
		name    string
		err     error
		outcome string
		status  string
	}{
		{"artifact invalid", fmt.Errorf("decoder says: %w", ErrArtifactInvalid), "rejected", "rejected: caller assertion: decoder says: "},
		{"canceled", fmt.Errorf("decoder: %w", context.Canceled), "unavailable", "unavailable: decoder: context canceled"},
		{"other", errors.New("boom"), "unavailable", "unavailable: boom"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v := func(context.Context, []byte) (string, error) { return "ignored", tt.err }
			r, err := goTraceValidateCall(t, context.Background(), goTraceValidateBundle(goTraceValidateTrace), v)
			if !errors.Is(err, tt.err) || r == nil || r.Outcome != tt.outcome || !strings.HasPrefix(r.Status, tt.status) {
				t.Errorf("got %+v, %v", r, err)
			}
			if r != nil && r.Decoder != "" {
				t.Errorf("Decoder = %q for a failed validation", r.Decoder)
			}
		})
	}
}

// R7.
func TestValidateGoTraceDecoded(t *testing.T) {
	b := goTraceValidateBundle(goTraceValidateTrace)
	r, err := goTraceValidateCall(t, context.Background(), b, goTraceValidateOK)
	if err != nil {
		t.Fatal(err)
	}
	want := GoTraceValidation{
		Outcome:     "decoded",
		Status:      "decoded: caller assertion by test decoder; timing unvalidated",
		Authority:   GoTraceValidatorAuthority,
		Decoder:     "test decoder",
		InputSHA256: goTraceValidateSHA(goTraceValidateTrace),
		InputBytes:  int64(len(goTraceValidateTrace)),
	}
	if *r != want {
		t.Errorf("got %+v, want %+v", *r, want)
	}
}

// R8: only the explicit entry point names a validator; nothing in the
// recorder's snapshot path can reach one.
func TestValidateGoTraceIsolation(t *testing.T) {
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
		if name == "gotrace_validate.go" {
			continue
		}
		for _, word := range []string{"ValidateGoTrace", "GoTraceValidator", "validateGoTraceBundle"} {
			if bytes.Contains(data, []byte(word)) {
				t.Errorf("%s names %s", name, word)
			}
		}
	}
	if checked < 10 {
		t.Fatalf("checked %d non-test files", checked)
	}
}

// R1b and R9b: the bridge over an unselected or header-refusing adapter is
// unavailable, never rejected, and never panics.
func TestValidateGoTraceBridge(t *testing.T) {
	producer := stack.GoProducer{GoVersion: "go1.27.2", GOOS: "darwin", GOARCH: "arm64"}
	n9 := []byte("go 1.27 trace\x00not trace records")
	for _, tt := range []struct {
		name string
		dec  *stack.GoTraceDecoder
		data []byte
		want error
	}{
		{"nil decoder", nil, goTraceValidateTrace, stack.ErrGoTraceDecoderUnselected},
		{"zero decoder", new(stack.GoTraceDecoder), goTraceValidateTrace, stack.ErrGoTraceDecoderUnselected},
		{"historical header", &stack.GoTraceDecoder{Tool: "/nonexistent-c6d1/trace", Scratch: "/nonexistent-c6d1/scratch"}, n9, stack.ErrGoTraceHeader},
		{"unadmitted", &stack.GoTraceDecoder{Tool: "/nonexistent-c6d1/trace", Scratch: "/nonexistent-c6d1/scratch"}, goTraceValidateTrace, stack.ErrGoTraceUnadmitted},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := goTraceValidateCall(t, context.Background(), goTraceValidateBundle(tt.data), goTraceValidateBridge(tt.dec, producer))
			if !errors.Is(err, tt.want) || errors.Is(err, ErrArtifactInvalid) || r == nil || r.Outcome != "unavailable" {
				t.Errorf("got %+v, %v", r, err)
			}
		})
	}
}

// R9 (stage 2) needs an admitted decoder; D1 admits none. The name keeps it
// out of the stage U selection; it never passes.
func TestStage2ValidateGoTraceAdmitted(t *testing.T) {
	if os.Getenv("CUDAPROF_GO_TRACE_REQUIRED") == "1" {
		t.Fatal("stage 2 R9 requires an admitted decoder record; none is admitted")
	}
	t.Skip("stage 2 R9: NOT RUN (no admitted decoder record)")
}

func ExampleValidateGoTrace() {
	b := NewBundle(Manifest{Reason: "example"})
	b.Files["go.trace"] = []byte("go 1.26 trace\x00\x00\x00")
	v := func(ctx context.Context, data []byte) (string, error) {
		return "example decoder", nil
	}
	r, err := ValidateGoTrace(context.Background(), b, v)
	fmt.Println(r.Outcome, err)
	fmt.Println(r.Authority)
	// Output:
	// decoded <nil>
	// caller-selected validator assertion; not independent decoder authority; timing and population unvalidated
}
