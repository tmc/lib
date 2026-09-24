package cudatest_test

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

// recorder is a testing.TB that records failures instead of failing the
// test, so that the harness can be run against planted defects.
type recorder struct {
	testing.TB
	msgs   []string
	failed bool
}

func (r *recorder) Helper() {}

func (r *recorder) Errorf(format string, args ...any) {
	r.failed = true
	r.msgs = append(r.msgs, fmt.Sprintf(format, args...))
}

func (r *recorder) Fatalf(format string, args ...any) {
	r.Errorf(format, args...)
	runtime.Goexit()
}

// run calls f with a recorder on its own goroutine, so that Fatalf can stop
// f as it stops a test.
func run(f func(t testing.TB)) *recorder {
	r := new(recorder)
	done := make(chan struct{})
	go func() {
		defer close(done)
		f(r)
	}()
	<-done
	return r
}

type pair struct {
	Key string
	N   int
}

// pairDecoder parses "key=n". Its encoder is correct unless drop is set.
func pairDecoder(drop bool) cudatest.Decoder[pair] {
	return cudatest.Decoder[pair]{
		Decode: func(b []byte) (pair, error) {
			k, v, ok := strings.Cut(string(b), "=")
			if !ok {
				return pair{}, errors.New("no =")
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				return pair{}, err
			}
			return pair{k, n}, nil
		},
		Encode: func(p pair) ([]byte, error) {
			if drop {
				return []byte(p.Key + "=0"), nil
			}
			return []byte(p.Key + "=" + strconv.Itoa(p.N)), nil
		},
		Invariants: []func(pair) error{func(p pair) error {
			if p.N < 0 {
				return errors.New("negative")
			}
			return nil
		}},
	}
}

func TestDecoderCheck(t *testing.T) {
	zero := cudatest.Decoder[*pair]{Decode: func([]byte) (*pair, error) { return nil, nil }}
	tests := []struct {
		name string
		f    func(t testing.TB)
		want string // substring of the failure, or "" for a pass
	}{
		{"good", func(t testing.TB) { pairDecoder(false).Check(t, []byte("a=1")) }, ""},
		{"rejected input", func(t testing.TB) { pairDecoder(false).Check(t, []byte("a")) }, ""},
		{"zero value", func(t testing.TB) { zero.Check(t, []byte("x")) }, "zero value"},
		{"invariant", func(t testing.TB) { pairDecoder(false).Check(t, []byte("a=-1")) }, "invariant 0"},
		{"lossy encoder", func(t testing.TB) { pairDecoder(true).Check(t, []byte("a=1")) }, "changed the value"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := run(tt.f)
			switch {
			case tt.want == "" && r.failed:
				t.Errorf("failed: %q", r.msgs)
			case tt.want != "" && !strings.Contains(strings.Join(r.msgs, "\n"), tt.want):
				t.Errorf("failures %q, want one containing %q", r.msgs, tt.want)
			}
		})
	}
}

func TestDecoderCheckMutant(t *testing.T) {
	// ignoreSecond reads only the first digit of n, so a mutant of the
	// second digit decodes to the seed's value.
	ignoreSecond := pairDecoder(false)
	decode := ignoreSecond.Decode
	ignoreSecond.Decode = func(b []byte) (pair, error) {
		if i := strings.IndexByte(string(b), '='); i >= 0 && len(b) > i+2 {
			b = b[:i+2]
		}
		return decode(b)
	}
	seed := []byte("a=12")
	tests := []struct {
		name string
		d    cudatest.Decoder[pair]
		i    int
		b    byte
		want string
	}{
		{"detected", pairDecoder(false), 3, '3', ""},
		{"rejected", pairDecoder(false), 1, ':', ""},
		{"absorbed", ignoreSecond, 3, '3', "decoded to the seed's value"},
		{"no mutation", pairDecoder(false), 3, '2', "nothing mutated"},
		{"out of range", pairDecoder(false), 4, '0', "out of range"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := run(func(t testing.TB) { tt.d.CheckMutant(t, seed, tt.i, tt.b) })
			switch {
			case tt.want == "" && r.failed:
				t.Errorf("failed: %q", r.msgs)
			case tt.want != "" && !strings.Contains(strings.Join(r.msgs, "\n"), tt.want):
				t.Errorf("failures %q, want one containing %q", r.msgs, tt.want)
			}
		})
	}
}

func TestCheckSeedCoverage(t *testing.T) {
	dir := t.TempDir()
	for _, f := range []string{"a.ptx", "sub/b.ptx", "gen.sh", "fuzz/FuzzX/0123"} {
		p := filepath.Join(dir, f)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	a, b, gen := filepath.Join(dir, "a.ptx"), filepath.Join(dir, "sub/b.ptx"), filepath.Join(dir, "gen.sh")
	tests := []struct {
		name     string
		seeded   []string
		excluded map[string]string
		want     string
	}{
		{"covered", []string{a, b}, map[string]string{gen: "generator"}, ""},
		{"file in a subdirectory unseeded", []string{a}, map[string]string{gen: "generator"}, "sub/b.ptx is neither"},
		{"no reason", []string{a, b}, map[string]string{gen: ""}, "gives no reason"},
		{"stale exclusion", []string{a, b}, map[string]string{gen: "generator", filepath.Join(dir, "gone"): "x"}, "names no fixture"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := run(func(t testing.TB) { cudatest.CheckSeedCoverage(t, dir, tt.seeded, tt.excluded) })
			switch {
			case tt.want == "" && r.failed:
				t.Errorf("failed: %q", r.msgs)
			case tt.want != "" && !strings.Contains(strings.Join(r.msgs, "\n"), tt.want):
				t.Errorf("failures %q, want one containing %q", r.msgs, tt.want)
			}
		})
	}
}

func TestReadSeedsRefusesEmptyPattern(t *testing.T) {
	pattern := filepath.Join(t.TempDir(), "*.ptx")
	r := run(func(t testing.TB) { cudatest.ReadSeeds(t, pattern) })
	if !r.failed {
		t.Error("ReadSeeds accepted a pattern matching no file")
	}
}
