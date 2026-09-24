package cudatest

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sort"
	"testing"
)

// A Decoder describes a function that turns untrusted bytes into a typed
// value, so that fuzz targets in different packages check their parsers the
// same way.
//
// Decode must not panic on any input. The harness does not recover a panic:
// the fuzz engine reports it with the input that caused it, and that report is
// the finding.
type Decoder[T any] struct {
	// Decode parses data.
	Decode func(data []byte) (T, error)

	// Encode re-encodes a decoded value. It is nil when the package has no
	// encoder; a test must not supply one written only for the test.
	Encode func(v T) ([]byte, error)

	// Equal reports whether two decoded values are the same value. Nil means
	// reflect.DeepEqual. Equality is on the typed value, not on bytes: an
	// encoder that normalizes whitespace is correct.
	Equal func(a, b T) bool

	// Invariants are properties every successfully decoded value has. Each
	// must be able to fail on some input; one that holds by construction
	// tests nothing.
	Invariants []func(v T) error
}

func (d Decoder[T]) equal(a, b T) bool {
	if d.Equal != nil {
		return d.Equal(a, b)
	}
	return reflect.DeepEqual(a, b)
}

// Check decodes data and fails t for each property that does not hold: a nil
// error together with a zero value, a failed invariant, and, when Encode is
// set, a value that is not reproduced by Encode followed by a second Decode.
// It returns the decoded value and whether Decode succeeded.
func (d Decoder[T]) Check(t testing.TB, data []byte) (T, bool) {
	t.Helper()
	v, err := d.Decode(data)
	if err != nil {
		return v, false
	}
	if reflect.ValueOf(&v).Elem().IsZero() {
		t.Errorf("decode %s: nil error with a zero value", quote(data))
		return v, true
	}
	for i, inv := range d.Invariants {
		if err := inv(v); err != nil {
			t.Errorf("decode %s: invariant %d: %v", quote(data), i, err)
		}
	}
	if d.Encode == nil {
		return v, true
	}
	enc, err := d.Encode(v)
	if err != nil {
		t.Errorf("decode %s: encode decoded value: %v", quote(data), err)
		return v, true
	}
	w, err := d.Decode(enc)
	if err != nil {
		t.Errorf("decode %s: re-decode of %s: %v", quote(data), quote(enc), err)
		return v, true
	}
	if !d.equal(v, w) {
		t.Errorf("decode %s: round trip through %s changed the value\nfirst:  %s\nsecond: %s",
			quote(data), quote(enc), show(v), show(w))
	}
	return v, true
}

// CheckMutant decodes seed and a copy of seed with byte i replaced by b, and
// fails t if the copy decodes without error to a value equal to the seed's: a
// parser that accepts a changed input and returns the original value cannot
// be told from a correct one by any test that sees only valid input.
//
// It also fails t if seed does not decode, if i is out of range, or if seed[i]
// is already b, so that a control cannot pass by mutating nothing. The caller
// chooses i among bytes the decoded value retains; a byte of a comment is
// rightly absorbed.
func (d Decoder[T]) CheckMutant(t testing.TB, seed []byte, i int, b byte) {
	t.Helper()
	if i < 0 || i >= len(seed) {
		t.Errorf("mutant at %d: out of range for a %d-byte seed", i, len(seed))
		return
	}
	if seed[i] == b {
		t.Errorf("mutant at %d: seed already holds %q, nothing mutated", i, b)
		return
	}
	v, err := d.Decode(seed)
	if err != nil {
		t.Errorf("mutant at %d: seed does not decode: %v", i, err)
		return
	}
	m := bytes.Clone(seed)
	m[i] = b
	w, err := d.Decode(m)
	if err != nil {
		return
	}
	if d.equal(v, w) {
		t.Errorf("mutant at %d (%q -> %q, in %s): accepted and decoded to the seed's value",
			i, seed[i], b, quote(around(seed, i)))
	}
}

// ReadSeeds returns the contents of the files matching the glob patterns,
// keyed by path. It fails t if a pattern matches no file, so that a renamed
// fixture cannot silently empty a corpus.
func ReadSeeds(t testing.TB, patterns ...string) map[string][]byte {
	t.Helper()
	seeds := map[string][]byte{}
	for _, p := range patterns {
		matches, err := filepath.Glob(p)
		if err != nil {
			t.Fatalf("seed pattern %q: %v", p, err)
		}
		if len(matches) == 0 {
			t.Fatalf("seed pattern %q matches no file", p)
		}
		for _, m := range matches {
			data, err := os.ReadFile(m)
			if err != nil {
				t.Fatalf("read seed: %v", err)
			}
			seeds[m] = data
		}
	}
	return seeds
}

// SortedPaths returns the keys of seeds in order, so that a fuzz target adds
// its seed corpus in the same order every run.
func SortedPaths(seeds map[string][]byte) []string {
	paths := make([]string, 0, len(seeds))
	for p := range seeds {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}

// CheckSeedCoverage fails t unless every file under dir is either in seeded
// or in excluded. Files under dir/fuzz are the fuzz engine's own corpus and
// are skipped. Every key of excluded must be a file that exists and must have
// a reason, so that the exclusions cannot outlive the fixtures they excuse.
//
// Paths are as filepath.Join(dir, ...) spells them.
func CheckSeedCoverage(t testing.TB, dir string, seeded []string, excluded map[string]string) {
	t.Helper()
	fuzzDir := filepath.Join(dir, "fuzz")
	var files []string
	err := filepath.WalkDir(dir, func(path string, e fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if e.IsDir() {
			if path == fuzzDir {
				return filepath.SkipDir
			}
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", dir, err)
	}
	if len(files) == 0 {
		t.Fatalf("%s holds no fixtures", dir)
	}
	for _, f := range files {
		if slices.Contains(seeded, f) {
			continue
		}
		if _, ok := excluded[f]; !ok {
			t.Errorf("fixture %s is neither a seed nor excluded with a reason", f)
		}
	}
	for f, why := range excluded {
		if why == "" {
			t.Errorf("exclusion of %s gives no reason", f)
		}
		if !slices.Contains(files, f) {
			t.Errorf("exclusion of %s names no fixture", f)
		}
	}
}

// quote formats input for a failure message, truncated so that a large seed
// does not bury the report.
func quote(b []byte) string {
	const limit = 200
	if len(b) > limit {
		return fmt.Sprintf("%q... (%d bytes)", b[:limit], len(b))
	}
	return fmt.Sprintf("%q", b)
}

func show(v any) string {
	s := fmt.Sprintf("%+v", v)
	if len(s) > 2000 {
		s = s[:2000] + "..."
	}
	return s
}

// around returns the bytes around seed[i].
func around(seed []byte, i int) []byte {
	lo, hi := max(0, i-20), min(len(seed), i+20)
	return seed[lo:hi]
}
