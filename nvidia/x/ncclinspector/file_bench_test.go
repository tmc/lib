package ncclinspector

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// statsRecords returns n valid stats records. With vary, pid digit counts
// differ, so record lengths differ.
func statsRecords(t testing.TB, n int, vary bool) [][]byte {
	b := fixture(t, "stats")
	out := make([][]byte, n)
	for i := range out {
		pid := 1639453
		if vary {
			pid = 1 + i*i*i*7919%100000000
		}
		out[i] = bytes.Replace(b, []byte(`"pid":1639453`), []byte(`"pid":`+strconv.Itoa(pid)), 1)
	}
	return out
}

func writeRecords(t testing.TB, records [][]byte) string {
	path := filepath.Join(t.TempDir(), "capture")
	if err := os.WriteFile(path, bytes.Join(records, nil), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestFileHashWork checks the exact number of bytes written to the prefix
// hash. A successful Next costs the committed prefix plus the new record; a
// Next that returns io.EOF, Position, Finish and resume each cost the prefix.
func TestFileHashWork(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	for _, n := range []int{1, 2, 7, 33} {
		for _, vary := range []bool{false, true} {
			t.Run(fmt.Sprintf("n=%d/vary=%v", n, vary), func(t *testing.T) {
				records := statsRecords(t, n, vary)
				d, err := OpenFile(writeRecords(t, records), Limits{}, nil)
				if err != nil {
					t.Fatal(err)
				}
				defer d.Close()
				var prefix, want int64
				for k, rec := range records {
					before := d.hashed
					r, err := d.Next()
					if err != nil || !bytes.Equal(r.Raw, rec) {
						t.Fatalf("record %d: %v", k, err)
					}
					cost := prefix + int64(len(rec))
					if got := d.hashed - before; got != cost {
						t.Fatalf("Next %d hashed %d, want %d", k, got, cost)
					}
					prefix += int64(len(rec))
					want += cost
				}
				if d.hashed != want {
					t.Fatalf("successful Next total %d, want %d", d.hashed, want)
				}
				if _, err = d.Next(); err != io.EOF {
					t.Fatal(err)
				}
				want += prefix
				if d.hashed != want {
					t.Fatalf("total with terminal EOF %d, want %d", d.hashed, want)
				}
				if !vary {
					// Equal records: L*N*(N+1)/2 for the records plus L*N for EOF.
					l, nn := int64(len(records[0])), int64(n)
					if formula := l*nn*(nn+1)/2 + l*nn; d.hashed != formula {
						t.Fatalf("total %d, formula %d", d.hashed, formula)
					}
				}
				before := d.hashed
				if _, err = d.Position(); err != nil || d.hashed-before != prefix {
					t.Fatalf("Position hashed %d, want %d: %v", d.hashed-before, prefix, err)
				}
				before = d.hashed
				if err = d.Finish(); err != nil || d.hashed-before != prefix {
					t.Fatalf("Finish hashed %d, want %d: %v", d.hashed-before, prefix, err)
				}
			})
		}
	}
}

func TestFileHashWorkAppendAndResume(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	records := statsRecords(t, 3, true)
	l0, l1, l2 := int64(len(records[0])), int64(len(records[1])), int64(len(records[2]))
	path := writeRecords(t, [][]byte{records[0], records[1][:10]})
	d, err := OpenFile(path, Limits{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	step := func(name string, want int64, wantErr error) {
		t.Helper()
		before := d.hashed
		if _, err := d.Next(); err != wantErr {
			t.Fatalf("%s: %v, want %v", name, err, wantErr)
		}
		if got := d.hashed - before; got != want {
			t.Fatalf("%s hashed %d, want %d", name, got, want)
		}
	}
	step("first", l0, nil)
	step("pending", l0, io.EOF)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.Write(append(records[1][10:], records[2]...)); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	step("completed", l0+l1, nil)
	pos, err := d.Position()
	if err != nil {
		t.Fatal(err)
	}
	if err = d.Close(); err != nil {
		t.Fatal(err)
	}
	d, err = OpenFile(path, Limits{}, &pos)
	if err != nil {
		t.Fatal(err)
	}
	if d.hashed != l0+l1 {
		t.Fatalf("resume hashed %d, want %d", d.hashed, l0+l1)
	}
	step("resumed", l0+l1+l2, nil)
	step("terminal", l0+l1+l2, io.EOF)
	if err = d.Finish(); err != nil {
		t.Fatal(err)
	}
}

// BenchmarkFileDecoderNext reads n records and one terminal EOF per
// iteration. It reports bytes hashed per record; timings are host-dependent.
func BenchmarkFileDecoderNext(b *testing.B) {
	if !fileSupported() {
		b.Skip("file adapter unsupported")
	}
	for _, n := range []int{250, 1000} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			path := writeRecords(b, statsRecords(b, n, false))
			var hashed int64
			for b.Loop() {
				d, err := OpenFile(path, Limits{}, nil)
				if err != nil {
					b.Fatal(err)
				}
				for {
					if _, err = d.Next(); err == io.EOF {
						break
					} else if err != nil {
						b.Fatal(err)
					}
				}
				hashed = d.hashed
				d.Close()
			}
			b.ReportMetric(float64(hashed)/float64(n), "hashed-bytes/record")
		})
	}
}
