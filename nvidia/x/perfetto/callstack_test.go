package perfetto

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestCallstack(t *testing.T) {
	var b bytes.Buffer
	w := NewWriter(&b)
	w.SequenceID = 91
	fs := []StackFrame{{Function: "leaf", File: "work.go", Line: 7}, {Function: "root", File: "main.go", Line: 2}}
	id, err := w.InternCallstack(fs)
	if err != nil {
		t.Fatal(err)
	}
	n := b.Len()
	again, err := w.InternCallstack(fs)
	if err != nil || again != id || n != b.Len() {
		t.Fatal("not interned")
	}
	if err := w.WriteTrackDescriptor(1, "work", 0); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteSliceBeginWithCallstack(1, 10, "submission", "go", nil, id); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteSliceEnd(1, 20); err != nil {
		t.Fatal(err)
	}
	tp := os.Getenv("CUDAPROF_TEST_TRACE_PROCESSOR")
	if tp == "" {
		return
	}
	dir := t.TempDir()
	trace := filepath.Join(dir, "stack.pftrace")
	if err := os.WriteFile(trace, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	sql := filepath.Join(dir, "query.sql")
	os.WriteFile(sql, []byte("select depth,name from stack_profile_callsite join stack_profile_frame on frame_id=stack_profile_frame.id order by depth;"), 0600)
	out, err := exec.Command(tp, "-q", sql, trace).CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	if !strings.Contains(string(out), "root") || !strings.Contains(string(out), "leaf") {
		t.Fatalf("missing stack: %s", out)
	}
}

func TestCallstackReference(t *testing.T) {
	for _, instant := range []bool{false, true} {
		t.Run(fmt.Sprint(instant), func(t *testing.T) {
			var b bytes.Buffer
			w := NewWriter(&b)
			write := func(id uint64) error {
				if instant {
					return w.WriteInstantWithCallstack(1, 1, "sample", "go", nil, id)
				}
				return w.WriteSliceBeginWithCallstack(1, 1, "launch", "go", nil, id)
			}
			if err := write(1); err == nil || b.Len() != 0 {
				t.Fatal("unknown reference wrote trace bytes")
			}
			id, err := w.InternCallstack([]StackFrame{{Function: "leaf"}})
			if err != nil {
				t.Fatal(err)
			}
			n := b.Len()
			if err := write(id + 1); err == nil || b.Len() != n {
				t.Fatal("unknown reference wrote trace bytes")
			}
			w.SequenceID = 99
			if err := write(id); err == nil || b.Len() != n {
				t.Fatal("changed sequence wrote trace bytes")
			}
			if _, err := w.InternCallstack([]StackFrame{{Function: "other"}}); err == nil || b.Len() != n {
				t.Fatal("changed sequence interned symbols")
			}
		})
	}
}

type failedStackWriter struct{ writes int }

func (w *failedStackWriter) Write(p []byte) (int, error) { w.writes++; return len(p) - 1, nil }
func TestCallstackWriteFailure(t *testing.T) {
	out := new(failedStackWriter)
	w := NewWriter(out)
	fs := []StackFrame{{Function: "leaf"}}
	if _, err := w.InternCallstack(fs); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("got %v", err)
	}
	if _, err := w.InternCallstack(fs); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("got %v", err)
	}
	if err := w.WriteInstantWithCallstack(1, 1, "sample", "go", nil, 1); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("got %v", err)
	}
	if out.writes != 1 {
		t.Fatalf("retried failed stream: %d writes", out.writes)
	}
}

func TestCallstackWriterSequences(t *testing.T) {
	var b bytes.Buffer
	first := NewWriter(&b)
	first.SequenceID = 101
	id, err := first.InternCallstack([]StackFrame{{Function: "first"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := first.WriteInstantWithCallstack(1, 1, "first", "go", nil, id); err != nil {
		t.Fatal(err)
	}
	second := NewWriter(&b)
	second.SequenceID = 102
	id, err = second.InternCallstack([]StackFrame{{Function: "second"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := second.WriteInstantWithCallstack(2, 2, "second", "go", nil, id); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b.Bytes(), []byte("first")) || !bytes.Contains(b.Bytes(), []byte("second")) {
		t.Fatal("writer dictionaries reused across sequences")
	}
}
