package journal

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

func rec(s byte, n uint64) wire.Record {
	var id [16]byte
	id[0] = s
	return wire.Record{Identity: wire.Identity{Session: id, Process: 1, Generation: 1}, Sequence: n, Payload: []byte("x")}
}

func TestRecoverTruncatedAndFinalize(t *testing.T) {
	d := t.TempDir()
	p := filepath.Join(d, "run.journal")
	var s [16]byte
	s[0] = 1
	j, e := Create(p, s, Options{SyncEach: true})
	if e != nil {
		t.Fatal(e)
	}
	if e = j.Append(rec(1, 1)); e != nil {
		t.Fatal(e)
	}
	if e = j.Append(rec(1, 2)); e != nil {
		t.Fatal(e)
	}
	if e := j.Finalize(filepath.Join(d, "run.capture"), func(f Finalization) error {
		if f.Records != 2 {
			return errors.New("bad count")
		}
		return nil
	}); e != nil {
		t.Fatal(e)
	}
	if _, e := os.Stat(p); !os.IsNotExist(e) {
		t.Fatalf("journal remains: %v", e)
	}
	got, e := Recover(filepath.Join(d, "run.capture"), wire.Limits{})
	if e != nil || !got.Complete || len(got.Records) != 2 {
		t.Fatalf("recovery=%+v err=%v", got, e)
	}
}
func TestRecoverPartialPrefix(t *testing.T) {
	d := t.TempDir()
	p := filepath.Join(d, "partial")
	var s [16]byte
	s[0] = 2
	j, _ := Create(p, s, Options{})
	j.Append(rec(2, 1))
	j.Close()
	f, _ := os.OpenFile(p, os.O_WRONLY|os.O_APPEND, 0600)
	f.Write([]byte{1, 2, 3})
	f.Close()
	got, e := Recover(p, wire.Limits{})
	if e != nil || !got.Truncated || len(got.Records) != 1 {
		t.Fatalf("recovery=%+v err=%v", got, e)
	}
}

func TestRecoverRejectsMixedSessions(t *testing.T) {
	d := t.TempDir()
	p := filepath.Join(d, "mixed")
	var s [16]byte
	s[0] = 1
	j, err := Create(p, s, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := j.Append(rec(1, 1)); err != nil {
		t.Fatal(err)
	}
	j.Close()
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if err := wire.Write(f, rec(2, 2), wire.Limits{}); err != nil {
		t.Fatal(err)
	}
	f.Close()
	if _, err := Recover(p, wire.Limits{}); !errors.Is(err, wire.ErrSession) {
		t.Fatalf("mixed sessions: %v", err)
	}
}
func TestSessionAndExistingDestination(t *testing.T) {
	d := t.TempDir()
	p := filepath.Join(d, "j")
	var s [16]byte
	s[0] = 3
	j, _ := Create(p, s, Options{})
	if e := j.Append(rec(4, 1)); !errors.Is(e, wire.ErrSession) {
		t.Fatal(e)
	}
	j.Append(rec(3, 1))
	dst := filepath.Join(d, "dst")
	os.WriteFile(dst, []byte("keep"), 0600)
	if e := j.Finalize(dst, nil); e == nil {
		t.Fatal("expected existing destination error")
	}
	if b, _ := os.ReadFile(dst); string(b) != "keep" {
		t.Fatal("destination replaced")
	}
}

func TestCreateFilesystemError(t *testing.T) {
	d := t.TempDir()
	if _, err := Create(filepath.Join(d, "missing", "journal"), [16]byte{}, Options{}); err == nil {
		t.Fatal("expected create error")
	}
}
func TestConcurrentSessions(t *testing.T) {
	d := t.TempDir()
	var wg sync.WaitGroup
	for i := byte(1); i <= 4; i++ {
		wg.Add(1)
		go func(s byte) {
			defer wg.Done()
			var id [16]byte
			id[0] = s
			j, e := Create(filepath.Join(d, string([]byte{'a' + s})), id, Options{})
			if e != nil {
				t.Error(e)
				return
			}
			for n := uint64(1); n < 20; n++ {
				if e = j.Append(rec(s, n)); e != nil {
					t.Error(e)
				}
			}
			if e = j.Close(); e != nil {
				t.Error(e)
			}
		}(i)
	}
	wg.Wait()
	ents, _ := os.ReadDir(d)
	if len(ents) != 4 {
		t.Fatalf("files=%d", len(ents))
	}
}

func TestConcurrentNamespacedStreamsRecoverWithoutCrossTalk(t *testing.T) {
	d := t.TempDir()
	const streams = 8
	const records = 256
	var wg sync.WaitGroup
	for i := 0; i < streams; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var id [16]byte
			id[0] = byte(i + 1)
			path := filepath.Join(d, "process-"+string(rune('a'+i)), "capture.journal")
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Error(err)
				return
			}
			j, err := Create(path, id, Options{})
			if err != nil {
				t.Error(err)
				return
			}
			for n := uint64(1); n <= records; n++ {
				r := rec(byte(i+1), n)
				r.Identity.Process = uint64(i + 100)
				r.Identity.Generation = uint64(i + 1)
				if err := j.Append(r); err != nil {
					t.Error(err)
					return
				}
			}
			if err := j.Close(); err != nil {
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()
	for i := 0; i < streams; i++ {
		path := filepath.Join(d, "process-"+string(rune('a'+i)), "capture.journal")
		got, err := Recover(path, wire.Limits{})
		if err != nil || !got.Complete || len(got.Records) != records {
			t.Fatalf("stream %d recovery=%+v err=%v", i, got, err)
		}
		for _, r := range got.Records {
			if r.Identity.Process != uint64(i+100) || r.Identity.Generation != uint64(i+1) {
				t.Fatalf("stream %d identity=%+v", i, r.Identity)
			}
		}
	}
}
