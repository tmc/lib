// Package journal stores capture records as an append-only wire-frame journal.
// A journal can be inspected after an interrupted writer and published without
// replacing an existing recording.
package journal

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// Options controls journal durability and ingestion bounds.
type Options struct {
	Limits   wire.Limits
	SyncEach bool
}

// Journal is safe for concurrent Append calls. Append is bounded by Limits;
// it copies no more than the supplied record payload into the kernel writer.
type Journal struct {
	mu           sync.Mutex
	file         *os.File
	path         string
	session      [16]byte
	options      Options
	count, bytes uint64
	closed       bool
}

// Create opens path as a new journal. Existing files are never overwritten.
func Create(path string, session [16]byte, options Options) (*Journal, error) {
	if path == "" {
		return nil, errors.New("journal: empty path")
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, fmt.Errorf("create journal: %w", err)
	}
	return &Journal{file: f, path: path, session: session, options: options}, nil
}

// Append adds one record. A record from another session is rejected before
// touching the journal. SyncEach makes every successful append durable.
func (j *Journal) Append(r wire.Record) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed {
		return io.ErrClosedPipe
	}
	if r.Session != j.session {
		return wire.ErrSession
	}
	b, err := r.MarshalBinary(j.options.Limits)
	if err != nil {
		return err
	}
	if err = writeAll(j.file, b); err != nil {
		return fmt.Errorf("append journal: %w", err)
	}
	if j.options.SyncEach {
		if err = j.file.Sync(); err != nil {
			return fmt.Errorf("sync journal: %w", err)
		}
	}
	j.count++
	j.bytes += uint64(len(b))
	return nil
}

func writeAll(w io.Writer, p []byte) error {
	for len(p) > 0 {
		n, err := w.Write(p)
		if n > 0 {
			p = p[n:]
		}
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

// Close flushes and closes the journal. It may be called more than once.
func (j *Journal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.closed {
		return nil
	}
	j.closed = true
	if err := j.file.Sync(); err != nil {
		_ = j.file.Close()
		return fmt.Errorf("close journal: %w", err)
	}
	if err := j.file.Close(); err != nil {
		return fmt.Errorf("close journal: %w", err)
	}
	return nil
}

// Finalization describes the journal immediately before publication.
type Finalization struct {
	JournalPath    string
	Destination    string
	Session        [16]byte
	Records, Bytes uint64
}
type FinalizeHook func(Finalization) error

// Finalize closes the journal, runs hook, and atomically publishes it at dst.
// The destination is created with a link operation, so an existing recording
// is never replaced. If the hook or publication fails, the source is retained.
func (j *Journal) Finalize(dst string, hook FinalizeHook) error {
	j.mu.Lock()
	if j.closed {
		j.mu.Unlock()
		return io.ErrClosedPipe
	}
	if err := j.file.Sync(); err != nil {
		j.mu.Unlock()
		return fmt.Errorf("finalize journal: %w", err)
	}
	if err := j.file.Close(); err != nil {
		j.mu.Unlock()
		return fmt.Errorf("finalize journal: %w", err)
	}
	j.closed = true
	f := Finalization{JournalPath: j.path, Destination: dst, Session: j.session, Records: j.count, Bytes: j.bytes}
	j.mu.Unlock()
	if hook != nil {
		if err := hook(f); err != nil {
			return fmt.Errorf("finalize hook: %w", err)
		}
	}
	if err := os.Link(j.path, dst); err != nil {
		return fmt.Errorf("publish journal: %w", err)
	}
	if err := os.Remove(j.path); err != nil {
		return fmt.Errorf("remove journal: %w", err)
	}
	return nil
}

// Recovery is the valid prefix and its terminal condition.
type Recovery struct {
	Records             []wire.Record
	Stats               wire.Stats
	Complete, Truncated bool
}

// Recover decodes a journal without executing or loading any captured code.
// A truncated final frame is recoverable and is reported with Truncated=true.
// Other malformed input returns the valid prefix plus an error.
func Recover(path string, limits wire.Limits) (Recovery, error) {
	f, err := os.Open(path)
	if err != nil {
		return Recovery{}, fmt.Errorf("open journal: %w", err)
	}
	defer f.Close()
	d := wire.NewDecoder(f, limits)
	var out Recovery
	first := true
	for {
		r, e := d.Next()
		if e == nil {
			if first {
				d.ExpectSession(r.Session)
				first = false
			}
			out.Records = append(out.Records, r)
			continue
		}
		out.Stats = d.Stats()
		switch {
		case errors.Is(e, io.EOF):
			out.Complete = true
			return out, nil
		case errors.Is(e, io.ErrUnexpectedEOF):
			out.Truncated = true
			return out, nil
		default:
			return out, fmt.Errorf("recover journal: %w", e)
		}
	}
}
