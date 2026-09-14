package cudacapture

import (
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

// Journal member names. A capture carries the target's raw per-record journal
// under JournalName and each worker generation's under a
// "worker-process-<pid>-<generation>-" prefix. A journal compressed before
// embedding carries JournalGzipSuffix as well, so the member is
// self-describing: a reader knows from the name alone whether to decompress.
const (
	JournalName       = "capture.journal"
	JournalGzipSuffix = ".gz"
)

// ErrJournalTooLarge reports that a journal's compressed form did not fit the
// caller's limit. The journal is then omitted from the capture rather than
// buffered, so a producer never holds an unbounded artifact in memory.
var ErrJournalTooLarge = errors.New("cudacapture: compressed journal exceeds limit")

// IsJournalName reports whether name is a capture journal member, in either
// the plain or the compressed form.
func IsJournalName(name string) bool {
	return strings.HasSuffix(strings.TrimSuffix(name, JournalGzipSuffix), JournalName)
}

// JournalNames lists the capture journals embedded in c, sorted by name. Names
// are the members as stored, so a compressed journal keeps its suffix.
func (c *Capture) JournalNames() []string {
	var names []string
	for name := range c.Files {
		if IsJournalName(name) {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

// isGzip reports whether data carries the gzip magic number. Detection is by
// content rather than by name so that a journal stored compressed under a
// plain name still opens; every writer in this repo adds JournalGzipSuffix,
// but a reader that trusts the name alone silently hands back gzip bytes.
func isGzip(data []byte) bool {
	return len(data) >= 2 && data[0] == 0x1f && data[1] == 0x8b
}

// Journal returns the raw journal bytes of the named member, decompressing a
// member that was stored compressed. Readers should use it rather than
// indexing Files by name, so captures written in either form open.
func (c *Capture) Journal(name string) ([]byte, error) {
	data, ok := c.Files[name]
	if !ok {
		return nil, fmt.Errorf("cudacapture: no capture journal %q", name)
	}
	// A member is compressed if its name says so or its bytes say so. The two
	// tests catch different faults: the name alone misses gzip bytes stored
	// under a plain name, and the bytes alone silently pass a .gz member whose
	// content is not gzip, which is corruption rather than a plain journal.
	named := strings.HasSuffix(name, JournalGzipSuffix)
	if !named && !isGzip(data) {
		return data, nil
	}
	if named && !isGzip(data) {
		return nil, fmt.Errorf("cudacapture: %s is named as compressed but does not carry the gzip magic number", name)
	}
	zr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read %s: %w", name, err)
	}
	defer zr.Close()
	// The decompressed size is not bounded by the embedded size, so cap it at
	// the same bound the package applies to any single decoded input.
	out, err := io.ReadAll(io.LimitReader(zr, MaxCaptureInputBytes+1))
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read %s: %w", name, err)
	}
	if int64(len(out)) > MaxCaptureInputBytes {
		return nil, fmt.Errorf("cudacapture: journal %q exceeds %d bytes decompressed", name, int64(MaxCaptureInputBytes))
	}
	return out, nil
}

// CompressJournal returns the gzip-compressed contents of r. It stops as soon
// as the compressed form exceeds limit bytes and reports ErrJournalTooLarge,
// so the caller never buffers more than the budget it is willing to embed. A
// limit of zero or less means MaxCaptureFileBytes.
func CompressJournal(r io.Reader, limit int64) ([]byte, error) {
	if limit <= 0 {
		limit = MaxCaptureFileBytes
	}
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	chunk := make([]byte, 1<<20)
	for {
		n, err := r.Read(chunk)
		if n > 0 {
			if _, werr := zw.Write(chunk[:n]); werr != nil {
				return nil, fmt.Errorf("cudacapture: compress journal: %w", werr)
			}
			if int64(buf.Len()) > limit {
				return nil, ErrJournalTooLarge
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("cudacapture: compress journal: %w", err)
		}
	}
	if err := zw.Close(); err != nil {
		return nil, fmt.Errorf("cudacapture: compress journal: %w", err)
	}
	if int64(buf.Len()) > limit {
		return nil, ErrJournalTooLarge
	}
	return buf.Bytes(), nil
}
