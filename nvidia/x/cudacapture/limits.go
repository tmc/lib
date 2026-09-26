package cudacapture

import (
	"fmt"
	"strings"
)

// Limits bounds the size of a capture in both directions. The zero value means
// the package defaults, and a zero field means that field's default, so a
// caller may raise one bound without restating the others.
//
// Decoding a capture from an untrusted source should use the defaults. Every
// embedded file is held in memory and checksummed, so these bounds are what
// keeps a hostile artifact from exhausting the reader. Raise them only for a
// capture the caller is itself producing from data it already trusts, such as
// a sustained serving journal written into the producer's own scratch
// directory.
//
// A writer records in the capture every bound it raised above the default.
// A reader that trusts the capture can set Recorded to read it under those
// bounds without restating them.
type Limits struct {
	// Files bounds the number of embedded files; default MaxCaptureFiles.
	Files int `json:"files,omitempty"`
	// FileBytes bounds one embedded file; default MaxCaptureFileBytes.
	FileBytes int64 `json:"file_bytes,omitempty"`
	// TotalBytes bounds the sum of all embedded files; default
	// MaxCaptureTotalBytes.
	TotalBytes int64 `json:"total_bytes,omitempty"`
	// InputBytes bounds one serialized profile read from an io.Reader or from
	// gpu.pprof; default MaxCaptureInputBytes. It is never recorded: a
	// reader has read the input before it can learn what its writer used.
	InputBytes int64 `json:"input_bytes,omitempty"`

	// Recorded raises each of Files, FileBytes and TotalBytes left zero to
	// the bound the capture records it was written under, when that is above
	// the default. On a write the recorded bounds are the capture's own
	// Limits field. The bounds come from the capture itself, so set Recorded
	// only for a capture the caller trusts, such as one produced on this
	// machine.
	Recorded bool `json:"-"`
}

// withDefaults substitutes the package default for every unset field. It never
// lowers a bound the caller set and never removes one.
func (l Limits) withDefaults() Limits {
	if l.Files <= 0 {
		l.Files = MaxCaptureFiles
	}
	if l.FileBytes <= 0 {
		l.FileBytes = MaxCaptureFileBytes
	}
	if l.TotalBytes <= 0 {
		l.TotalBytes = MaxCaptureTotalBytes
	}
	if l.InputBytes <= 0 {
		l.InputBytes = MaxCaptureInputBytes
	}
	return l
}

// admit returns l with defaults, first raising every field l leaves unset to
// the one rec records, if l.Recorded. A bound the caller set is used as set,
// even below what rec records.
func (l Limits) admit(rec *Limits) Limits {
	if l.Recorded && rec != nil {
		if l.Files <= 0 {
			l.Files = max(MaxCaptureFiles, rec.Files)
		}
		if l.FileBytes <= 0 {
			l.FileBytes = max(MaxCaptureFileBytes, rec.FileBytes)
		}
		if l.TotalBytes <= 0 {
			l.TotalBytes = max(MaxCaptureTotalBytes, rec.TotalBytes)
		}
	}
	return l.withDefaults()
}

// raised returns the bounds of l above the package defaults, which is what a
// writer records for its readers, or nil when l raises none.
func (l Limits) raised() *Limits {
	var r Limits
	if l.Files > MaxCaptureFiles {
		r.Files = l.Files
	}
	if l.FileBytes > MaxCaptureFileBytes {
		r.FileBytes = l.FileBytes
	}
	if l.TotalBytes > MaxCaptureTotalBytes {
		r.TotalBytes = l.TotalBytes
	}
	if r == (Limits{}) {
		return nil
	}
	return &r
}

// A LimitError reports a capture whose files exceed the Limits it was read or
// written under. Need holds, for each bound exceeded, the value that would
// admit the capture; its other fields are zero.
type LimitError struct {
	// File names the largest file, when Need.FileBytes is set.
	File  string
	Need  Limits
	Limit Limits
}

func (e *LimitError) Error() string {
	var parts []string
	if e.Need.FileBytes > 0 {
		parts = append(parts, fmt.Sprintf("file %q is %d bytes, over the size limit of %d", e.File, e.Need.FileBytes, e.Limit.FileBytes))
	}
	if e.Need.TotalBytes > 0 {
		parts = append(parts, fmt.Sprintf("files total %d bytes, over the total size limit of %d", e.Need.TotalBytes, e.Limit.TotalBytes))
	}
	return "cudacapture: " + strings.Join(parts, "; ")
}

// checkSizes reports a *LimitError when the files named in sizes exceed the
// per-file or total bound of l. It reports both at once, so one error names
// every bound a caller must raise.
func checkSizes(sizes map[string]int64, l Limits) error {
	var largest string
	var total int64
	for name, n := range sizes {
		total += n
		if largest == "" || n > sizes[largest] || n == sizes[largest] && name < largest {
			largest = name
		}
	}
	e := &LimitError{Limit: l}
	if n := sizes[largest]; n > l.FileBytes {
		e.File, e.Need.FileBytes = largest, n
	}
	if total > l.TotalBytes {
		e.Need.TotalBytes = total
	}
	if e.Need == (Limits{}) {
		return nil
	}
	return e
}

// fileSizes returns the size of every file in files.
func fileSizes(files map[string][]byte) map[string]int64 {
	sizes := make(map[string]int64, len(files))
	for name, data := range files {
		sizes[name] = int64(len(data))
	}
	return sizes
}

// ownLimits returns the bounds a capture about to be written carries, which a
// Recorded write admits.
func ownLimits(c *Capture) *Limits {
	if c == nil {
		return nil
	}
	return &c.Limits
}

// recordedLimits returns the bounds a manifest records, or the zero Limits.
func recordedLimits(rec *Limits) Limits {
	if rec == nil {
		return Limits{}
	}
	return *rec
}
