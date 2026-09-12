package cudacapture

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
type Limits struct {
	// Files bounds the number of embedded files; default MaxCaptureFiles.
	Files int
	// FileBytes bounds one embedded file; default MaxCaptureFileBytes.
	FileBytes int64
	// TotalBytes bounds the sum of all embedded files; default
	// MaxCaptureTotalBytes.
	TotalBytes int64
	// InputBytes bounds one serialized profile read from an io.Reader or from
	// gpu.pprof; default MaxCaptureInputBytes.
	InputBytes int64
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
