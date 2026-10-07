package cudainject

import (
	"io"

	"github.com/tmc/lib/nvidia/internal/cudainject/journal"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// WireVersion is the injection record protocol version.
const WireVersion = wire.Version

// Record is a versioned injection record. Payload retains the producer's JSON.
type Record = wire.Record

// Identity identifies one agent generation in one process and capture session.
type Identity = wire.Identity

// Limits bounds the wire decoder. Zero fields select the existing defaults.
type Limits = wire.Limits

// Stats reports retained records, missing sequences and cumulative drops.
type Stats = wire.Stats

// Decoder reads bounded wire records. Next must have a single caller.
type Decoder = wire.Decoder

// Recovery retains a journal's valid prefix and its completeness and loss data.
type Recovery = journal.Recovery

// Record kinds identify the producer payload schema.
const (
	KindBootstrap  = wire.KindBootstrap
	KindActivity   = wire.KindActivity
	KindFinalized  = wire.KindFinalized
	KindCapability = wire.KindCapability
	KindFailure    = wire.KindFailure
	KindExternal   = wire.KindExternal
	KindAPI        = wire.KindAPI
	KindHost       = wire.KindHost
	KindModule     = wire.KindModule
	KindPCSample   = wire.KindPCSample
	KindGEMM       = wire.KindGEMM
)

// Decoder errors can be tested with errors.Is.
var (
	ErrBadMagic    = wire.ErrBadMagic
	ErrBadVersion  = wire.ErrBadVersion
	ErrMalformed   = wire.ErrMalformed
	ErrOversize    = wire.ErrOversize
	ErrSession     = wire.ErrSession
	ErrDuplicate   = wire.ErrDuplicate
	ErrRecordLimit = wire.ErrRecordLimit
)

// NewDecoder creates a reader with bounded payloads and record counts.
func NewDecoder(r io.Reader, limits Limits) *Decoder { return wire.NewDecoder(r, limits) }

// WriteRecord writes one record using the same bounds as NewDecoder.
func WriteRecord(w io.Writer, record Record, limits Limits) error {
	return wire.Write(w, record, limits)
}

// ReadJournal returns the valid prefix even when a journal is torn or invalid.
// Check Complete, Truncated and Stats before using it as complete evidence.
// Complete means frame EOF, not that the agent finalized or all providers ran.
func ReadJournal(path string, limits Limits) (Recovery, error) { return journal.Recover(path, limits) }

// ReadCapture reads and validates the existing bounded capture format and its
// manifest version. It does not decode injection-wire records as a capture.
func ReadCapture(r io.Reader, limits cudacapture.Limits) (*cudacapture.Capture, error) {
	return cudacapture.ReadLimits(r, limits)
}
