// Package dsv4kvstore defines the private, versioned disk envelope used for
// host-side state persistence. The payload is owned by a backend; this package
// does not interpret device state or claim compatibility with upstream ds4.
//
// The DSKV envelope is deliberately not the upstream KVC format. Upstream
// files are inspected or read through the bounded opaque APIs in upstream.go;
// their backend payload is never interpreted by this package. No byte-
// compatible writer or backend restore is provided until the payload contract
// is verified against a matching build.
package dsv4kvstore

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

const (
	// Version is the current on-disk envelope version.
	Version uint16 = 1

	// QuantQ2 and QuantQ4 identify the two supported checkpoint quantization
	// families. The values are deliberately part of the file contract.
	QuantQ2 uint8 = 2
	QuantQ4 uint8 = 4

	headerSize        = 64
	defaultMaxTokens  = 1 << 24
	defaultMaxPayload = 1 << 34
)

// Options selects the host-side cache policy. The directory and budget
// fields are reserved for the on-disk store; StoreLen uses only the token
// boundary fields.
type Options struct {
	Dir                  string
	BudgetMB             int
	MinTokens            int
	ColdMaxTokens        int
	ContinuedInterval    int
	BoundaryTrimTokens   int
	BoundaryAlignTokens  int
	RejectDifferentQuant bool
}

const (
	defaultMinTokens         = 512
	defaultBudgetMB          = 4096
	defaultColdMaxTokens     = 30000
	defaultContinuedInterval = 10000
	defaultBoundaryTrim      = 32
	defaultBoundaryAlignment = 2048
)

// DefaultOptions returns the cache policy used by the upstream server. The
// returned value is independent and may be changed by the caller.
func DefaultOptions() Options {
	return Options{
		BudgetMB:            defaultBudgetMB,
		MinTokens:           defaultMinTokens,
		ColdMaxTokens:       defaultColdMaxTokens,
		ContinuedInterval:   defaultContinuedInterval,
		BoundaryTrimTokens:  defaultBoundaryTrim,
		BoundaryAlignTokens: defaultBoundaryAlignment,
	}
}

// StoreLen returns the prefix length that may be persisted at a cache
// boundary. Long prefixes are rounded down after the trim margin, and short
// prefixes remain unrounded. A zero or negative policy value selects the
// documented default. This function does not inspect model state or write a
// file; callers still decide whether a backend can serialize the prefix.
func StoreLen(tokens int, options Options) int {
	if tokens <= 0 {
		return 0
	}
	minTokens := options.MinTokens
	if minTokens <= 0 {
		minTokens = defaultMinTokens
	}
	trim := options.BoundaryTrimTokens
	if trim <= 0 {
		trim = defaultBoundaryTrim
	}
	align := options.BoundaryAlignTokens
	if align <= 0 {
		align = defaultBoundaryAlignment
	}
	if tokens <= minTokens+trim {
		return tokens
	}
	boundary := (tokens - trim) / align * align
	if boundary < minTokens {
		return tokens
	}
	return boundary
}

// ChatAnchor returns the last user-marker position before the first
// assistant marker when that position satisfies the cache minimum. It returns
// -1 when either marker is invalid, no user marker precedes the assistant
// marker, or the anchor is too short to be useful. Positions are zero-based.
// The marker IDs are supplied by the tokenizer; this package does not assign
// model-specific token IDs.
func ChatAnchor(tokens []int32, userTokenID, assistantTokenID int, options Options) int {
	if userTokenID < 0 || assistantTokenID < 0 {
		return -1
	}
	minTokens := options.MinTokens
	if minTokens <= 0 {
		minTokens = defaultMinTokens
	}
	lastUser := -1
	for i, token := range tokens {
		if int(token) == assistantTokenID {
			break
		}
		if int(token) == userTokenID {
			lastUser = i
		}
	}
	if lastUser < minTokens {
		return -1
	}
	return lastUser
}

// ContinuedStoreTarget returns the token count at which an already-live
// prefix may receive a continued checkpoint. The interval is rounded up to
// the boundary alignment. It returns zero when the interval is disabled, the
// prefix is below the minimum, the position is not aligned, or the position
// was already stored.
func ContinuedStoreTarget(liveTokens, lastStore int, options Options) int {
	defaults := DefaultOptions()
	step := options.ContinuedInterval
	if step <= 0 {
		return 0
	}
	align := options.BoundaryAlignTokens
	if align <= 0 {
		align = defaults.BoundaryAlignTokens
	}
	minTokens := options.MinTokens
	if minTokens == 0 {
		minTokens = defaults.MinTokens
	}
	if minTokens < 0 || liveTokens < minTokens {
		return 0
	}
	step = ((step + align - 1) / align) * align
	if liveTokens%step != 0 || liveTokens <= lastStore {
		return 0
	}
	return liveTokens
}

var magic = [4]byte{'D', 'S', 'K', 'V'}

var (
	ErrBadMagic       = errors.New("dsv4kvstore: bad magic")
	ErrUpstreamFormat = errors.New("dsv4kvstore: upstream KVC format is unsupported")
	ErrBadVersion     = errors.New("dsv4kvstore: unsupported version")
	ErrBadHeader      = errors.New("dsv4kvstore: invalid header")
	ErrBadQuant       = errors.New("dsv4kvstore: unsupported quantization")
	ErrQuantMismatch  = errors.New("dsv4kvstore: quantization mismatch")
	ErrTooLarge       = errors.New("dsv4kvstore: record exceeds decode limit")
	ErrTruncated      = errors.New("dsv4kvstore: truncated record")
	ErrTrailingData   = errors.New("dsv4kvstore: trailing data")
)

// Header describes the state associated with a Record. ModelDigest is an
// optional profile digest; a zero value means that the caller did not bind
// the record to a model identity.
type Header struct {
	Version     uint16
	QuantBits   uint8
	Context     uint32
	RawWindow   uint32
	ModelDigest [32]byte
}

// Record is a host-owned serialized state. Tokens and Payload are copied by
// Encode and Decode; callers may reuse their buffers afterwards. Payload is
// opaque so each backend can define its own state representation.
type Record struct {
	Header  Header
	Tokens  []int32
	Payload []byte
}

// DecodeOptions bounds allocations and optionally enforces a quantization
// policy. A zero limit uses the package default. ExpectedQuantBits is ignored
// unless RejectDifferentQuant is true.
type DecodeOptions struct {
	ExpectedQuantBits       uint8
	RejectDifferentQuant    bool
	ExpectedContext         uint32
	ExpectedRawWindow       uint32
	RejectDifferentGeometry bool
	MaxTokens               uint32
	MaxPayload              uint64
}

// Encode writes one complete record in the private DSKV envelope.
func Encode(w io.Writer, record Record) error {
	if w == nil {
		return errors.New("dsv4kvstore: nil writer")
	}
	if err := validateHeader(record.Header); err != nil {
		return err
	}
	if uint64(len(record.Tokens)) > defaultMaxTokens || uint64(len(record.Payload)) > defaultMaxPayload {
		return ErrTooLarge
	}
	h := record.Header
	h.Version = Version
	var b [headerSize]byte
	copy(b[0:4], magic[:])
	binary.LittleEndian.PutUint16(b[4:6], h.Version)
	binary.LittleEndian.PutUint16(b[6:8], headerSize)
	b[8] = h.QuantBits
	binary.LittleEndian.PutUint32(b[12:16], h.Context)
	binary.LittleEndian.PutUint32(b[16:20], h.RawWindow)
	binary.LittleEndian.PutUint32(b[20:24], uint32(len(record.Tokens)))
	binary.LittleEndian.PutUint64(b[24:32], uint64(len(record.Payload)))
	copy(b[32:64], h.ModelDigest[:])
	if err := writeFull(w, b[:]); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	var token [4]byte
	for _, value := range record.Tokens {
		binary.LittleEndian.PutUint32(token[:], uint32(value))
		if err := writeFull(w, token[:]); err != nil {
			return fmt.Errorf("write token: %w", err)
		}
	}
	if err := writeFull(w, record.Payload); err != nil {
		return fmt.Errorf("write payload: %w", err)
	}
	return nil
}

// Decode reads one record using the default allocation limits.
func Decode(r io.Reader) (Record, error) { return DecodeWithOptions(r, DecodeOptions{}) }

// DecodeWithOptions reads one record and applies explicit allocation and
// quantization limits before allocating token or payload buffers.
func DecodeWithOptions(r io.Reader, options DecodeOptions) (Record, error) {
	if r == nil {
		return Record{}, errors.New("dsv4kvstore: nil reader")
	}
	maxTokens, maxPayload := options.MaxTokens, options.MaxPayload
	if maxTokens == 0 {
		maxTokens = defaultMaxTokens
	}
	if maxPayload == 0 {
		maxPayload = defaultMaxPayload
	}
	var b [headerSize]byte
	if err := readFull(r, b[:]); err != nil {
		return Record{}, err
	}
	if string(b[0:3]) == "KVC" {
		return Record{}, fmt.Errorf("%w: %w", ErrUpstreamFormat, ErrUpstreamPayloadUnsupported)
	}
	if string(b[0:4]) != string(magic[:]) {
		return Record{}, ErrBadMagic
	}
	if binary.LittleEndian.Uint16(b[4:6]) != Version {
		return Record{}, ErrBadVersion
	}
	if binary.LittleEndian.Uint16(b[6:8]) != headerSize || b[9] != 0 || b[10] != 0 || b[11] != 0 {
		return Record{}, ErrBadHeader
	}
	h := Header{
		Version:   Version,
		QuantBits: b[8],
		Context:   binary.LittleEndian.Uint32(b[12:16]),
		RawWindow: binary.LittleEndian.Uint32(b[16:20]),
	}
	copy(h.ModelDigest[:], b[32:64])
	if err := validateHeader(h); err != nil {
		return Record{}, err
	}
	if options.RejectDifferentQuant && options.ExpectedQuantBits != h.QuantBits {
		return Record{}, fmt.Errorf("%w: file=%d expected=%d", ErrQuantMismatch, h.QuantBits, options.ExpectedQuantBits)
	}
	if options.RejectDifferentGeometry && (options.ExpectedContext != h.Context || options.ExpectedRawWindow != h.RawWindow) {
		return Record{}, fmt.Errorf("%w: context=%d/%d raw_window=%d/%d", ErrBadHeader,
			h.Context, options.ExpectedContext, h.RawWindow, options.ExpectedRawWindow)
	}
	n := binary.LittleEndian.Uint32(b[20:24])
	payloadBytes := binary.LittleEndian.Uint64(b[24:32])
	if n > maxTokens || payloadBytes > maxPayload || uint64(n) > uint64(^uint(0)>>1)/4 {
		return Record{}, ErrTooLarge
	}
	record := Record{Header: h, Tokens: make([]int32, n), Payload: make([]byte, payloadBytes)}
	var token [4]byte
	for i := range record.Tokens {
		if err := readFull(r, token[:]); err != nil {
			return Record{}, err
		}
		record.Tokens[i] = int32(binary.LittleEndian.Uint32(token[:]))
	}
	if err := readFull(r, record.Payload); err != nil {
		return Record{}, err
	}
	var extra [1]byte
	if n, err := r.Read(extra[:]); n != 0 {
		return Record{}, ErrTrailingData
	} else if err != nil && !errors.Is(err, io.EOF) {
		return Record{}, err
	}
	return record, nil
}

// CompatibleQuant reports whether a record may be loaded by a backend using
// quantBits. A zero quantBits means "unspecified" and is never compatible.
func CompatibleQuant(record Record, quantBits uint8) bool {
	return record.Header.QuantBits != 0 && record.Header.QuantBits == quantBits
}

func validateHeader(h Header) error {
	if h.Version != 0 && h.Version != Version {
		return ErrBadVersion
	}
	if h.QuantBits != QuantQ2 && h.QuantBits != QuantQ4 {
		return ErrBadQuant
	}
	if h.Context == 0 || h.RawWindow == 0 {
		return ErrBadHeader
	}
	return nil
}

func writeFull(w io.Writer, p []byte) error {
	for len(p) > 0 {
		n, err := w.Write(p)
		if err != nil {
			return err
		}
		if n <= 0 {
			return io.ErrShortWrite
		}
		p = p[n:]
	}
	return nil
}

func readFull(r io.Reader, p []byte) error {
	if _, err := io.ReadFull(r, p); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return ErrTruncated
		}
		return err
	}
	return nil
}
