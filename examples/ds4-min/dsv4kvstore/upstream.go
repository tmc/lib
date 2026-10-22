package dsv4kvstore

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

const (
	upstreamKVCHeaderSize     = 48
	upstreamKVCPayloadABI     = 2
	upstreamPayloadHeaderSize = 13 * 4
	upstreamPayloadMagic      = uint32(0x34565344) // "DSV4" in little-endian form.
	upstreamPayloadVersion    = 2
	defaultUpstreamMaxText    = 1 << 30
	defaultUpstreamMaxPayload = 1 << 34
	defaultUpstreamMaxFile    = upstreamKVCHeaderSize + 4 + defaultUpstreamMaxText + defaultUpstreamMaxPayload
)

var (
	// ErrUpstreamPayloadUnsupported reports that an upstream KVC payload is
	// present but cannot be restored by this package. The payload contains
	// backend-owned session state and is intentionally not interpreted here.
	ErrUpstreamPayloadUnsupported = errors.New("dsv4kvstore: upstream KVC payload is unsupported")
	ErrUpstreamPayloadHeader      = errors.New("dsv4kvstore: invalid upstream payload header")
	ErrUpstreamBadHeader          = errors.New("dsv4kvstore: invalid upstream KVC header")
	ErrUpstreamBadVersion         = errors.New("dsv4kvstore: unsupported upstream KVC version")
	ErrUpstreamBadPayloadABI      = errors.New("dsv4kvstore: unsupported upstream KVC payload ABI")
	ErrUpstreamBadQuant           = errors.New("dsv4kvstore: unsupported upstream KVC quantization")
	ErrUpstreamTooLarge           = errors.New("dsv4kvstore: upstream KVC exceeds inspection limit")
	ErrUpstreamTruncated          = errors.New("dsv4kvstore: truncated upstream KVC")
)

// UpstreamInspectOptions bounds the bytes consumed by InspectUpstream. Zero
// values select conservative defaults. The limits apply to declared text and
// payload sizes and to the complete file, including an opaque trailer.
type UpstreamInspectOptions struct {
	MaxTextBytes    uint64
	MaxPayloadBytes uint64
	MaxFileBytes    uint64
}

// UpstreamMetadata is the validated metadata from an upstream KVC file. The
// payload and any trailer remain opaque; no device state is loaded.
type UpstreamMetadata struct {
	Version      uint8
	QuantBits    uint8
	Reason       uint8
	ModelID      uint8
	Tokens       uint32
	Hits         uint32
	Context      uint32
	PayloadABI   uint8
	CreatedAt    uint64
	LastUsed     uint64
	PayloadBytes uint64
	TextBytes    uint32
	TrailerBytes uint64
	FileBytes    uint64
}

// UpstreamRecord is a validated upstream KVC envelope whose backend payload
// is still opaque to this package. ReadUpstream copies the declared text and
// payload so a backend-specific owner can consume them; it does not restore
// device state or claim cross-backend numerical compatibility.
type UpstreamRecord struct {
	Metadata UpstreamMetadata
	Text     []byte
	Payload  []byte
	Trailer  []byte
	// PayloadSHA256 identifies the opaque payload bytes returned by ReadUpstream.
	// It is a transport fingerprint only: matching bytes do not establish that a
	// backend can restore the payload or that the restored state is compatible.
	PayloadSHA256 [sha256.Size]byte
}

// UpstreamPayloadMetadata is the version-qualified header of the DeepSeek V4
// session payload used by upstream KVC files. The tensor bytes following this
// header remain opaque; this type does not make them safe to restore.
//
// The field meanings are defined by upstream's DS4_SESSION_PAYLOAD_VERSION 2
// contract. They are reported so a backend owner can decide whether it has a
// matching codec, rather than being silently guessed by this package.
type UpstreamPayloadMetadata struct {
	Magic           uint32
	Version         uint32
	Context         uint32
	PrefillCapacity uint32
	RawCapacity     uint32
	RawWindow       uint32
	CompressedCap   uint32
	Tokens          uint32
	Layers          uint32
	HeadDim         uint32
	IndexerHeadDim  uint32
	Vocab           uint32
	RawLive         uint32
}

// UpstreamPayloadSection identifies one bounded tensor section in an
// upstream payload. Offsets are relative to the payload start.
type UpstreamPayloadSection struct {
	Layer  int
	Name   string
	Offset uint64
	Bytes  uint64
}

// UpstreamPayloadLayout is a checked map of the fixed DSV4 payload sections.
// It is an inspection result only; it does not make the opaque tensor bytes
// safe to copy into a backend state.
type UpstreamPayloadLayout struct {
	Metadata UpstreamPayloadMetadata
	Sections []UpstreamPayloadSection
	Bytes    uint64
}

// HashUpstreamPayloadTokenIDs returns the SHA-256 digest of the payload's
// token-ID section. The section is encoded as little-endian uint32 values
// immediately after the fixed payload header, matching the upstream token
// provenance convention. The helper hashes bounded chunks and does not
// allocate according to the token count.
func HashUpstreamPayloadTokenIDs(r io.ReaderAt, layout UpstreamPayloadLayout) ([sha256.Size]byte, error) {
	if r == nil {
		return [sha256.Size]byte{}, errors.New("dsv4kvstore: nil upstream payload reader")
	}
	if layout.Bytes < upstreamPayloadHeaderSize {
		return [sha256.Size]byte{}, ErrUpstreamPayloadHeader
	}
	tokenBytes, ok := checkedUpstreamProduct(uint64(layout.Metadata.Tokens), 4)
	if !ok || tokenBytes > layout.Bytes-upstreamPayloadHeaderSize {
		return [sha256.Size]byte{}, ErrUpstreamPayloadHeader
	}
	h := sha256.New()
	const chunkSize = 64 << 10
	buf := make([]byte, chunkSize)
	remaining := tokenBytes
	offset := uint64(upstreamPayloadHeaderSize)
	for remaining != 0 {
		n := uint64(len(buf))
		if n > remaining {
			n = remaining
		}
		if err := readUpstreamAt(r, buf[:n], int64(offset)); err != nil {
			return [sha256.Size]byte{}, upstreamReadError(err)
		}
		if _, err := h.Write(buf[:n]); err != nil {
			return [sha256.Size]byte{}, fmt.Errorf("hash upstream token IDs: %w", err)
		}
		offset += n
		remaining -= n
	}
	var digest [sha256.Size]byte
	copy(digest[:], h.Sum(nil))
	return digest, nil
}

// UpstreamRestoreContract contains caller-owned provenance and geometry
// requirements for a backend restore. The KVC envelope does not carry a model
// hash or token-ID hash, so a backend must supply and verify both explicitly.
type UpstreamRestoreContract struct {
	ModelSHA256         [sha256.Size]byte
	TokenIDsSHA256      [sha256.Size]byte
	PayloadABI          uint8
	QuantBits           uint8
	Context             uint32
	PrefillCapacity     uint32
	CompressedCap       uint32
	Tokens              uint32
	Layers              uint32
	HeadDim             uint32
	IndexerHeadDim      uint32
	Vocab               uint32
	UpstreamRawCapacity uint32
	TargetRawCapacity   uint32
	RawWindow           uint32
	Ratios              []uint32
}

// ValidateUpstreamRestoreContract checks the provenance and geometry gates
// required before a backend may copy an upstream payload into device state.
// It performs no device work and does not claim that the payload is
// restorable; it only rejects missing or mismatched caller requirements.
func ValidateUpstreamRestoreContract(layout UpstreamPayloadLayout, contract UpstreamRestoreContract) error {
	if layout.Metadata.Layers == 0 || layout.Bytes == 0 {
		return ErrUpstreamPayloadHeader
	}
	if contract.ModelSHA256 == ([sha256.Size]byte{}) || contract.TokenIDsSHA256 == ([sha256.Size]byte{}) {
		return errors.New("dsv4kvstore: restore provenance hashes are required")
	}
	m := layout.Metadata
	if contract.PayloadABI != upstreamKVCPayloadABI || contract.QuantBits == 0 ||
		contract.TargetRawCapacity == 0 || contract.TargetRawCapacity < m.RawLive ||
		contract.Context != m.Context || contract.PrefillCapacity != m.PrefillCapacity ||
		contract.CompressedCap != m.CompressedCap || contract.Tokens != m.Tokens ||
		contract.Layers != m.Layers ||
		contract.HeadDim != m.HeadDim || contract.IndexerHeadDim != m.IndexerHeadDim ||
		contract.Vocab != m.Vocab || contract.UpstreamRawCapacity != m.RawCapacity ||
		contract.RawWindow != m.RawWindow {
		return fmt.Errorf("dsv4kvstore: restore geometry does not match payload")
	}
	if len(contract.Ratios) != int(m.Layers) {
		return fmt.Errorf("dsv4kvstore: restore ratio count %d, want %d", len(contract.Ratios), m.Layers)
	}
	for i, ratio := range contract.Ratios {
		if ratio != 0 && ratio != 4 && ratio != 128 {
			return fmt.Errorf("dsv4kvstore: restore ratio %d at layer %d is unsupported", ratio, i)
		}
	}
	return nil
}

// UpstreamRawRowMapping identifies the destination slot for one logical raw
// row. Upstream retains a logical window while ds4-min uses a compact ring.
type UpstreamRawRowMapping struct {
	Position uint32
	Slot     uint32
	Offset   uint64
}

// MapUpstreamRawRows maps the live upstream raw window into a target ring. It
// is a pure geometry operation and does not authorize a device upload.
func MapUpstreamRawRows(meta UpstreamPayloadMetadata, targetRawCapacity uint32) ([]UpstreamRawRowMapping, error) {
	if targetRawCapacity == 0 || meta.RawLive == 0 || meta.RawLive > targetRawCapacity || meta.RawLive > meta.Tokens || meta.HeadDim == 0 {
		return nil, ErrUpstreamPayloadHeader
	}
	start := meta.Tokens - meta.RawLive
	rows := make([]UpstreamRawRowMapping, meta.RawLive)
	for i := range rows {
		position := start + uint32(i)
		slot := position % targetRawCapacity
		rows[i] = UpstreamRawRowMapping{
			Position: position,
			Slot:     slot,
			Offset:   uint64(slot) * uint64(meta.HeadDim) * 4,
		}
	}
	return rows, nil
}

// ReadUpstreamPayloadSection copies one validated section into dst. The
// caller owns dst; this helper never allocates based on section metadata.
func ReadUpstreamPayloadSection(r io.ReaderAt, layout UpstreamPayloadLayout, layer int, name string, dst []byte) error {
	if r == nil {
		return errors.New("dsv4kvstore: nil upstream payload reader")
	}
	var found *UpstreamPayloadSection
	for i := range layout.Sections {
		section := &layout.Sections[i]
		if section.Layer == layer && section.Name == name {
			if found != nil {
				return fmt.Errorf("dsv4kvstore: duplicate payload section %d/%s", layer, name)
			}
			found = section
		}
	}
	if found == nil {
		return fmt.Errorf("dsv4kvstore: payload section %d/%s not found", layer, name)
	}
	if found.Bytes > layout.Bytes || found.Offset > layout.Bytes-found.Bytes || found.Offset > uint64(^uint64(0)>>1) {
		return ErrUpstreamPayloadHeader
	}
	if uint64(len(dst)) != found.Bytes {
		return fmt.Errorf("dsv4kvstore: section buffer %d, want %d", len(dst), found.Bytes)
	}
	if err := readUpstreamAt(r, dst, int64(found.Offset)); err != nil {
		return upstreamReadError(err)
	}
	return nil
}

// ValidateUpstreamRecordLayout checks that an envelope and its payload agree
// before a backend attempts to interpret the payload. It verifies the bounded
// payload length and fingerprint, then walks the payload using the caller's
// version-qualified compression ratios. The returned layout is still an
// inspection result; it does not authorize a device restore.
func ValidateUpstreamRecordLayout(record UpstreamRecord, ratios []uint32) (UpstreamPayloadLayout, error) {
	if uint64(len(record.Payload)) != record.Metadata.PayloadBytes {
		return UpstreamPayloadLayout{}, fmt.Errorf("dsv4kvstore: payload length %d, want %d", len(record.Payload), record.Metadata.PayloadBytes)
	}
	if got := sha256.Sum256(record.Payload); got != record.PayloadSHA256 {
		return UpstreamPayloadLayout{}, errors.New("dsv4kvstore: payload fingerprint does not match record")
	}
	layout, err := InspectUpstreamPayloadLayout(bytes.NewReader(record.Payload), uint64(len(record.Payload)), ratios)
	if err != nil {
		return UpstreamPayloadLayout{}, err
	}
	if layout.Metadata.Context != record.Metadata.Context || layout.Metadata.Tokens != record.Metadata.Tokens {
		return UpstreamPayloadLayout{}, errors.New("dsv4kvstore: payload metadata does not match envelope")
	}
	return layout, nil
}

// InspectUpstreamPayload reads and validates the fixed DSV4 payload header.
// It consumes exactly 52 bytes and does not interpret or allocate any tensor
// state. A successful result is a compatibility fact about the header only;
// callers still need a backend codec, synchronization, and graph policy before
// restoring the remaining payload.
func InspectUpstreamPayload(r io.Reader, payloadBytes uint64) (UpstreamPayloadMetadata, error) {
	if r == nil {
		return UpstreamPayloadMetadata{}, errors.New("dsv4kvstore: nil upstream payload reader")
	}
	if payloadBytes < upstreamPayloadHeaderSize {
		return UpstreamPayloadMetadata{}, ErrUpstreamTruncated
	}
	var b [upstreamPayloadHeaderSize]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return UpstreamPayloadMetadata{}, upstreamReadError(err)
	}
	values := make([]uint32, upstreamPayloadHeaderSize/4)
	for i := range values {
		values[i] = binary.LittleEndian.Uint32(b[i*4:])
	}
	if values[0] != upstreamPayloadMagic || values[1] != upstreamPayloadVersion {
		return UpstreamPayloadMetadata{}, ErrUpstreamPayloadHeader
	}
	meta := UpstreamPayloadMetadata{
		Magic: values[0], Version: values[1], Context: values[2],
		PrefillCapacity: values[3], RawCapacity: values[4],
		RawWindow: values[5], CompressedCap: values[6], Tokens: values[7],
		Layers: values[8], HeadDim: values[9], IndexerHeadDim: values[10],
		Vocab: values[11], RawLive: values[12],
	}
	if meta.Context == 0 || meta.Tokens == 0 || meta.Tokens >= meta.Context ||
		meta.RawCapacity == 0 || meta.RawWindow == 0 || meta.RawWindow > meta.RawCapacity ||
		meta.Layers == 0 || meta.HeadDim == 0 || meta.IndexerHeadDim == 0 ||
		meta.Vocab == 0 || meta.RawLive > meta.RawCapacity {
		return UpstreamPayloadMetadata{}, ErrUpstreamPayloadHeader
	}
	expectedRawLive := meta.Tokens
	if expectedRawLive > meta.RawWindow {
		expectedRawLive = meta.RawWindow
	}
	if meta.RawLive != expectedRawLive {
		return UpstreamPayloadMetadata{}, ErrUpstreamPayloadHeader
	}
	return meta, nil
}

// InspectUpstreamPayloadLayout validates the complete tensor layout of an
// upstream DSV4 payload. ratios must contain one compression ratio per layer;
// zero, four, and 128 are the only ratios defined by the payload ABI. The
// reader must support ReaderAt because this function validates the per-layer
// counts before walking the sections.
func InspectUpstreamPayloadLayout(r io.ReaderAt, payloadBytes uint64, ratios []uint32) (UpstreamPayloadLayout, error) {
	if r == nil {
		return UpstreamPayloadLayout{}, errors.New("dsv4kvstore: nil upstream payload reader")
	}
	if payloadBytes < upstreamPayloadHeaderSize {
		return UpstreamPayloadLayout{}, ErrUpstreamTruncated
	}
	var header [upstreamPayloadHeaderSize]byte
	if err := readUpstreamAt(r, header[:], 0); err != nil {
		return UpstreamPayloadLayout{}, upstreamReadError(err)
	}
	meta, err := InspectUpstreamPayload(bytes.NewReader(header[:]), payloadBytes)
	if err != nil {
		return UpstreamPayloadLayout{}, err
	}
	if uint64(len(ratios)) != uint64(meta.Layers) {
		return UpstreamPayloadLayout{}, fmt.Errorf("dsv4kvstore: compression ratio count %d, want %d", len(ratios), meta.Layers)
	}
	add := func(a, b uint64) (uint64, error) {
		if b > ^uint64(0)-a {
			return 0, ErrUpstreamTooLarge
		}
		return a + b, nil
	}
	mul := func(a, b uint64) (uint64, error) {
		if a != 0 && b > ^uint64(0)/a {
			return 0, ErrUpstreamTooLarge
		}
		return a * b, nil
	}
	sectionBytes := func(count, width uint64) (uint64, error) {
		return mul(count, width*4)
	}
	offset := uint64(upstreamPayloadHeaderSize)
	var n, bytes uint64
	if n, err = mul(uint64(meta.Tokens), 4); err != nil {
		return UpstreamPayloadLayout{}, err
	}
	if offset, err = add(offset, n); err != nil {
		return UpstreamPayloadLayout{}, err
	}
	if n, err = mul(uint64(meta.Vocab), 4); err != nil {
		return UpstreamPayloadLayout{}, err
	}
	if offset, err = add(offset, n); err != nil {
		return UpstreamPayloadLayout{}, err
	}
	counts := offset
	if n, err = mul(uint64(meta.Layers), 4*2); err != nil {
		return UpstreamPayloadLayout{}, err
	}
	if offset, err = add(offset, n); err != nil {
		return UpstreamPayloadLayout{}, err
	}
	readU32 := func(at uint64) (uint32, error) {
		var b [4]byte
		if err := readUpstreamAt(r, b[:], int64(at)); err != nil {
			return 0, upstreamReadError(err)
		}
		return binary.LittleEndian.Uint32(b[:]), nil
	}
	sections := make([]UpstreamPayloadSection, 0, int(meta.Layers)*5)
	addSection := func(layer int, name string, size uint64) error {
		if size == 0 {
			return nil
		}
		if offset, err = add(offset, size); err != nil {
			return err
		}
		sections = append(sections, UpstreamPayloadSection{Layer: layer, Name: name, Offset: offset - size, Bytes: size})
		return nil
	}
	for layer, ratio := range ratios {
		if ratio != 0 && ratio != 4 && ratio != 128 {
			return UpstreamPayloadLayout{}, fmt.Errorf("dsv4kvstore: unsupported compression ratio %d at layer %d", ratio, layer)
		}
		nComp, err := readU32(counts + uint64(layer)*4)
		if err != nil {
			return UpstreamPayloadLayout{}, err
		}
		nIndex, err := readU32(counts + uint64(meta.Layers)*4 + uint64(layer)*4)
		if err != nil {
			return UpstreamPayloadLayout{}, err
		}
		rawBytes, err := sectionBytes(uint64(meta.RawLive), uint64(meta.HeadDim))
		if err != nil {
			return UpstreamPayloadLayout{}, err
		}
		if err := addSection(layer, "raw", rawBytes); err != nil {
			return UpstreamPayloadLayout{}, err
		}
		if ratio == 0 {
			continue
		}
		compressedBytes, err := sectionBytes(uint64(nComp), uint64(meta.HeadDim))
		if err != nil {
			return UpstreamPayloadLayout{}, err
		}
		if err := addSection(layer, "compressed", compressedBytes); err != nil {
			return UpstreamPayloadLayout{}, err
		}
		coff := uint64(1)
		if ratio == 4 {
			coff = 2
		}
		state, err := mul(coff*uint64(meta.HeadDim), coff*uint64(ratio)*4)
		if err != nil {
			return UpstreamPayloadLayout{}, err
		}
		if err := addSection(layer, "attention_state_kv", state); err != nil {
			return UpstreamPayloadLayout{}, err
		}
		if err := addSection(layer, "attention_state_score", state); err != nil {
			return UpstreamPayloadLayout{}, err
		}
		if ratio == 4 {
			indexerBytes, err := sectionBytes(uint64(nIndex), uint64(meta.IndexerHeadDim))
			if err != nil {
				return UpstreamPayloadLayout{}, err
			}
			if err := addSection(layer, "indexer", indexerBytes); err != nil {
				return UpstreamPayloadLayout{}, err
			}
			indexState, err := mul(coff*uint64(meta.IndexerHeadDim), coff*uint64(ratio)*4)
			if err != nil {
				return UpstreamPayloadLayout{}, err
			}
			if err := addSection(layer, "index_state_kv", indexState); err != nil {
				return UpstreamPayloadLayout{}, err
			}
			if err := addSection(layer, "index_state_score", indexState); err != nil {
				return UpstreamPayloadLayout{}, err
			}
		}
	}
	bytes = offset
	if bytes != payloadBytes {
		return UpstreamPayloadLayout{}, fmt.Errorf("dsv4kvstore: payload layout ends at %d, want %d", bytes, payloadBytes)
	}
	return UpstreamPayloadLayout{Metadata: meta, Sections: sections, Bytes: bytes}, nil
}

func checkedUpstreamProduct(a, b uint64) (uint64, bool) {
	if a != 0 && b > ^uint64(0)/a {
		return 0, false
	}
	return a * b, true
}

func readUpstreamAt(r io.ReaderAt, dst []byte, offset int64) error {
	for len(dst) != 0 {
		n, err := r.ReadAt(dst, offset)
		offset += int64(n)
		dst = dst[n:]
		if err != nil {
			if err == io.EOF && len(dst) == 0 {
				return nil
			}
			return err
		}
		if n == 0 {
			return io.ErrUnexpectedEOF
		}
	}
	return nil
}

// ReadUpstream validates and reads an upstream KVC envelope under the limits
// in options. Declared body sizes are checked before allocation. The payload
// remains uninterpreted and callers must not pass it to a backend without a
// matching payload ABI and model/quantization check.
func ReadUpstream(r io.Reader, options UpstreamInspectOptions) (UpstreamRecord, error) {
	if r == nil {
		return UpstreamRecord{}, errors.New("dsv4kvstore: nil upstream KVC reader")
	}
	options = normalizeUpstreamInspectOptions(options)
	if options.MaxFileBytes < upstreamKVCHeaderSize+4 {
		return UpstreamRecord{}, ErrUpstreamTooLarge
	}
	var h [upstreamKVCHeaderSize]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		return UpstreamRecord{}, upstreamReadError(err)
	}
	var textLen [4]byte
	if _, err := io.ReadFull(r, textLen[:]); err != nil {
		return UpstreamRecord{}, upstreamReadError(err)
	}
	fixed := make([]byte, len(h)+len(textLen))
	copy(fixed, h[:])
	copy(fixed[len(h):], textLen[:])
	meta, err := parseUpstreamFixed(fixed, options)
	if err != nil {
		return UpstreamRecord{}, err
	}
	bodyBytes := uint64(upstreamKVCHeaderSize + 4)
	if meta.TextBytes > 0 {
		bodyBytes += uint64(meta.TextBytes)
	}
	if meta.PayloadBytes > ^uint64(0)-bodyBytes {
		return UpstreamRecord{}, ErrUpstreamTooLarge
	}
	bodyBytes += meta.PayloadBytes
	if bodyBytes > options.MaxFileBytes {
		return UpstreamRecord{}, ErrUpstreamTooLarge
	}
	text := make([]byte, meta.TextBytes)
	if _, err := io.ReadFull(r, text); err != nil {
		return UpstreamRecord{}, upstreamReadError(err)
	}
	payload := make([]byte, meta.PayloadBytes)
	if _, err := io.ReadFull(r, payload); err != nil {
		return UpstreamRecord{}, upstreamReadError(err)
	}
	remaining := options.MaxFileBytes - bodyBytes
	if remaining >= uint64(^uint64(0)>>1) {
		return UpstreamRecord{}, ErrUpstreamTooLarge
	}
	trailer, err := io.ReadAll(io.LimitReader(r, int64(remaining)+1))
	if err != nil {
		return UpstreamRecord{}, fmt.Errorf("read upstream KVC trailer: %w", err)
	}
	if uint64(len(trailer)) > remaining {
		return UpstreamRecord{}, ErrUpstreamTooLarge
	}
	meta.FileBytes = bodyBytes + uint64(len(trailer))
	meta.TrailerBytes = uint64(len(trailer))
	return UpstreamRecord{
		Metadata:      meta,
		Text:          text,
		Payload:       payload,
		Trailer:       trailer,
		PayloadSHA256: sha256.Sum256(payload),
	}, nil
}

// ReadUpstreamPath opens a regular upstream KVC file and reads its validated
// opaque envelope. The file size is checked before body allocation.
func ReadUpstreamPath(path string, options UpstreamInspectOptions) (UpstreamRecord, error) {
	if path == "" {
		return UpstreamRecord{}, errors.New("dsv4kvstore: missing upstream KVC path")
	}
	f, err := os.Open(path)
	if err != nil {
		return UpstreamRecord{}, fmt.Errorf("open upstream KVC: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return UpstreamRecord{}, fmt.Errorf("stat upstream KVC: %w", err)
	}
	if !info.Mode().IsRegular() {
		return UpstreamRecord{}, errors.New("dsv4kvstore: upstream KVC is not a regular file")
	}
	options = normalizeUpstreamInspectOptions(options)
	if uint64(info.Size()) > options.MaxFileBytes {
		return UpstreamRecord{}, ErrUpstreamTooLarge
	}
	record, err := ReadUpstream(f, options)
	if err != nil {
		return UpstreamRecord{}, err
	}
	if record.Metadata.FileBytes != uint64(info.Size()) {
		return UpstreamRecord{}, ErrUpstreamTruncated
	}
	return record, nil
}

// InspectUpstreamFile validates the fixed metadata and declared body sizes of
// an upstream KVC file without reading its opaque body. size must be the file
// size supplied by the caller. The returned TrailerBytes is the bytes after
// the declared text and payload. This is an inspection API only; it does not
// make the backend payload loadable by dsv4kvstore.
func InspectUpstreamFile(r io.ReaderAt, size uint64, options UpstreamInspectOptions) (UpstreamMetadata, error) {
	if r == nil {
		return UpstreamMetadata{}, errors.New("dsv4kvstore: nil upstream KVC reader")
	}
	options = normalizeUpstreamInspectOptions(options)
	if size < upstreamKVCHeaderSize+4 {
		return UpstreamMetadata{}, ErrUpstreamTruncated
	}
	if size > options.MaxFileBytes {
		return UpstreamMetadata{}, ErrUpstreamTooLarge
	}

	var fixed [upstreamKVCHeaderSize + 4]byte
	if _, err := io.ReadFull(io.NewSectionReader(r, 0, int64(len(fixed))), fixed[:]); err != nil {
		return UpstreamMetadata{}, upstreamReadError(err)
	}
	meta, err := parseUpstreamFixed(fixed[:], options)
	if err != nil {
		return UpstreamMetadata{}, err
	}
	bodyBytes := uint64(upstreamKVCHeaderSize+4) + uint64(meta.TextBytes)
	if meta.PayloadBytes > ^uint64(0)-bodyBytes {
		return UpstreamMetadata{}, ErrUpstreamTooLarge
	}
	bodyBytes += meta.PayloadBytes
	if bodyBytes > size {
		return UpstreamMetadata{}, ErrUpstreamTruncated
	}
	meta.FileBytes = size
	meta.TrailerBytes = size - bodyBytes
	return meta, nil
}

// InspectUpstreamPath opens path and validates its fixed KVC metadata without
// reading the opaque text, payload, or trailer. The file size is checked
// before any bytes are read, so an oversized cache file is rejected without
// allocating according to its declarations. This is an inspection API only;
// it does not make the backend payload loadable by dsv4kvstore.
func InspectUpstreamPath(path string, options UpstreamInspectOptions) (UpstreamMetadata, error) {
	if path == "" {
		return UpstreamMetadata{}, errors.New("dsv4kvstore: missing upstream KVC path")
	}
	f, err := os.Open(path)
	if err != nil {
		return UpstreamMetadata{}, fmt.Errorf("open upstream KVC: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return UpstreamMetadata{}, fmt.Errorf("stat upstream KVC: %w", err)
	}
	if !info.Mode().IsRegular() {
		return UpstreamMetadata{}, errors.New("dsv4kvstore: upstream KVC is not a regular file")
	}
	return InspectUpstreamFile(f, uint64(info.Size()), options)
}

// InspectUpstream validates and summarizes an upstream KVC file. It consumes
// the opaque text, payload, and trailer without allocating them. This is a
// compatibility inspection API only; it does not make the backend payload
// loadable by dsv4kvstore.
func InspectUpstream(r io.Reader, options UpstreamInspectOptions) (UpstreamMetadata, error) {
	if r == nil {
		return UpstreamMetadata{}, errors.New("dsv4kvstore: nil upstream KVC reader")
	}
	options = normalizeUpstreamInspectOptions(options)
	if options.MaxFileBytes < upstreamKVCHeaderSize+4 {
		return UpstreamMetadata{}, ErrUpstreamTooLarge
	}

	var h [upstreamKVCHeaderSize]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		return UpstreamMetadata{}, upstreamReadError(err)
	}
	var textLen [4]byte
	if _, err := io.ReadFull(r, textLen[:]); err != nil {
		return UpstreamMetadata{}, upstreamReadError(err)
	}
	fixed := make([]byte, len(h)+len(textLen))
	copy(fixed, h[:])
	copy(fixed[len(h):], textLen[:])
	meta, err := parseUpstreamFixed(fixed, options)
	if err != nil {
		return UpstreamMetadata{}, err
	}
	if err := discardUpstream(r, uint64(meta.TextBytes), &meta.FileBytes, options.MaxFileBytes); err != nil {
		return UpstreamMetadata{}, err
	}
	if err := discardUpstream(r, meta.PayloadBytes, &meta.FileBytes, options.MaxFileBytes); err != nil {
		return UpstreamMetadata{}, err
	}
	for {
		var b [4096]byte
		n, err := r.Read(b[:])
		if n > 0 {
			if uint64(n) > options.MaxFileBytes-meta.FileBytes {
				return UpstreamMetadata{}, ErrUpstreamTooLarge
			}
			meta.FileBytes += uint64(n)
			meta.TrailerBytes += uint64(n)
		}
		if err == io.EOF {
			return meta, nil
		}
		if err != nil {
			return UpstreamMetadata{}, err
		}
		if n == 0 {
			return UpstreamMetadata{}, io.ErrNoProgress
		}
	}
}

func normalizeUpstreamInspectOptions(options UpstreamInspectOptions) UpstreamInspectOptions {
	if options.MaxTextBytes == 0 {
		options.MaxTextBytes = defaultUpstreamMaxText
	}
	if options.MaxPayloadBytes == 0 {
		options.MaxPayloadBytes = defaultUpstreamMaxPayload
	}
	if options.MaxFileBytes == 0 {
		options.MaxFileBytes = defaultUpstreamMaxFile
	}
	return options
}

func parseUpstreamFixed(fixed []byte, options UpstreamInspectOptions) (UpstreamMetadata, error) {
	if len(fixed) != upstreamKVCHeaderSize+4 {
		return UpstreamMetadata{}, ErrUpstreamBadHeader
	}
	h := fixed[:upstreamKVCHeaderSize]
	if string(h[:3]) != "KVC" {
		return UpstreamMetadata{}, ErrUpstreamFormat
	}
	if h[3] != 1 {
		return UpstreamMetadata{}, ErrUpstreamBadVersion
	}
	if h[20] != upstreamKVCPayloadABI {
		return UpstreamMetadata{}, ErrUpstreamBadPayloadABI
	}
	if h[4] != QuantQ2 && h[4] != QuantQ4 {
		return UpstreamMetadata{}, ErrUpstreamBadQuant
	}
	meta := UpstreamMetadata{
		Version: h[3], QuantBits: h[4], Reason: h[5], ModelID: h[7],
		Tokens:       binary.LittleEndian.Uint32(h[8:12]),
		Hits:         binary.LittleEndian.Uint32(h[12:16]),
		Context:      binary.LittleEndian.Uint32(h[16:20]),
		PayloadABI:   h[20],
		CreatedAt:    binary.LittleEndian.Uint64(h[24:32]),
		LastUsed:     binary.LittleEndian.Uint64(h[32:40]),
		PayloadBytes: binary.LittleEndian.Uint64(h[40:48]),
		TextBytes:    binary.LittleEndian.Uint32(fixed[48:52]),
		FileBytes:    upstreamKVCHeaderSize + 4,
	}
	if meta.Tokens == 0 || meta.Context == 0 {
		return UpstreamMetadata{}, ErrUpstreamBadHeader
	}
	if uint64(meta.TextBytes) > options.MaxTextBytes || meta.PayloadBytes > options.MaxPayloadBytes {
		return UpstreamMetadata{}, ErrUpstreamTooLarge
	}
	if meta.PayloadBytes > options.MaxFileBytes-meta.FileBytes || uint64(meta.TextBytes) > options.MaxFileBytes-meta.FileBytes-meta.PayloadBytes {
		return UpstreamMetadata{}, ErrUpstreamTooLarge
	}
	return meta, nil
}

// UpstreamPayloadError makes the unsupported part of the compatibility
// boundary explicit for callers that want to restore an inspected payload.
// ReadUpstream provides bytes; backend restoration still requires a
// version-qualified payload contract.
func UpstreamPayloadError() error { return ErrUpstreamPayloadUnsupported }

func discardUpstream(r io.Reader, n uint64, total *uint64, max uint64) error {
	if n == 0 {
		return nil
	}
	if n > uint64(^uint64(0)>>1) {
		return ErrUpstreamTooLarge
	}
	if *total > max-n {
		return ErrUpstreamTooLarge
	}
	read, err := io.CopyN(io.Discard, r, int64(n))
	if err != nil {
		return upstreamReadError(err)
	}
	if uint64(read) != n {
		return ErrUpstreamTruncated
	}
	*total += n
	return nil
}

func upstreamReadError(err error) error {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return ErrUpstreamTruncated
	}
	return fmt.Errorf("read upstream KVC: %w", err)
}
