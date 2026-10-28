package cudadecomp

import (
	"bytes"
	"compress/flate"
	"context"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"strconv"
)

// A Codec is a chunk compression format.
type Codec uint8

const (
	CodecStored  Codec = iota // uncompressed
	CodecLZ4                  // LZ4 block format
	CodecDeflate              // raw Deflate (RFC 1951)
)

func (c Codec) String() string {
	switch c {
	case CodecStored:
		return "stored"
	case CodecLZ4:
		return "lz4"
	case CodecDeflate:
		return "deflate"
	}
	return "Codec(" + strconv.Itoa(int(c)) + ")"
}

// A Layout says how a tensor's decompressed bytes are arranged.
type Layout uint8

const (
	LayoutBytes      Layout = iota // the tensor's bytes, unchanged
	LayoutBytePlanes               // BF16: all high bytes, then all low bytes
)

// DefaultChunkSize is the uncompressed chunk size [CompressBF16] uses.
const DefaultChunkSize = 64 << 10

// A Chunk describes one independently compressed piece of a [Tensor].
type Chunk struct {
	// Codec is the chunk's format: the tensor's codec, or CodecStored
	// if compression did not make the chunk smaller.
	Codec     Codec
	SrcOffset uint64 // offset of the compressed bytes in Payload
	SrcSize   uint32
	DstOffset uint64 // offset of the uncompressed bytes in the tensor
	DstSize   uint32
	CRC32     uint32 // CRC-32C of the uncompressed bytes
}

// A Tensor is a chunk-compressed byte stream.
type Tensor struct {
	Name      string
	Codec     Codec
	Layout    Layout
	ChunkSize int
	Size      int // uncompressed bytes
	Chunks    []Chunk
	Payload   []byte
}

// Ratio returns the uncompressed size divided by the payload size.
func (t Tensor) Ratio() float64 {
	if len(t.Payload) == 0 {
		return 0
	}
	return float64(t.Size) / float64(len(t.Payload))
}

// ErrChunkCorrupted reports a chunk that failed to decode or whose checksum
// does not match.
var ErrChunkCorrupted = errors.New("cudadecomp: chunk corrupted")

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// Compress cuts data into chunkSize pieces and compresses each with codec.
// A piece that does not shrink is stored.
func Compress(name string, data []byte, codec Codec, chunkSize int) (Tensor, error) {
	return compress(name, data, codec, LayoutBytes, chunkSize, len(data))
}

// CompressBF16 splits w into byte planes with [SplitBytePlanes] and
// compresses each plane in [DefaultChunkSize] chunks. No chunk spans the
// two planes.
func CompressBF16(name string, w []uint16, codec Codec) (Tensor, error) {
	return compress(name, SplitBytePlanes(w), codec, LayoutBytePlanes, DefaultChunkSize, len(w))
}

// compress chunks data, restarting the chunking at every multiple of
// segment bytes.
func compress(name string, data []byte, codec Codec, layout Layout, chunkSize, segment int) (Tensor, error) {
	if chunkSize <= 0 || chunkSize > 1<<30 {
		return Tensor{}, fmt.Errorf("cudadecomp: chunk size %d out of range", chunkSize)
	}
	if codec > CodecDeflate {
		return Tensor{}, fmt.Errorf("cudadecomp: unknown codec %v", codec)
	}
	t := Tensor{Name: name, Codec: codec, Layout: layout, ChunkSize: chunkSize, Size: len(data)}
	var buf bytes.Buffer
	for seg := 0; seg < len(data); seg += segment {
		end := min(seg+segment, len(data))
		for off := seg; off < end; off += chunkSize {
			raw := data[off:min(off+chunkSize, end)]
			c := Chunk{
				Codec:     codec,
				SrcOffset: uint64(len(t.Payload)),
				DstOffset: uint64(off),
				DstSize:   uint32(len(raw)),
				CRC32:     crc32.Checksum(raw, castagnoli),
			}
			var enc []byte
			switch codec {
			case CodecLZ4:
				enc = lz4Compress(nil, raw)
			case CodecDeflate:
				buf.Reset()
				w, err := flate.NewWriter(&buf, flate.BestCompression)
				if err != nil {
					return Tensor{}, err
				}
				w.Write(raw)
				if err := w.Close(); err != nil {
					return Tensor{}, fmt.Errorf("cudadecomp: deflate: %w", err)
				}
				enc = buf.Bytes()
			}
			if codec == CodecStored || len(enc) >= len(raw) {
				c.Codec, enc = CodecStored, raw
			}
			c.SrcSize = uint32(len(enc))
			t.Payload = append(t.Payload, enc...)
			t.Chunks = append(t.Chunks, c)
		}
	}
	return t, nil
}

// DecompressChunk decodes chunk c of t's payload into dst, which must hold
// exactly c.DstSize bytes, and checks its CRC.
func (t Tensor) DecompressChunk(dst []byte, c Chunk) error {
	end := c.SrcOffset + uint64(c.SrcSize)
	if end > uint64(len(t.Payload)) || end < c.SrcOffset {
		return fmt.Errorf("%w: payload range [%d, %d) beyond %d bytes", ErrChunkCorrupted, c.SrcOffset, end, len(t.Payload))
	}
	return decodeChunk(dst, t.Payload[c.SrcOffset:end], c)
}

// decodeChunk decodes src, the compressed bytes of c, into dst.
func decodeChunk(dst, src []byte, c Chunk) error {
	if len(dst) != int(c.DstSize) {
		return fmt.Errorf("cudadecomp: destination holds %d bytes, chunk %d", len(dst), c.DstSize)
	}
	switch c.Codec {
	case CodecStored:
		if len(src) != len(dst) {
			return fmt.Errorf("%w: stored chunk of %d bytes, want %d", ErrChunkCorrupted, len(src), len(dst))
		}
		copy(dst, src)
	case CodecLZ4:
		if err := lz4Decompress(dst, src); err != nil {
			return fmt.Errorf("%w: %v", ErrChunkCorrupted, err)
		}
	case CodecDeflate:
		r := flate.NewReader(bytes.NewReader(src))
		if _, err := io.ReadFull(r, dst); err != nil {
			return fmt.Errorf("%w: deflate: %v", ErrChunkCorrupted, err)
		}
		var b [1]byte
		if n, _ := r.Read(b[:]); n != 0 {
			return fmt.Errorf("%w: deflate: trailing data", ErrChunkCorrupted)
		}
	default:
		return fmt.Errorf("%w: unknown codec %v", ErrChunkCorrupted, c.Codec)
	}
	if got := crc32.Checksum(dst, castagnoli); got != c.CRC32 {
		return fmt.Errorf("%w: at offset %d: crc %#08x, want %#08x", ErrChunkCorrupted, c.DstOffset, got, c.CRC32)
	}
	return nil
}

// Decompress decodes every chunk of t and returns the uncompressed bytes,
// in t's layout. It checks ctx between chunks.
func (t Tensor) Decompress(ctx context.Context) ([]byte, error) {
	out := make([]byte, t.Size)
	covered := 0
	for i, c := range t.Chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := c.DstOffset + uint64(c.DstSize)
		if end > uint64(len(out)) || end < c.DstOffset {
			return nil, fmt.Errorf("%w: chunk %d: destination beyond %d bytes", ErrChunkCorrupted, i, len(out))
		}
		if err := t.DecompressChunk(out[c.DstOffset:end], c); err != nil {
			return nil, fmt.Errorf("chunk %d: %w", i, err)
		}
		covered += int(c.DstSize)
	}
	if covered != t.Size {
		return nil, fmt.Errorf("%w: chunks cover %d of %d bytes", ErrChunkCorrupted, covered, t.Size)
	}
	return out, nil
}

// DecompressBF16 decodes a tensor made by [CompressBF16] and returns the
// original weights.
func DecompressBF16(ctx context.Context, t Tensor) ([]uint16, error) {
	if t.Layout != LayoutBytePlanes || t.Size%2 != 0 {
		return nil, fmt.Errorf("cudadecomp: tensor %q is not BF16 byte planes", t.Name)
	}
	planes, err := t.Decompress(ctx)
	if err != nil {
		return nil, err
	}
	return MergeBytePlanes(planes), nil
}

// SplitBytePlanes returns the high bytes of w followed by the low bytes.
func SplitBytePlanes(w []uint16) []byte {
	p := make([]byte, 2*len(w))
	for i, v := range w {
		p[i] = byte(v >> 8)
		p[len(w)+i] = byte(v)
	}
	return p
}

// MergeBytePlanes inverts [SplitBytePlanes].
func MergeBytePlanes(p []byte) []uint16 {
	n := len(p) / 2
	w := make([]uint16, n)
	for i := range w {
		w[i] = uint16(p[i])<<8 | uint16(p[n+i])
	}
	return w
}
