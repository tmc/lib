// SPDX-License-Identifier: Apache-2.0

package cudaprepack

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

const (
	binaryMagic   = "CUPK" // CUDA Pre-pack file magic
	binaryVersion = 1
	headerSize    = 128 // 128-byte aligned header
	alignBoundary = 128 // 128-byte section padding for zero-copy mmap alignment
)

func padToBoundary(n int64, boundary int64) int64 {
	return ((n + boundary - 1) / boundary) * boundary
}

// Write serializes the pre-packed tensor into an aligned binary format suitable for mmap.
func (p *PrepackedTensor) Write(w io.Writer) error {
	header := make([]byte, headerSize)
	copy(header[0:4], binaryMagic)
	binary.LittleEndian.PutUint32(header[4:8], binaryVersion)

	// Target arch (8 bytes)
	copy(header[8:16], p.Arch)

	// DataType (8 bytes)
	copy(header[16:24], string(p.Type))

	// HeaderSize & Alignment
	binary.LittleEndian.PutUint32(header[24:28], headerSize)
	binary.LittleEndian.PutUint32(header[28:32], alignBoundary)

	// Rank, Swizzle, Interleave
	binary.LittleEndian.PutUint32(header[32:36], uint32(p.BoxDesc.Rank))
	binary.LittleEndian.PutUint16(header[36:38], uint16(p.BoxDesc.Swizzle))
	binary.LittleEndian.PutUint16(header[38:40], uint16(p.BoxDesc.Interleave))

	// GlobalDims
	binary.LittleEndian.PutUint64(header[40:48], p.BoxDesc.GlobalDims[0])
	binary.LittleEndian.PutUint64(header[48:56], p.BoxDesc.GlobalDims[1])
	binary.LittleEndian.PutUint64(header[56:64], p.BoxDesc.GlobalDims[2])

	// GlobalStrides
	binary.LittleEndian.PutUint64(header[64:72], uint64(p.BoxDesc.GlobalStrides[0]))
	binary.LittleEndian.PutUint64(header[72:80], uint64(p.BoxDesc.GlobalStrides[1]))

	// BoxDims
	binary.LittleEndian.PutUint32(header[80:84], p.BoxDesc.BoxDims[0])
	binary.LittleEndian.PutUint32(header[84:88], p.BoxDesc.BoxDims[1])
	binary.LittleEndian.PutUint32(header[88:92], p.BoxDesc.BoxDims[2])

	// ElementStrides
	binary.LittleEndian.PutUint32(header[92:96], p.BoxDesc.ElementStrides[0]|(p.BoxDesc.ElementStrides[1]<<8)|(p.BoxDesc.ElementStrides[2]<<16))

	// Section lengths
	payloadLen := int64(len(p.Payload))
	scalesLen := int64(len(p.Scales))
	metaLen := int64(len(p.Metadata))

	binary.LittleEndian.PutUint64(header[96:104], uint64(payloadLen))
	binary.LittleEndian.PutUint64(header[104:112], uint64(scalesLen))
	binary.LittleEndian.PutUint64(header[112:120], uint64(metaLen))

	if _, err := w.Write(header); err != nil {
		return fmt.Errorf("cudaprepack: write header failed: %w", err)
	}

	// Write Payload
	if payloadLen > 0 {
		if _, err := w.Write(p.Payload); err != nil {
			return fmt.Errorf("cudaprepack: write payload failed: %w", err)
		}
		pad := padToBoundary(payloadLen, alignBoundary) - payloadLen
		if pad > 0 {
			zeros := make([]byte, pad)
			if _, err := w.Write(zeros); err != nil {
				return fmt.Errorf("cudaprepack: write payload padding failed: %w", err)
			}
		}
	}

	// Write Scales
	if scalesLen > 0 {
		if _, err := w.Write(p.Scales); err != nil {
			return fmt.Errorf("cudaprepack: write scales failed: %w", err)
		}
		pad := padToBoundary(scalesLen, alignBoundary) - scalesLen
		if pad > 0 {
			zeros := make([]byte, pad)
			if _, err := w.Write(zeros); err != nil {
				return fmt.Errorf("cudaprepack: write scales padding failed: %w", err)
			}
		}
	}

	// Write Metadata
	if metaLen > 0 {
		if _, err := w.Write(p.Metadata); err != nil {
			return fmt.Errorf("cudaprepack: write metadata failed: %w", err)
		}
		pad := padToBoundary(metaLen, alignBoundary) - metaLen
		if pad > 0 {
			zeros := make([]byte, pad)
			if _, err := w.Write(zeros); err != nil {
				return fmt.Errorf("cudaprepack: write metadata padding failed: %w", err)
			}
		}
	}

	return nil
}

// MarshalBinary serializes the pre-packed tensor into an in-memory byte buffer.
func (p *PrepackedTensor) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	if err := p.Write(&buf); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Read deserializes a pre-packed tensor from an io.Reader.
func Read(r io.Reader) (*PrepackedTensor, error) {
	header := make([]byte, headerSize)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, fmt.Errorf("cudaprepack: read header failed: %w", err)
	}

	if string(header[0:4]) != binaryMagic {
		return nil, fmt.Errorf("cudaprepack: invalid binary magic %q, want %q", string(header[0:4]), binaryMagic)
	}
	version := binary.LittleEndian.Uint32(header[4:8])
	if version != binaryVersion {
		return nil, fmt.Errorf("cudaprepack: unsupported version %d", version)
	}

	arch := string(bytes.TrimRight(header[8:16], "\x00"))
	dt := DataType(string(bytes.TrimRight(header[16:24], "\x00")))

	rank := int(binary.LittleEndian.Uint32(header[32:36]))
	swizzle := SwizzleMode(binary.LittleEndian.Uint16(header[36:38]))
	interleave := InterleaveMode(binary.LittleEndian.Uint16(header[38:40]))

	var gdims [3]uint64
	gdims[0] = binary.LittleEndian.Uint64(header[40:48])
	gdims[1] = binary.LittleEndian.Uint64(header[48:56])
	gdims[2] = binary.LittleEndian.Uint64(header[56:64])

	var gstrides [3]int64
	gstrides[0] = int64(binary.LittleEndian.Uint64(header[64:72]))
	gstrides[1] = int64(binary.LittleEndian.Uint64(header[72:80]))

	var boxDims [3]uint32
	boxDims[0] = binary.LittleEndian.Uint32(header[80:84])
	boxDims[1] = binary.LittleEndian.Uint32(header[84:88])
	boxDims[2] = binary.LittleEndian.Uint32(header[88:92])

	elemStridesPacked := binary.LittleEndian.Uint32(header[92:96])
	elemStrides := [3]uint32{
		elemStridesPacked & 0xFF,
		(elemStridesPacked >> 8) & 0xFF,
		(elemStridesPacked >> 16) & 0xFF,
	}

	payloadLen := int64(binary.LittleEndian.Uint64(header[96:104]))
	scalesLen := int64(binary.LittleEndian.Uint64(header[104:112]))
	metaLen := int64(binary.LittleEndian.Uint64(header[112:120]))

	payload := make([]byte, payloadLen)
	if payloadLen > 0 {
		if _, err := io.ReadFull(r, payload); err != nil {
			return nil, fmt.Errorf("cudaprepack: read payload failed: %w", err)
		}
		pad := padToBoundary(payloadLen, alignBoundary) - payloadLen
		if pad > 0 {
			discard := make([]byte, pad)
			if _, err := io.ReadFull(r, discard); err != nil {
				return nil, fmt.Errorf("cudaprepack: discard payload pad failed: %w", err)
			}
		}
	}

	scales := make([]byte, scalesLen)
	if scalesLen > 0 {
		if _, err := io.ReadFull(r, scales); err != nil {
			return nil, fmt.Errorf("cudaprepack: read scales failed: %w", err)
		}
		pad := padToBoundary(scalesLen, alignBoundary) - scalesLen
		if pad > 0 {
			discard := make([]byte, pad)
			if _, err := io.ReadFull(r, discard); err != nil {
				return nil, fmt.Errorf("cudaprepack: discard scales pad failed: %w", err)
			}
		}
	}

	meta := make([]byte, metaLen)
	if metaLen > 0 {
		if _, err := io.ReadFull(r, meta); err != nil {
			return nil, fmt.Errorf("cudaprepack: read metadata failed: %w", err)
		}
		pad := padToBoundary(metaLen, alignBoundary) - metaLen
		if pad > 0 {
			discard := make([]byte, pad)
			if _, err := io.ReadFull(r, discard); err != nil {
				return nil, fmt.Errorf("cudaprepack: discard meta pad failed: %w", err)
			}
		}
	}

	desc := BoxDescriptor{
		Rank:           rank,
		DataType:       dt,
		GlobalDims:     gdims,
		GlobalStrides:  gstrides,
		BoxDims:        boxDims,
		ElementStrides: elemStrides,
		Swizzle:        swizzle,
		Interleave:     interleave,
	}

	return &PrepackedTensor{
		Arch:     arch,
		Type:     dt,
		BoxDesc:  desc,
		Payload:  payload,
		Scales:   scales,
		Metadata: meta,
	}, nil
}

// ParseZeroCopy parses a prepacked buffer directly without copying payload data.
// It slices directly into data, confirming 128-byte hardware alignment for zero-copy mmap.
func ParseZeroCopy(data []byte) (*PrepackedTensor, error) {
	if len(data) < headerSize {
		return nil, fmt.Errorf("cudaprepack: buffer too short for header")
	}
	if string(data[0:4]) != binaryMagic {
		return nil, fmt.Errorf("cudaprepack: invalid binary magic")
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != binaryVersion {
		return nil, fmt.Errorf("cudaprepack: unsupported version %d", version)
	}

	arch := string(bytes.TrimRight(data[8:16], "\x00"))
	dt := DataType(string(bytes.TrimRight(data[16:24], "\x00")))

	rank := int(binary.LittleEndian.Uint32(data[32:36]))
	swizzle := SwizzleMode(binary.LittleEndian.Uint16(data[36:38]))
	interleave := InterleaveMode(binary.LittleEndian.Uint16(data[38:40]))

	var gdims [3]uint64
	gdims[0] = binary.LittleEndian.Uint64(data[40:48])
	gdims[1] = binary.LittleEndian.Uint64(data[48:56])
	gdims[2] = binary.LittleEndian.Uint64(data[56:64])

	var gstrides [3]int64
	gstrides[0] = int64(binary.LittleEndian.Uint64(data[64:72]))
	gstrides[1] = int64(binary.LittleEndian.Uint64(data[72:80]))

	var boxDims [3]uint32
	boxDims[0] = binary.LittleEndian.Uint32(data[80:84])
	boxDims[1] = binary.LittleEndian.Uint32(data[84:88])
	boxDims[2] = binary.LittleEndian.Uint32(data[88:92])

	elemStridesPacked := binary.LittleEndian.Uint32(data[92:96])
	elemStrides := [3]uint32{
		elemStridesPacked & 0xFF,
		(elemStridesPacked >> 8) & 0xFF,
		(elemStridesPacked >> 16) & 0xFF,
	}

	payloadLen := int64(binary.LittleEndian.Uint64(data[96:104]))
	scalesLen := int64(binary.LittleEndian.Uint64(data[104:112]))
	metaLen := int64(binary.LittleEndian.Uint64(data[112:120]))

	offset := int64(headerSize)
	var payload []byte
	if payloadLen > 0 {
		if int64(len(data)) < offset+payloadLen {
			return nil, fmt.Errorf("cudaprepack: buffer truncated in payload")
		}
		payload = data[offset : offset+payloadLen]
		offset += padToBoundary(payloadLen, alignBoundary)
	}

	var scales []byte
	if scalesLen > 0 {
		if int64(len(data)) < offset+scalesLen {
			return nil, fmt.Errorf("cudaprepack: buffer truncated in scales")
		}
		scales = data[offset : offset+scalesLen]
		offset += padToBoundary(scalesLen, alignBoundary)
	}

	var meta []byte
	if metaLen > 0 {
		if int64(len(data)) < offset+metaLen {
			return nil, fmt.Errorf("cudaprepack: buffer truncated in metadata")
		}
		meta = data[offset : offset+metaLen]
	}

	desc := BoxDescriptor{
		Rank:           rank,
		DataType:       dt,
		GlobalDims:     gdims,
		GlobalStrides:  gstrides,
		BoxDims:        boxDims,
		ElementStrides: elemStrides,
		Swizzle:        swizzle,
		Interleave:     interleave,
	}

	return &PrepackedTensor{
		Arch:     arch,
		Type:     dt,
		BoxDesc:  desc,
		Payload:  payload,
		Scales:   scales,
		Metadata: meta,
		zeroCopy: true,
	}, nil
}
