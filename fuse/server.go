package fuse

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"strings"
)

const (
	errEPERM  int32 = 1
	errEIO    int32 = 5
	errENOENT int32 = 2
	errEINVAL int32 = 22
)

func Serve(ctx context.Context, rw io.ReadWriter, fs FileSystem) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if rw == nil || fs == nil {
		return ErrInvalidArgument
	}
	if closer, ok := rw.(io.Closer); ok {
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				_ = closer.Close()
			case <-done:
			}
		}()
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		frame, err := ReadFrame(rw)
		if err != nil {
			return err
		}
		resp, err := HandleFrame(ctx, fs, frame)
		if err != nil {
			return err
		}
		if _, err := rw.Write(resp); err != nil {
			return err
		}
	}
}

func HandleFrame(ctx context.Context, fs FileSystem, frame Frame) ([]byte, error) {
	h := frame.Header
	switch h.Opcode {
	case OpcodeInit:
		return EncodeResponse(h.Unique, 0, encodeInitOut()), nil
	case OpcodeLookup:
		name := strings.TrimRight(string(frame.Data), "\x00")
		entry, err := fs.Lookup(ctx, h.NodeID, name)
		if err != nil {
			return EncodeResponse(h.Unique, errno(err), nil), nil
		}
		return EncodeResponse(h.Unique, 0, EncodeEntry(entry)), nil
	case OpcodeGetattr:
		attr, err := fs.Getattr(ctx, h.NodeID)
		if err != nil {
			return EncodeResponse(h.Unique, errno(err), nil), nil
		}
		return EncodeResponse(h.Unique, 0, EncodeAttr(attr)), nil
	case OpcodeOpen, OpcodeOpendir:
		var flags uint32
		if len(frame.Data) >= 4 {
			flags = binary.LittleEndian.Uint32(frame.Data[:4])
		}
		fh, err := fs.Open(ctx, h.NodeID, flags)
		if err != nil {
			return EncodeResponse(h.Unique, errno(err), nil), nil
		}
		var out bytes.Buffer
		_ = binary.Write(&out, binary.LittleEndian, uint64(fh))
		_ = binary.Write(&out, binary.LittleEndian, uint32(0))
		_ = binary.Write(&out, binary.LittleEndian, uint32(0))
		return EncodeResponse(h.Unique, 0, out.Bytes()), nil
	case OpcodeRead, OpcodeReaddir:
		fh, off, size, ok := parseRead(frame.Data)
		if !ok {
			return EncodeResponse(h.Unique, errEINVAL, nil), nil
		}
		if h.Opcode == OpcodeReaddir {
			dirents, err := fs.Readdir(ctx, h.NodeID, fh, off, size)
			if err != nil {
				return EncodeResponse(h.Unique, errno(err), nil), nil
			}
			return EncodeResponse(h.Unique, 0, EncodeDirents(dirents)), nil
		}
		data, err := fs.Read(ctx, h.NodeID, fh, off, size)
		if err != nil {
			return EncodeResponse(h.Unique, errno(err), nil), nil
		}
		return EncodeResponse(h.Unique, 0, data), nil
	case OpcodeWrite:
		fh, off, size, data, ok := parseWrite(frame.Data)
		if !ok {
			return EncodeResponse(h.Unique, errEINVAL, nil), nil
		}
		if uint32(len(data)) != size {
			return EncodeResponse(h.Unique, errEINVAL, nil), nil
		}
		n, err := fs.Write(ctx, h.NodeID, fh, off, data)
		if err != nil {
			return EncodeResponse(h.Unique, errno(err), nil), nil
		}
		var out bytes.Buffer
		_ = binary.Write(&out, binary.LittleEndian, n)
		_ = binary.Write(&out, binary.LittleEndian, uint32(0))
		return EncodeResponse(h.Unique, 0, out.Bytes()), nil
	case OpcodeFlush:
		fh := parseHandle(frame.Data)
		return EncodeResponse(h.Unique, errno(fs.Flush(ctx, h.NodeID, fh)), nil), nil
	case OpcodeFsync:
		fh := parseHandle(frame.Data)
		return EncodeResponse(h.Unique, errno(fs.Fsync(ctx, h.NodeID, fh)), nil), nil
	case OpcodeRelease:
		fh := parseHandle(frame.Data)
		return EncodeResponse(h.Unique, errno(fs.Release(ctx, h.NodeID, fh)), nil), nil
	default:
		return EncodeResponse(h.Unique, errEINVAL, nil), nil
	}
}

func parseHandle(data []byte) HandleID {
	if len(data) < 8 {
		return 0
	}
	return HandleID(binary.LittleEndian.Uint64(data[:8]))
}

func parseRead(data []byte) (HandleID, uint64, uint32, bool) {
	if len(data) < 24 {
		return 0, 0, 0, false
	}
	return HandleID(binary.LittleEndian.Uint64(data[0:8])), binary.LittleEndian.Uint64(data[8:16]), binary.LittleEndian.Uint32(data[16:20]), true
}

func parseWrite(data []byte) (HandleID, uint64, uint32, []byte, bool) {
	if len(data) < 24 {
		return 0, 0, 0, nil, false
	}
	fh := HandleID(binary.LittleEndian.Uint64(data[0:8]))
	off := binary.LittleEndian.Uint64(data[8:16])
	size := binary.LittleEndian.Uint32(data[16:20])
	return fh, off, size, data[24:], true
}

func encodeInitOut() []byte {
	var out bytes.Buffer
	for _, v := range []any{uint32(7), uint32(38), uint32(0), uint32(0), uint32(maxFrameSize), uint32(0), uint32(0), uint32(0)} {
		_ = binary.Write(&out, binary.LittleEndian, v)
	}
	return out.Bytes()
}

func errno(err error) int32 {
	switch err {
	case nil:
		return 0
	case ErrNotExist:
		return errENOENT
	case ErrInvalidArgument:
		return errEINVAL
	case ErrReadOnly:
		return errEPERM
	default:
		return errEIO
	}
}
