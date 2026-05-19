package fuse

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

const (
	RootID NodeID = 1

	maxFrameSize = 32 << 20
)

type NodeID uint64
type HandleID uint64
type Opcode uint32

const (
	OpcodeLookup  Opcode = 1
	OpcodeGetattr Opcode = 3
	OpcodeSetattr Opcode = 4
	OpcodeOpen    Opcode = 14
	OpcodeRead    Opcode = 15
	OpcodeWrite   Opcode = 16
	OpcodeRelease Opcode = 18
	OpcodeFsync   Opcode = 20
	OpcodeFlush   Opcode = 25
	OpcodeInit    Opcode = 26
	OpcodeOpendir Opcode = 27
	OpcodeReaddir Opcode = 28
)

type Frame struct {
	Header InHeader
	Data   []byte
}

type InHeader struct {
	Len     uint32
	Opcode  Opcode
	Unique  uint64
	NodeID  NodeID
	UID     uint32
	GID     uint32
	PID     uint32
	Padding uint32
}

type OutHeader struct {
	Len    uint32
	Error  int32
	Unique uint64
}

type Attr struct {
	Ino       uint64
	Size      uint64
	Blocks    uint64
	Atime     uint64
	Mtime     uint64
	Ctime     uint64
	AtimeNsec uint32
	MtimeNsec uint32
	CtimeNsec uint32
	Mode      uint32
	Nlink     uint32
	UID       uint32
	GID       uint32
	Rdev      uint32
	Blksize   uint32
	Padding   uint32
}

type Entry struct {
	NodeID NodeID
	Attr   Attr
}

type Setattr struct {
	Mode  uint32
	Size  uint64
	Valid uint32
}

type Dirent struct {
	Ino  uint64
	Off  uint64
	Type uint32
	Name string
}

func ReadFrame(r io.Reader) (Frame, error) {
	var h InHeader
	if err := binary.Read(r, binary.LittleEndian, &h.Len); err != nil {
		return Frame{}, err
	}
	if h.Len < 40 || h.Len > maxFrameSize {
		return Frame{}, fmt.Errorf("%w: frame length %d", ErrInvalidArgument, h.Len)
	}
	rest := make([]byte, 36)
	if _, err := io.ReadFull(r, rest); err != nil {
		return Frame{}, err
	}
	buf := bytes.NewReader(rest)
	for _, p := range []any{&h.Opcode, &h.Unique, &h.NodeID, &h.UID, &h.GID, &h.PID, &h.Padding} {
		if err := binary.Read(buf, binary.LittleEndian, p); err != nil {
			return Frame{}, err
		}
	}
	data := make([]byte, int(h.Len)-40)
	if _, err := io.ReadFull(r, data); err != nil {
		return Frame{}, err
	}
	return Frame{Header: h, Data: data}, nil
}

func WriteRequest(w io.Writer, h InHeader, data []byte) error {
	h.Len = uint32(40 + len(data))
	if h.Len > maxFrameSize {
		return ErrInvalidArgument
	}
	if err := binary.Write(w, binary.LittleEndian, h.Len); err != nil {
		return err
	}
	for _, v := range []any{h.Opcode, h.Unique, h.NodeID, h.UID, h.GID, h.PID, h.Padding} {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	_, err := w.Write(data)
	return err
}

func EncodeResponse(unique uint64, errno int32, payload []byte) []byte {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, OutHeader{Len: uint32(16 + len(payload)), Error: -errno, Unique: unique})
	buf.Write(payload)
	return buf.Bytes()
}

func WriteResponse(w io.Writer, unique uint64, errno int32, payload []byte) error {
	_, err := w.Write(EncodeResponse(unique, errno, payload))
	return err
}

func ReadResponse(r io.Reader) (OutHeader, []byte, error) {
	var h OutHeader
	if err := binary.Read(r, binary.LittleEndian, &h); err != nil {
		return h, nil, err
	}
	if h.Len < 16 || h.Len > maxFrameSize {
		return h, nil, ErrInvalidArgument
	}
	data := make([]byte, int(h.Len)-16)
	if _, err := io.ReadFull(r, data); err != nil {
		return h, nil, err
	}
	return h, data, nil
}

func EncodeAttr(a Attr) []byte {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(0))
	_ = binary.Write(&buf, binary.LittleEndian, a)
	return buf.Bytes()
}

func EncodeEntry(e Entry) []byte {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(e.NodeID))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint64(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(0))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(0))
	_ = binary.Write(&buf, binary.LittleEndian, e.Attr)
	return buf.Bytes()
}

func EncodeDirents(dirents []Dirent) []byte {
	var buf bytes.Buffer
	for _, d := range dirents {
		_ = binary.Write(&buf, binary.LittleEndian, d.Ino)
		_ = binary.Write(&buf, binary.LittleEndian, d.Off)
		_ = binary.Write(&buf, binary.LittleEndian, uint32(len(d.Name)))
		_ = binary.Write(&buf, binary.LittleEndian, d.Type)
		buf.WriteString(d.Name)
		for buf.Len()%8 != 0 {
			buf.WriteByte(0)
		}
	}
	return buf.Bytes()
}
