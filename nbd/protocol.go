package nbd

import (
	"encoding/binary"
	"fmt"
	"io"
)

const (
	oldMagic         uint64 = 0x4e42444d41474943
	optionMagic      uint64 = 0x49484156454f5054
	requestMagic     uint32 = 0x25609513
	simpleReplyMagic uint32 = 0x67446698

	optionExportName uint32 = 1

	flagFixedNewstyle uint16 = 1 << 0
	flagNoZeroes      uint16 = 1 << 1

	transmissionFlagHasFlags  uint16 = 1 << 0
	transmissionFlagReadOnly  uint16 = 1 << 1
	transmissionFlagSendFlush uint16 = 1 << 2
	transmissionFlagSendTrim  uint16 = 1 << 5

	errEPERM  uint32 = 1
	errEIO    uint32 = 5
	errEINVAL uint32 = 22
	errENOSPC uint32 = 28

	maxPayload = 32 << 20
	maxInt64   = uint64(1<<63 - 1)
)

// Command is an NBD transmission command.
type Command uint16

const (
	CommandRead  Command = 0
	CommandWrite Command = 1
	CommandDisc  Command = 2
	CommandFlush Command = 3
	CommandTrim  Command = 4
)

// Request is one NBD transmission request.
type Request struct {
	Flags  uint16
	Type   Command
	Handle uint64
	Offset uint64
	Length uint32
}

func writeHandshake(rw io.ReadWriter, exportName string, size uint64, flags uint16) error {
	if err := binary.Write(rw, binary.BigEndian, oldMagic); err != nil {
		return fmt.Errorf("write nbd old magic: %w", err)
	}
	if err := binary.Write(rw, binary.BigEndian, optionMagic); err != nil {
		return fmt.Errorf("write nbd option magic: %w", err)
	}
	if err := binary.Write(rw, binary.BigEndian, flagFixedNewstyle|flagNoZeroes); err != nil {
		return fmt.Errorf("write nbd handshake flags: %w", err)
	}
	var clientFlags uint32
	if err := binary.Read(rw, binary.BigEndian, &clientFlags); err != nil {
		return fmt.Errorf("read nbd client flags: %w", err)
	}
	for {
		var magic uint64
		var option uint32
		var length uint32
		if err := binary.Read(rw, binary.BigEndian, &magic); err != nil {
			return fmt.Errorf("read nbd option magic: %w", err)
		}
		if magic != optionMagic {
			return fmt.Errorf("%w: option magic %#x", ErrInvalidArgument, magic)
		}
		if err := binary.Read(rw, binary.BigEndian, &option); err != nil {
			return err
		}
		if err := binary.Read(rw, binary.BigEndian, &length); err != nil {
			return err
		}
		payload := make([]byte, length)
		if _, err := io.ReadFull(rw, payload); err != nil {
			return fmt.Errorf("read nbd option payload: %w", err)
		}
		if option != optionExportName {
			return fmt.Errorf("%w: option %d", ErrUnsupported, option)
		}
		if string(payload) != exportName {
			return fmt.Errorf("%w: export %q", ErrInvalidArgument, payload)
		}
		if err := binary.Write(rw, binary.BigEndian, size); err != nil {
			return fmt.Errorf("write nbd export size: %w", err)
		}
		if err := binary.Write(rw, binary.BigEndian, flags); err != nil {
			return fmt.Errorf("write nbd export flags: %w", err)
		}
		if clientFlags&uint32(flagNoZeroes) != 0 {
			return nil
		}
		_, err := rw.Write(make([]byte, 124))
		return err
	}
}

func readHandshake(rw io.ReadWriter, exportName string) (uint64, uint16, error) {
	var magic uint64
	var opts uint64
	var flags uint16
	if err := binary.Read(rw, binary.BigEndian, &magic); err != nil {
		return 0, 0, fmt.Errorf("read nbd old magic: %w", err)
	}
	if magic != oldMagic {
		return 0, 0, fmt.Errorf("%w: old magic %#x", ErrInvalidArgument, magic)
	}
	if err := binary.Read(rw, binary.BigEndian, &opts); err != nil {
		return 0, 0, err
	}
	if opts != optionMagic {
		return 0, 0, fmt.Errorf("%w: option magic %#x", ErrInvalidArgument, opts)
	}
	if err := binary.Read(rw, binary.BigEndian, &flags); err != nil {
		return 0, 0, err
	}
	if err := binary.Write(rw, binary.BigEndian, uint32(0)); err != nil {
		return 0, 0, err
	}
	if err := binary.Write(rw, binary.BigEndian, optionMagic); err != nil {
		return 0, 0, err
	}
	if err := binary.Write(rw, binary.BigEndian, optionExportName); err != nil {
		return 0, 0, err
	}
	if err := binary.Write(rw, binary.BigEndian, uint32(len(exportName))); err != nil {
		return 0, 0, err
	}
	if _, err := io.WriteString(rw, exportName); err != nil {
		return 0, 0, err
	}
	var size uint64
	var exportFlags uint16
	if err := binary.Read(rw, binary.BigEndian, &size); err != nil {
		return 0, 0, err
	}
	if err := binary.Read(rw, binary.BigEndian, &exportFlags); err != nil {
		return 0, 0, err
	}
	if _, err := io.CopyN(io.Discard, rw, 124); err != nil {
		return 0, 0, err
	}
	return size, exportFlags, nil
}

func readRequest(r io.Reader) (Request, error) {
	var magic uint32
	if err := binary.Read(r, binary.BigEndian, &magic); err != nil {
		return Request{}, err
	}
	if magic != requestMagic {
		return Request{}, fmt.Errorf("%w: request magic %#x", ErrInvalidArgument, magic)
	}
	var req Request
	if err := binary.Read(r, binary.BigEndian, &req.Flags); err != nil {
		return Request{}, err
	}
	var typ uint16
	if err := binary.Read(r, binary.BigEndian, &typ); err != nil {
		return Request{}, err
	}
	req.Type = Command(typ)
	if err := binary.Read(r, binary.BigEndian, &req.Handle); err != nil {
		return Request{}, err
	}
	if err := binary.Read(r, binary.BigEndian, &req.Offset); err != nil {
		return Request{}, err
	}
	if err := binary.Read(r, binary.BigEndian, &req.Length); err != nil {
		return Request{}, err
	}
	return req, nil
}

func writeRequest(w io.Writer, req Request) error {
	for _, v := range []any{requestMagic, req.Flags, uint16(req.Type), req.Handle, req.Offset, req.Length} {
		if err := binary.Write(w, binary.BigEndian, v); err != nil {
			return err
		}
	}
	return nil
}

func writeReply(w io.Writer, handle uint64, errno uint32, data []byte) error {
	for _, v := range []any{simpleReplyMagic, errno, handle} {
		if err := binary.Write(w, binary.BigEndian, v); err != nil {
			return err
		}
	}
	if len(data) != 0 {
		_, err := w.Write(data)
		return err
	}
	return nil
}

func readReply(r io.Reader, handle uint64, data []byte) error {
	var magic uint32
	var errno uint32
	var got uint64
	for _, p := range []any{&magic, &errno, &got} {
		if err := binary.Read(r, binary.BigEndian, p); err != nil {
			return err
		}
	}
	if magic != simpleReplyMagic {
		return fmt.Errorf("%w: reply magic %#x", ErrInvalidArgument, magic)
	}
	if got != handle {
		return fmt.Errorf("%w: reply handle %d", ErrInvalidArgument, got)
	}
	if errno != 0 {
		return fmt.Errorf("nbd: remote error %d", errno)
	}
	if len(data) != 0 {
		_, err := io.ReadFull(r, data)
		return err
	}
	return nil
}
