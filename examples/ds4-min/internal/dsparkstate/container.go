package dsparkstate

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

var containerMagic = [8]byte{'D', 'S', '4', 'S', 'U', 'P', 'S', '1'}

// The container preserves the existing platform core payload byte-for-byte.
// Only CUDA stores a core capture generation; Metal has no such field.
func WriteContainer(dst io.Writer, core []byte, support *Envelope, frontier int, generation uint64, maximum uint64) error {
	if dst == nil || len(core) == 0 {
		return errors.New("dspark support container: nil writer or empty core")
	}
	var encoded bytes.Buffer
	if err := Write(&encoded, support, frontier, generation); err != nil {
		return err
	}
	kind := uint8(2)
	header := uint64(25)
	if support.Capture != nil {
		kind = 1
		header += 8
	}
	if uint64(len(core)) > maximum || uint64(encoded.Len()) > maximum-uint64(len(core)) || header > maximum-uint64(len(core))-uint64(encoded.Len()) {
		return errors.New("dspark support container: payload exceeds limit")
	}
	w := writer{w: dst}
	w.value(containerMagic)
	if support.Platform != 0 {
		kind = support.Platform
	}
	w.value(kind)
	if kind == 1 {
		w.value(generation)
	}
	w.value(uint64(len(core)))
	w.value(uint64(encoded.Len()))
	w.value(core)
	w.value(encoded.Bytes())
	if w.err != nil {
		return fmt.Errorf("dspark support container: write: %w", w.err)
	}
	return nil
}

func ReadContainer(src io.Reader, platform uint8, maximum uint64) (core, support []byte, generation uint64, err error) {
	if src == nil || maximum == 0 || maximum >= uint64(^uint64(0)>>1) {
		return nil, nil, 0, errors.New("dspark support container: invalid reader or limit")
	}
	data, err := io.ReadAll(io.LimitReader(src, int64(maximum)+1))
	if err != nil {
		return nil, nil, 0, fmt.Errorf("dspark support container: read: %w", err)
	}
	if uint64(len(data)) > maximum {
		return nil, nil, 0, errors.New("dspark support container: payload exceeds limit")
	}
	r := bytes.NewReader(data)
	var magic [8]byte
	var kind uint8
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, nil, 0, err
	}
	if err := binary.Read(r, binary.LittleEndian, &kind); err != nil {
		return nil, nil, 0, err
	}
	want := platform
	if want < 1 || want > 3 {
		return nil, nil, 0, errors.New("dspark support container: invalid platform")
	}
	if magic != containerMagic || kind != want {
		return nil, nil, 0, errors.New("dspark support container: missing or foreign support")
	}
	if kind == 1 {
		if err := binary.Read(r, binary.LittleEndian, &generation); err != nil {
			return nil, nil, 0, err
		}
	}
	var coreBytes, supportBytes uint64
	for _, value := range []any{&coreBytes, &supportBytes} {
		if err := binary.Read(r, binary.LittleEndian, value); err != nil {
			return nil, nil, 0, err
		}
	}
	remaining := uint64(r.Len())
	if coreBytes == 0 || supportBytes == 0 || coreBytes > remaining || supportBytes != remaining-coreBytes {
		return nil, nil, 0, errors.New("dspark support container: invalid section lengths")
	}
	// Core and support slices belong to this owned read buffer, not the input
	// reader. Destination mutation still requires both platform validators.
	start := len(data) - r.Len()
	return data[start : start+int(coreBytes)], data[start+int(coreBytes):], generation, nil
}
