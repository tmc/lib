package bpfload

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
)

// BTFType describes one raw BTF type. SizeOrType is interpreted according to
// Kind (the upstream BTF_KIND identifier). BTF names do not prove registration
// as struct_ops or kfuncs, and do not authorize calls from a BPF program.
type BTFType struct {
	ArrayElement, ArrayIndex, ArrayCount uint32
	ID                                   uint32
	Name                                 string
	Kind                                 uint8
	KindFlag                             bool
	SizeOrType                           uint32
	Members                              []BTFMember
}

// BTFMember describes a struct/union field, function parameter, or datasec
// entry. BitOffset and BitfieldSize apply only to struct/union members.
type BTFMember struct {
	Name         string
	Type         uint32
	BitOffset    uint32
	BitfieldSize uint8
}

// BTF contains validated raw BTF type records. Discovery of a function or a
// structure is metadata only; upstream attachment admission remains separate.
type BTF struct{ types []BTFType }

// ParseBTF parses a bounded raw BTF blob in either byte order. It validates
// record boundaries, string offsets and type references, and rejects kinds it
// does not understand. ELF containers must be extracted before calling it.
func ParseBTF(data []byte) (*BTF, error) {
	if len(data) < 24 || len(data) > 64<<20 {
		return nil, errors.New("bpf btf: invalid size")
	}
	var order binary.ByteOrder
	switch {
	case binary.LittleEndian.Uint16(data) == 0xeb9f:
		order = binary.LittleEndian
	case binary.BigEndian.Uint16(data) == 0xeb9f:
		order = binary.BigEndian
	default:
		return nil, errors.New("bpf btf: invalid magic")
	}
	if data[2] != 1 || data[3] != 0 {
		return nil, errors.New("bpf btf: unsupported version or flags")
	}
	hdr := uint64(order.Uint32(data[4:]))
	if hdr < 24 || hdr > uint64(len(data)) {
		return nil, errors.New("bpf btf: invalid header length")
	}
	typeStart, typeLength := uint64(order.Uint32(data[8:])), uint64(order.Uint32(data[12:]))
	stringStart, stringLength := uint64(order.Uint32(data[16:])), uint64(order.Uint32(data[20:]))
	if typeLength != 0 && stringLength != 0 && typeStart < stringStart+stringLength && stringStart < typeStart+typeLength {
		return nil, errors.New("bpf btf: overlapping type and string sections")
	}
	section := func(off, length uint32) ([]byte, error) {
		start := hdr + uint64(off)
		end := start + uint64(length)
		if end > uint64(len(data)) {
			return nil, errors.New("bpf btf: section out of bounds")
		}
		return data[start:end], nil
	}
	raw, err := section(order.Uint32(data[8:]), order.Uint32(data[12:]))
	if err != nil {
		return nil, err
	}
	strings, err := section(order.Uint32(data[16:]), order.Uint32(data[20:]))
	if err != nil {
		return nil, err
	}
	if len(strings) == 0 || strings[0] != 0 || strings[len(strings)-1] != 0 {
		return nil, errors.New("bpf btf: missing empty string")
	}
	cache := make(map[uint32]string)
	name := func(off uint32) (string, error) {
		if s, ok := cache[off]; ok {
			return s, nil
		}
		if uint64(off) >= uint64(len(strings)) {
			return "", errors.New("bpf btf: string offset out of bounds")
		}
		rest := strings[off:]
		n := bytes.IndexByte(rest, 0)
		if n < 0 {
			return "", errors.New("bpf btf: unterminated string")
		}
		if n > 4096 {
			return "", errors.New("bpf btf: name exceeds 4096 bytes")
		}
		s := string(rest[:n])
		cache[off] = s
		return s, nil
	}
	b := &BTF{}
	var arrayReferences []uint32
	for len(raw) > 0 {
		if len(b.types) >= 1000000 {
			return nil, errors.New("bpf btf: too many types")
		}
		if len(raw) < 12 {
			return nil, errors.New("bpf btf: truncated type")
		}
		n, err := name(order.Uint32(raw))
		if err != nil {
			return nil, err
		}
		info := order.Uint32(raw[4:])
		count := int(info & 0xffff)
		kind := uint8(info >> 24 & 31)
		t := BTFType{ID: uint32(len(b.types) + 1), Name: n, Kind: kind, KindFlag: info>>31 != 0, SizeOrType: order.Uint32(raw[8:])}
		extra := 0
		switch kind {
		case 1:
			extra = 4
		case 2, 7, 8, 9, 10, 11, 12, 16, 18:
		case 3:
			extra = 12
		case 4, 5:
			extra = 12 * count
		case 6:
			extra = 8 * count
		case 13:
			extra = 8 * count
		case 14:
			extra = 4
		case 15:
			extra = 12 * count
		case 17:
			extra = 4
		case 19:
			extra = 12 * count
		default:
			return nil, fmt.Errorf("bpf btf: unsupported kind %d", kind)
		}
		if extra > len(raw)-12 {
			return nil, errors.New("bpf btf: truncated type data")
		}
		tail := raw[12 : 12+extra]
		if kind == 3 {
			t.ArrayElement = order.Uint32(tail)
			t.ArrayIndex = order.Uint32(tail[4:])
			t.ArrayCount = order.Uint32(tail[8:])
			arrayReferences = append(arrayReferences, order.Uint32(tail), order.Uint32(tail[4:]))
		}
		if kind == 6 || kind == 19 {
			stride := 8
			if kind == 19 {
				stride = 12
			}
			for i := 0; i < count; i++ {
				if _, err := name(order.Uint32(tail[i*stride:])); err != nil {
					return nil, err
				}
			}
		}
		if kind == 4 || kind == 5 || kind == 13 || kind == 15 {
			stride := 12
			if kind == 13 {
				stride = 8
			}
			for i := 0; i < count; i++ {
				p := tail[i*stride:]
				m := BTFMember{}
				if kind == 15 {
					m.Type = order.Uint32(p)
					m.BitOffset = order.Uint32(p[4:]) * 8
				} else {
					m.Name, err = name(order.Uint32(p))
					if err != nil {
						return nil, err
					}
					m.Type = order.Uint32(p[4:])
					if kind != 13 {
						m.BitOffset = order.Uint32(p[8:])
						if t.KindFlag {
							m.BitfieldSize = uint8(m.BitOffset >> 24)
							m.BitOffset &= 0xffffff
						}
					}
				}
				t.Members = append(t.Members, m)
			}
		}
		b.types = append(b.types, t)
		raw = raw[12+extra:]
	}
	max := uint32(len(b.types))
	for _, ref := range arrayReferences {
		if ref > max {
			return nil, errors.New("bpf btf: array type out of bounds")
		}
	}
	for _, t := range b.types {
		switch t.Kind {
		case 2, 8, 9, 10, 11, 12, 13, 14, 17, 18:
			if t.SizeOrType > max {
				return nil, errors.New("bpf btf: type reference out of bounds")
			}
		}
		for _, m := range t.Members {
			if m.Type > max {
				return nil, errors.New("bpf btf: member type out of bounds")
			}
		}
	}
	return b, nil
}

// Types returns copies of all type records, in ID order.
func (b *BTF) Types() []BTFType {
	if b == nil {
		return nil
	}
	out := make([]BTFType, len(b.types))
	for i, t := range b.types {
		out[i] = t
		out[i].Members = append([]BTFMember(nil), t.Members...)
	}
	return out
}

// Type returns the record with id, or false for void (0) or unknown IDs.
func (b *BTF) Type(id uint32) (BTFType, bool) {
	if b == nil || id == 0 || uint64(id) > uint64(len(b.types)) {
		return BTFType{}, false
	}
	t := b.types[id-1]
	t.Members = append([]BTFMember(nil), t.Members...)
	return t, true
}

// Find returns all types with an exact name and kind, preserving duplicates.
// Callers must reject ambiguity rather than selecting an arbitrary match.
func (b *BTF) Find(name string, kind uint8) []BTFType {
	var out []BTFType
	if b != nil {
		for _, t := range b.types {
			if t.Name == name && t.Kind == kind {
				c, _ := b.Type(t.ID)
				out = append(out, c)
			}
		}
	}
	return out
}
