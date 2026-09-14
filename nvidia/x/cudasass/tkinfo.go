package cudasass

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

// Toolchain names a compiler that produced part of a CUDA ELF image, read from
// the .note.nv.tkinfo note ptxas stamps into every cubin it writes.
//
// Version carries the release, so two images built months apart are
// distinguishable by it. Tool is "ptxas" in every image measured.
type Toolchain struct {
	Tool    string // tki_toolName, e.g. "ptxas"
	Version string // tki_toolVersion, e.g. "Cuda compilation tools, release 13.0, V13.0.88"
	Count   int    // notes in the image carrying this exact stamp
}

// String returns the stamp as "tool: version", with a count when it is not 1.
func (t Toolchain) String() string {
	if t.Count == 1 {
		return t.Tool + ": " + t.Version
	}
	return fmt.Sprintf("%s: %s (x%d)", t.Tool, t.Version, t.Count)
}

// Fields of the .note.nv.tkinfo descriptor. The descriptor is a version word
// followed by a table of byte offsets into a string blob that follows the
// table. The field order is nvdisasm's own, as it renders this section under
// --emit-json: nv_note_tkinfo, tki_objFname, tki_toolName, tki_toolVersion,
// tki_toolBranch, tki_toolOptions. See testdata/nvdisasm-13.0-sm121.json.
const (
	tkinfoName    = "NVIDIA Corp" // ELF note name
	tkinfoType    = 2000          // ELF note type
	tkinfoVersion = 2             // descriptor version this package reads

	tkinfoFields      = 6
	tkinfoToolName    = 2
	tkinfoToolVersion = 3
)

// ReadToolchains returns the distinct toolchain stamps in a CUDA ELF image,
// sorted by tool and version, each with the number of notes that carry it.
//
// A single-element result is an image built by one compiler; more than one
// element is an image assembled from several, which is the condition this
// exists to detect. An image with no stamp yields no elements and no error:
// not every ELF is a cubin, and not every cubin is stamped.
//
// It reads r with [debug/elf] alone: no CUDA driver is loaded and no external
// tool is run. Both paths where the note has been observed are covered, a bare
// cubin and a host object or executable that carries .note.nv.tkinfo sections
// of its own. Cubins reachable only inside a compressed .nv_fatbin payload are
// not decompressed and so are not reported.
//
// [debug/elf] accepts only e_version EV_CURRENT, which every cubin from CUDA
// 13.0 on satisfies. A CUDA 12 cubin carries its own toolkit version there
// instead, as documented on [checkCUBIN], and is reported by name rather than
// left to [debug/elf] to reject as a mismatched version.
func ReadToolchains(r io.ReaderAt) ([]Toolchain, error) {
	if err := checkEVersion(r); err != nil {
		return nil, err
	}
	f, err := elf.NewFile(r)
	if err != nil {
		return nil, fmt.Errorf("cudasass: read elf: %w", err)
	}
	defer f.Close()

	counts := make(map[Toolchain]int)
	for _, sec := range f.Sections {
		if sec.Type != elf.SHT_NOTE {
			continue
		}
		data, err := sec.Data()
		if err != nil {
			return nil, fmt.Errorf("cudasass: read %s: %w", sec.Name, err)
		}
		for len(data) > 0 {
			name, typ, desc, rest, err := nextNote(data, f.ByteOrder)
			if err != nil || len(rest) >= len(data) {
				break // trailing padding, not a malformed image
			}
			data = rest
			if name != tkinfoName || typ != tkinfoType {
				continue
			}
			t, err := parseTKInfo(desc, f.ByteOrder)
			if err != nil {
				return nil, fmt.Errorf("cudasass: %s: %w", sec.Name, err)
			}
			counts[t]++
		}
	}

	out := make([]Toolchain, 0, len(counts))
	for t, n := range counts {
		t.Count = n
		out = append(out, t)
	}
	slices.SortFunc(out, func(a, b Toolchain) int {
		if a.Tool != b.Tool {
			return strings.Compare(a.Tool, b.Tool)
		}
		return strings.Compare(a.Version, b.Version)
	})
	return out, nil
}

// checkEVersion reports a little-endian ELF whose e_version is not EV_CURRENT,
// which [elf.NewFile] rejects with a message naming neither the file nor the
// toolkit that wrote it. Anything else is left for [elf.NewFile] to diagnose.
func checkEVersion(r io.ReaderAt) error {
	var hdr [24]byte
	if _, err := r.ReadAt(hdr[:], 0); err != nil {
		return nil
	}
	if string(hdr[:len(elf.ELFMAG)]) != elf.ELFMAG || elf.Data(hdr[elf.EI_DATA]) != elf.ELFDATA2LSB {
		return nil
	}
	if v := binary.LittleEndian.Uint32(hdr[20:24]); v != uint32(elf.EV_CURRENT) {
		return fmt.Errorf("cudasass: read elf: e_version is %d, not EV_CURRENT; CUDA 12 nvcc writes its toolkit version there and debug/elf cannot read the image", v)
	}
	return nil
}

// nextNote splits one ELF note off the front of b, returning the remainder.
// Note names and descriptors are each padded to a four-byte boundary.
func nextNote(b []byte, order binary.ByteOrder) (name string, typ uint32, desc, rest []byte, err error) {
	const hdr = 12
	if len(b) < hdr {
		return "", 0, nil, nil, errors.New("truncated note header")
	}
	namesz := uint64(order.Uint32(b[0:4]))
	descsz := uint64(order.Uint32(b[4:8]))
	typ = order.Uint32(b[8:12])
	nameEnd := hdr + (namesz+3)&^3
	descEnd := nameEnd + (descsz+3)&^3
	if descEnd > uint64(len(b)) {
		return "", 0, nil, nil, errors.New("truncated note")
	}
	name = strings.TrimRight(string(b[hdr:hdr+namesz]), "\x00")
	return name, typ, b[nameEnd : nameEnd+descsz], b[descEnd:], nil
}

// parseTKInfo reads the tool name and version out of a .note.nv.tkinfo
// descriptor. tki_toolBranch is deliberately not returned: on the images
// measured here it holds the unexpanded placeholder "Build system must define
// TOOLS_VERSION_EXTENDED" rather than a branch, so it cannot be relied on.
func parseTKInfo(desc []byte, order binary.ByteOrder) (Toolchain, error) {
	const table = tkinfoFields * 4
	if len(desc) < table {
		return Toolchain{}, errors.New("truncated tkinfo descriptor")
	}
	if v := order.Uint32(desc[0:4]); v != tkinfoVersion {
		return Toolchain{}, fmt.Errorf("tkinfo descriptor version %d, want %d", v, tkinfoVersion)
	}
	blob := desc[table:]
	str := func(field int) (string, error) {
		off := uint64(order.Uint32(desc[field*4:]))
		if off >= uint64(len(blob)) {
			return "", fmt.Errorf("tkinfo field %d offset %d past %d-byte string blob", field, off, len(blob))
		}
		s := blob[off:]
		n := bytes.IndexByte(s, 0)
		if n < 0 {
			return "", fmt.Errorf("tkinfo field %d is unterminated", field)
		}
		return string(s[:n]), nil
	}
	tool, err := str(tkinfoToolName)
	if err != nil {
		return Toolchain{}, err
	}
	version, err := str(tkinfoToolVersion)
	if err != nil {
		return Toolchain{}, err
	}
	// A descriptor whose table is a different width than this one assumes
	// would still index inside the blob and yield text from the middle of a
	// neighbouring string. Reject the unprintable results that produces.
	if tool == "" || strings.ContainsFunc(tool, func(r rune) bool { return r < ' ' || r > '~' }) {
		return Toolchain{}, fmt.Errorf("tkinfo tool name %q is not a tool name", tool)
	}
	return Toolchain{Tool: tool, Version: version}, nil
}
