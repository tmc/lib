package ebpfhost_test

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"reflect"
	"testing"

	"github.com/tmc/lib/ebpfhost"
)

// Inert ELF metadata, never loaded or executed. The symbol interval is
// [0x401020,0x401040), its file interval [0x1020,0x1040). No compiler is used.
func frameFixture(kind elf.Type, name string, dynamic, stripped bool) []byte {
	b := make([]byte, 0x1100)
	copy(b, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(b[16:], uint16(kind))
	binary.LittleEndian.PutUint16(b[18:], uint16(elf.EM_X86_64))
	binary.LittleEndian.PutUint32(b[20:], 1)
	binary.LittleEndian.PutUint64(b[32:], 64)
	binary.LittleEndian.PutUint64(b[40:], 0x500)
	for off, v := range map[int]uint16{52: 64, 54: 56, 56: 1, 58: 64, 60: 6, 62: 4} {
		binary.LittleEndian.PutUint16(b[off:], v)
	}
	binary.LittleEndian.PutUint32(b[64:], uint32(elf.PT_LOAD))
	binary.LittleEndian.PutUint32(b[68:], uint32(elf.PF_R|elf.PF_X))
	for off, v := range map[int]uint64{72: 0x1000, 80: 0x401000, 88: 0x401000, 96: 0x100, 104: 0x100, 112: 0x1000} {
		binary.LittleEndian.PutUint64(b[off:], v)
	}
	strings := []byte("\x00" + name + "\x00alias\x00")
	copy(b[0x280:], strings)
	names := []byte("\x00.text\x00.symtab\x00.strtab\x00.shstrtab\x00.note.gnu.build-id\x00")
	copy(b[0x300:], names)
	section := func(i int, name string, typ elf.SectionType, flags, addr, offset, size uint64, link uint32, align, entsize uint64) {
		s := b[0x500+i*64:]
		binary.LittleEndian.PutUint32(s, uint32(bytes.Index(names, []byte(name+"\x00"))))
		binary.LittleEndian.PutUint32(s[4:], uint32(typ))
		binary.LittleEndian.PutUint64(s[8:], flags)
		binary.LittleEndian.PutUint64(s[16:], addr)
		binary.LittleEndian.PutUint64(s[24:], offset)
		binary.LittleEndian.PutUint64(s[32:], size)
		binary.LittleEndian.PutUint32(s[40:], link)
		binary.LittleEndian.PutUint64(s[48:], align)
		binary.LittleEndian.PutUint64(s[56:], entsize)
	}
	section(1, ".text", elf.SHT_PROGBITS, uint64(elf.SHF_ALLOC|elf.SHF_EXECINSTR), 0x401000, 0x1000, 0x100, 0, 16, 0)
	typ := elf.SHT_SYMTAB
	if dynamic {
		typ = elf.SHT_DYNSYM
	}
	if stripped {
		typ = elf.SHT_PROGBITS
	}
	section(2, ".symtab", typ, 0, 0, 0x200, 48, 3, 8, 24)
	section(3, ".strtab", elf.SHT_STRTAB, 0, 0, 0x280, uint64(len(strings)), 0, 1, 0)
	section(4, ".shstrtab", elf.SHT_STRTAB, 0, 0, 0x300, uint64(len(names)), 0, 1, 0)
	section(5, ".note.gnu.build-id", elf.SHT_NOTE, 0, 0, 0x400, 20, 0, 4, 0)
	s := b[0x218:]
	binary.LittleEndian.PutUint32(s, 1)
	s[4] = elf.ST_INFO(elf.STB_GLOBAL, elf.STT_FUNC)
	binary.LittleEndian.PutUint16(s[6:], 1)
	binary.LittleEndian.PutUint64(s[8:], 0x401020)
	binary.LittleEndian.PutUint64(s[16:], 0x20)
	binary.LittleEndian.PutUint32(b[0x400:], 4)
	binary.LittleEndian.PutUint32(b[0x404:], 4)
	binary.LittleEndian.PutUint32(b[0x408:], 3)
	copy(b[0x40c:], []byte{'G', 'N', 'U', 0, 1, 2, 3, 4})
	return b
}

func frameImage(data []byte) ebpfhost.MappedFrameImage {
	sum := sha256.Sum256(data)
	return ebpfhost.MappedFrameImage{Identity: ebpfhost.Image{Path: "admitted-inert-fixture", BuildID: "01020304", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(data))}, Data: data, Start: 0x700000, End: 0x700100, FileOffset: 0x1000}
}
func frameContext() ebpfhost.FrameContext {
	return ebpfhost.FrameContext{PID: 10, Generation: 20, ExecID: 30}
}
func captureFrames(pcs ...uint64) ebpfhost.StackCapture {
	return ebpfhost.StackCapture{Context: frameContext(), Requested: true, ID: 7, PCs: pcs, FramePointersKnown: true, FramePointers: true}
}

func TestFrameResolverMappedSymbols(t *testing.T) {
	for _, tc := range []struct {
		name    string
		kind    elf.Type
		dynamic bool
	}{{"exec", elf.ET_EXEC, false}, {"relocated-dynsym", elf.ET_DYN, true}} {
		t.Run(tc.name, func(t *testing.T) {
			data := frameFixture(tc.kind, "observed", tc.dynamic, false)
			image := frameImage(data)
			r, err := ebpfhost.NewFrameResolver(frameContext(), []ebpfhost.MappedFrameImage{image})
			if err != nil {
				t.Fatal(err)
			}
			out, err := r.ResolveStack(captureFrames(0x700028, 0x700028, 0x700040, 0x900000))
			if err != nil {
				t.Fatal(err)
			}
			if len(out.Frames) != 4 || !out.Partial || !out.Truncated || out.IdentityScope != "caller_mapping_consistency_only" || out.LogicalFrames != "unresolved_not_captured" {
				t.Fatalf("report=%+v", out)
			}
			for i := 0; i < 2; i++ {
				f := out.Frames[i]
				if f.PC != 0x700028 || f.Symbol != "observed" || f.SymbolOffset != 8 || f.ImageSHA256 != image.Identity.SHA256 || f.DeclaredBuildID != "01020304" || f.Reason != "" {
					t.Fatalf("mapped frame=%+v", f)
				}
			}
			if out.Frames[2].Symbol != "" || out.Frames[2].Reason == "" || out.Frames[3].Reason != "unmapped_address" {
				t.Fatal("guessed half-open/unmapped symbols")
			}
		})
	}
}

func TestFrameResolverUnresolvedCoverage(t *testing.T) {
	for _, name := range []string{"stripped", "no-fp", "unknown-fp", "interpreted-native", "zero-size", "ifunc", "plt", "ambiguous", "unknown-exec", "capture-error", "unrequested", "cap"} {
		t.Run(name, func(t *testing.T) {
			data := frameFixture(elf.ET_DYN, "observed", false, name == "stripped")
			c := captureFrames(0x700028)
			switch name {
			case "no-fp":
				c.FramePointers = false
			case "unknown-fp":
				c.FramePointersKnown = false
			case "interpreted-native":
				data = frameFixture(elf.ET_DYN, "_PyEval_EvalFrameDefault", false, false)
			case "zero-size":
				binary.LittleEndian.PutUint64(data[0x228:], 0)
			case "ifunc":
				data[0x21c] = elf.ST_INFO(elf.STB_GLOBAL, elf.STT_GNU_IFUNC)
			case "plt":
				copy(data[0x301:], []byte(".plt\x00"))
			case "ambiguous":
				copy(data[0x230:0x248], data[0x218:0x230])
				binary.LittleEndian.PutUint32(data[0x230:], uint32(len("observed")+2))
				binary.LittleEndian.PutUint64(data[0x500+2*64+32:], 72)
			case "unknown-exec":
				c.Context.ExecID = 0
			case "capture-error":
				c.PCs = nil
				c.Unavailable = "kernel_stack_error:-14"
			case "unrequested":
				c.Requested = false
				c.PCs = nil
			case "cap":
				c.PCs = []uint64{0x700028, 0x700028, 0x700028, 0x700028, 0x700028, 0x700028, 0x700028, 0x700028}
			}
			r, err := ebpfhost.NewFrameResolver(frameContext(), []ebpfhost.MappedFrameImage{frameImage(data)})
			if err != nil {
				t.Fatal(err)
			}
			out, err := r.ResolveStack(c)
			if err != nil {
				t.Fatal(err)
			}
			if !out.Partial || !out.Truncated || out.LogicalFrames != "unresolved_not_captured" || len(out.Frames) != len(c.PCs) {
				t.Fatalf("invented coverage=%+v", out)
			}
			if name == "no-fp" && out.FramePointerStatus != "declared_absent" {
				t.Fatal("no-FP declaration lost")
			}
			if name == "unknown-fp" && out.FramePointerStatus != "unknown" {
				t.Fatal("guessed frame pointers")
			}
			if name == "interpreted-native" && out.Frames[0].Symbol != "_PyEval_EvalFrameDefault" {
				t.Fatal("native interpreter symbol lost")
			}
			if name == "stripped" || name == "zero-size" || name == "ifunc" || name == "plt" || name == "ambiguous" || name == "unknown-exec" {
				if out.Frames[0].Symbol != "" || out.Frames[0].Reason == "" {
					t.Fatalf("guessed symbol=%+v", out.Frames[0])
				}
			}
			if name == "capture-error" && out.Unavailable != c.Unavailable {
				t.Fatal("capture reason lost")
			}
		})
	}
}

func TestFrameResolverIdentityOwnership(t *testing.T) {
	data := frameFixture(elf.ET_DYN, "observed", false, false)
	images := []ebpfhost.MappedFrameImage{frameImage(data)}
	r, err := ebpfhost.NewFrameResolver(frameContext(), images)
	if err != nil {
		t.Fatal(err)
	}
	before, err := r.ResolveStack(captureFrames(0x700028))
	if err != nil {
		t.Fatal(err)
	}
	for i := range data {
		data[i] = 0
	}
	images[0].Start++
	images[0].Identity.BuildID = "changed"
	after, err := r.ResolveStack(captureFrames(0x700028))
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("caller mutation changed indexed image")
	}
	before.Frames[0].Symbol = "mutated"
	again, err := r.ResolveStack(captureFrames(0x700028))
	if err != nil || again.Frames[0].Symbol != "observed" {
		t.Fatal("returned frames alias resolver")
	}
	for _, name := range []string{"pid", "generation", "exec"} {
		c := captureFrames(0x700028)
		switch name {
		case "pid":
			c.Context.PID++
		case "generation":
			c.Context.Generation++
		case "exec":
			c.Context.ExecID++
		}
		if _, err := r.ResolveStack(c); err == nil {
			t.Fatalf("stale %s admitted", name)
		}
	}
	// A build-ID label is descriptor provenance, not independently authenticated.
	image := frameImage(frameFixture(elf.ET_DYN, "observed", false, false))
	image.Identity.BuildID = "caller-asserted"
	r, err = ebpfhost.NewFrameResolver(frameContext(), []ebpfhost.MappedFrameImage{image})
	if err != nil {
		t.Fatal(err)
	}
	out, err := r.ResolveStack(captureFrames(0x700028))
	if err != nil || out.Frames[0].DeclaredBuildID != "caller-asserted" {
		t.Fatal("descriptor provenance changed")
	}
}

func TestFrameResolverInputRefusals(t *testing.T) {
	for _, name := range []string{"context", "digest", "size", "range", "offset-overflow", "no-segment", "overlap", "mapping-count", "byte-limit"} {
		t.Run(name, func(t *testing.T) {
			ctx := frameContext()
			images := []ebpfhost.MappedFrameImage{frameImage(frameFixture(elf.ET_DYN, "observed", false, false))}
			switch name {
			case "context":
				ctx.Generation = 0
			case "digest":
				images[0].Identity.SHA256 = "bad"
			case "size":
				images[0].Identity.Size++
			case "range":
				images[0].End = images[0].Start
			case "offset-overflow":
				images[0].Start = 1
				images[0].End = ^uint64(0)
			case "no-segment":
				images[0].FileOffset = 0
				images[0].End = images[0].Start + 16
			case "overlap":
				images = append(images, images[0])
			case "mapping-count":
				images = make([]ebpfhost.MappedFrameImage, 65)
			case "byte-limit":
				images[0].Data = make([]byte, (16<<20)+1)
			}
			if r, err := ebpfhost.NewFrameResolver(ctx, images); r != nil || err == nil {
				t.Fatal("invalid image context admitted")
			}
		})
	}
	r, err := ebpfhost.NewFrameResolver(frameContext(), []ebpfhost.MappedFrameImage{frameImage(frameFixture(elf.ET_DYN, "observed", false, false))})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"too-many", "zero", "contradictory", "unrequested-address"} {
		c := captureFrames(0x700028)
		switch name {
		case "too-many":
			c.PCs = make([]uint64, 9)
		case "zero":
			c.PCs[0] = 0
		case "contradictory":
			c.Unavailable = "missing"
		case "unrequested-address":
			c.Requested = false
		}
		if _, err := r.ResolveStack(c); err == nil {
			t.Fatalf("invalid capture %s admitted", name)
		}
	}
	var zero ebpfhost.FrameResolver
	if _, err := zero.ResolveStack(captureFrames(0x700028)); err == nil {
		t.Fatal("zero resolver admitted")
	}
}

func TestFrameResolverTableBounds(t *testing.T) {
	for _, name := range []string{"symbol-count", "symbol-name"} {
		t.Run(name, func(t *testing.T) {
			base := frameFixture(elf.ET_DYN, "observed", false, false)
			var data []byte
			if name == "symbol-count" {
				const count = 65538
				data = make([]byte, 0x2000+count*24)
				copy(data, base)
				for i := 1; i < count; i++ {
					copy(data[0x2000+i*24:], base[0x218:0x230])
				}
				binary.LittleEndian.PutUint64(data[0x500+2*64+24:], 0x2000)
				binary.LittleEndian.PutUint64(data[0x500+2*64+32:], count*24)
			} else {
				text := append([]byte{0}, bytes.Repeat([]byte{'a'}, 4097)...)
				text = append(text, 0)
				data = make([]byte, 0x2000+len(text))
				copy(data, base)
				copy(data[0x2000:], text)
				binary.LittleEndian.PutUint64(data[0x500+3*64+24:], 0x2000)
				binary.LittleEndian.PutUint64(data[0x500+3*64+32:], uint64(len(text)))
			}
			if r, err := ebpfhost.NewFrameResolver(frameContext(), []ebpfhost.MappedFrameImage{frameImage(data)}); r != nil || err == nil {
				t.Fatal("ordinary table exceeds configured bound")
			}
		})
	}
}
