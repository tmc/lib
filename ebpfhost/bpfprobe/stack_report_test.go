package bpfprobe_test

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"testing"

	"github.com/tmc/lib/ebpfhost"
	"github.com/tmc/lib/ebpfhost/bpfprobe"
)

// Fixed ordinary ELF fixture metadata, never loaded or executed. The two
// packages test independently against the same literal interval authority.
func stackFixture(kind elf.Type, name string, dynamic, stripped bool) []byte {
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

func stackImage(data []byte) ebpfhost.MappedFrameImage {
	sum := sha256.Sum256(data)
	return ebpfhost.MappedFrameImage{Identity: ebpfhost.Image{Path: "admitted-inert-fixture", BuildID: "01020304", SHA256: hex.EncodeToString(sum[:]), Size: int64(len(data))}, Data: data, Start: 0x700000, End: 0x700100, FileOffset: 0x1000}
}

func capturedStack(t *testing.T) bpfprobe.Event {
	t.Helper()
	// Normalized wire fields from DecodeEvent, followed by the separately
	// retained stack-map addresses that Probe.Read attaches to that Event.
	var data [168]byte
	put := func(i int, v uint64) { binary.LittleEndian.PutUint64(data[i*8:], v) }
	put(0, 100)
	put(1, uint64(10)<<32|11)
	put(17, 1<<24)
	put(18, 40)
	put(19, 30)
	put(20, 7)
	// Unknown-generation bits22/23 are clear; word18 is thread,19 is exec.
	put(17, 1<<24)
	e, err := bpfprobe.DecodeEvent(data[:])
	if err != nil {
		t.Fatal(err)
	}
	e.Stack = []uint64{0x700028, 0x700028, 0x700040}
	return e
}

func stackResolver(t *testing.T, kind elf.Type) *ebpfhost.FrameResolver {
	t.Helper()
	r, err := ebpfhost.NewFrameResolver(ebpfhost.FrameContext{PID: 10, Generation: 20, ExecID: 30}, []ebpfhost.MappedFrameImage{stackImage(stackFixture(kind, "observed", true, false))})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestReportStackAdapter(t *testing.T) {
	for _, tc := range []struct {
		name string
		kind elf.Type
	}{{"exec", elf.ET_EXEC}, {"relocated", elf.ET_DYN}} {
		t.Run(tc.name, func(t *testing.T) {
			e := capturedStack(t)
			r := stackResolver(t, tc.kind)
			out, err := bpfprobe.ReportStack(e, bpfprobe.StackReportOptions{ProcessGeneration: 20, FramePointersKnown: true, FramePointers: true}, r)
			if err != nil {
				t.Fatal(err)
			}
			if len(out.Frames) != 3 || out.CaptureID != 7 || out.Context.PID != 10 || out.Context.Generation != 20 || out.Context.ExecID != 30 || !out.Partial || !out.Truncated {
				t.Fatalf("capture adapter=%+v", out)
			}
			for i := 0; i < 2; i++ {
				if out.Frames[i].PC != 0x700028 || out.Frames[i].Symbol != "observed" || out.Frames[i].SymbolOffset != 8 {
					t.Fatalf("actual mapped captured address=%+v", out.Frames[i])
				}
			}
			if out.Frames[2].Symbol != "" || out.Frames[2].Reason == "" {
				t.Fatal("guessed symbol at exclusive end")
			}
			e.Stack[0] = 0x900000
			if out.Frames[0].PC != 0x700028 {
				t.Fatal("report aliases Event.Stack")
			}
		})
	}
}

func TestReportStackUnknownContext(t *testing.T) {
	for _, name := range []string{"unknown-exec", "kernel-refusal", "no-fp", "unrequested"} {
		t.Run(name, func(t *testing.T) {
			e := capturedStack(t)
			o := bpfprobe.StackReportOptions{ProcessGeneration: 20, FramePointersKnown: true, FramePointers: true}
			switch name {
			case "unknown-exec":
				e.ExecIDKnown = false
			case "kernel-refusal":
				e.StackID = -14
				e.Stack = nil
				e.StackUnavailable = "kernel_stack_error:-14"
			case "no-fp":
				o.FramePointers = false
			case "unrequested":
				e.UserStackRequested = false
				e.Stack = nil
			}
			out, err := bpfprobe.ReportStack(e, o, stackResolver(t, elf.ET_DYN))
			if err != nil {
				t.Fatal(err)
			}
			if !out.Partial || !out.Truncated || len(out.Frames) != len(e.Stack) || out.LogicalFrames != "unresolved_not_captured" {
				t.Fatalf("invented coverage=%+v", out)
			}
			if name == "unknown-exec" && (out.Context.ExecID != 0 || out.Frames[0].Symbol != "" || out.Frames[0].Reason != "execution_lifetime_unavailable") {
				t.Fatal("invented execution generation")
			}
			if name == "kernel-refusal" && (out.CaptureID != -14 || out.Unavailable != e.StackUnavailable) {
				t.Fatal("kernel refusal lost")
			}
			if name == "no-fp" && out.FramePointerStatus != "declared_absent" {
				t.Fatal("no-FP status lost")
			}
		})
	}
}

func TestReportStackIdentityRefusal(t *testing.T) {
	for _, name := range []string{"pid", "process-generation", "exec", "missing-tid", "contradictory-id", "too-many"} {
		t.Run(name, func(t *testing.T) {
			e := capturedStack(t)
			o := bpfprobe.StackReportOptions{ProcessGeneration: 20}
			switch name {
			case "pid":
				e.PIDTGID = uint64(12)<<32 | 11
			case "process-generation":
				o.ProcessGeneration++
			case "exec":
				e.ExecID++
			case "missing-tid":
				e.PIDTGID = uint64(10) << 32
			case "contradictory-id":
				e.StackID = -14
			case "too-many":
				e.Stack = make([]uint64, 9)
			}
			if _, err := bpfprobe.ReportStack(e, o, stackResolver(t, elf.ET_DYN)); err == nil {
				t.Fatal("invalid actual event admitted")
			}
		})
	}
}
