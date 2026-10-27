package ebpfhost

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// FrameContext identifies one process image lifetime. Generation is the
// caller's process-start token, not a thread token. These fields express
// caller consistency only; this package does not inspect a running process.
type FrameContext struct {
	PID                uint32
	Generation, ExecID uint64
}

// MappedFrameImage supplies an admitted ordinary ELF image and one executable
// runtime mapping. Identity.Size and SHA256 must match Data. BuildID and Path
// remain caller descriptor provenance, not independently observed live facts.
// FileOffset is the image offset corresponding to Start; End is exclusive.
type MappedFrameImage struct {
	Identity               Image
	Data                   []byte
	Start, End, FileOffset uint64
}

// StackCapture contains only observed addresses and explicit coverage context.
// FramePointersKnown/FramePointers are caller declarations, not inferred from
// ELF symbols. ID is an optional producer identifier. PCs retain capture order.
type StackCapture struct {
	Context                           FrameContext
	ID                                int64
	PCs                               []uint64
	Requested                         bool
	Unavailable                       string
	FramePointersKnown, FramePointers bool
}

// Frame is one observed native address. An empty Symbol has an explicit Reason.
// DeclaredBuildID is copied from the admitted descriptor; ImageSHA256 is
// recomputed from the supplied bytes. Native symbols are not logical Python
// frames, and SymbolOffset is an offset into a known nonzero function interval.
type Frame struct {
	PC                                   uint64
	ImageSHA256, DeclaredBuildID, Symbol string
	SymbolVersion, Reason                string
	SymbolOffset                         uint64
}

// FrameReport never claims a complete unwind. Truncated means the bounded
// capture cannot establish completeness, not a count of missing frames.
// LogicalFrames is always unresolved_not_captured. Frames preserves every PC,
// including repeated addresses. Context and image mappings are caller assertions.
type FrameReport struct {
	Context                      FrameContext
	CaptureID                    int64
	Requested                    bool
	Unavailable                  string
	FramePointerStatus           string
	Frames                       []Frame
	Partial, Truncated           bool
	IdentityScope, LogicalFrames string
}

type frameSymbol struct {
	name, version       string
	value, size, offset uint64
}

type frameSegment struct{ offset, value, size uint64 }
type frameMapping struct {
	start, end, offset uint64
	digest, buildID    string
	segments           []frameSegment
	symbols            []frameSymbol
}

// FrameResolver snapshots admitted mapping bytes and indexes ordinary ELF
// symbol tables. Its zero value is not usable. It opens no path and performs
// no target-memory read, unwind, tool invocation or live identity verification.
// It is safe for concurrent calls after construction.
type FrameResolver struct {
	context  FrameContext
	mappings []frameMapping
}

// NewFrameResolver accepts at most64 mappings,16MiB total image bytes and65536
// table symbols, with names at most4096bytes. Inputs must be caller-admitted
// ordinary ELF images; these API/index bounds are not hostile-ELF declared-work
// or allocation-safety admission for debug/elf. No fuzz/security claim follows.
// Data and mappings are snapshotted; callers must not mutate them concurrently
// with this call. ELF symbol names/versions are retained without demangling.
func NewFrameResolver(context FrameContext, images []MappedFrameImage) (*FrameResolver, error) {
	if context.PID == 0 || context.Generation == 0 || context.ExecID == 0 || len(images) == 0 || len(images) > 64 {
		return nil, fmt.Errorf("frames: invalid process context or mapping count")
	}
	r := &FrameResolver{context: context}
	remainingBytes, remainingSymbols := 16<<20, 65536
	for _, input := range images {
		if len(input.Data) == 0 || len(input.Data) > remainingBytes || input.Identity.Size != int64(len(input.Data)) || len(input.Identity.Path) > 4096 || len(input.Identity.BuildID) > 4096 {
			return nil, fmt.Errorf("frames: invalid admitted image size or descriptor")
		}
		data := append([]byte(nil), input.Data...)
		sum := sha256.Sum256(data)
		digest := hex.EncodeToString(sum[:])
		if input.Identity.SHA256 != digest {
			return nil, fmt.Errorf("frames: image digest mismatch")
		}
		remainingBytes -= len(data)
		if input.Start >= input.End || input.FileOffset >= uint64(len(data)) || input.FileOffset > ^uint64(0)-(input.End-input.Start) {
			return nil, fmt.Errorf("frames: invalid mapping range")
		}
		for _, old := range r.mappings {
			if input.Start < old.end && old.start < input.End {
				return nil, fmt.Errorf("frames: overlapping mappings")
			}
		}
		f, err := elf.NewFile(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("frames: admitted ELF: %w", err)
		}
		if f.Type != elf.ET_EXEC && f.Type != elf.ET_DYN {
			f.Close()
			return nil, fmt.Errorf("frames: ELF is not an executable or shared image")
		}
		m := frameMapping{start: input.Start, end: input.End, offset: input.FileOffset, digest: digest, buildID: input.Identity.BuildID}
		for _, p := range f.Progs {
			if p.Type == elf.PT_LOAD && p.Flags&elf.PF_X != 0 && p.Filesz != 0 && p.Off <= uint64(len(data)) && p.Filesz <= uint64(len(data))-p.Off && p.Vaddr <= ^uint64(0)-p.Filesz {
				m.segments = append(m.segments, frameSegment{p.Off, p.Vaddr, p.Filesz})
			}
		}
		mapped := false
		for _, p := range m.segments {
			if input.FileOffset < p.offset+p.size && p.offset < input.FileOffset+(input.End-input.Start) {
				mapped = true
			}
		}
		if !mapped {
			f.Close()
			return nil, fmt.Errorf("frames: mapping has no executable file segment")
		}
		seen := make(map[frameSymbol]bool)
		for _, read := range []func() ([]elf.Symbol, error){f.Symbols, f.DynamicSymbols} {
			syms, err := read()
			if err != nil && !errors.Is(err, elf.ErrNoSymbols) {
				f.Close()
				return nil, fmt.Errorf("frames: symbol table: %w", err)
			}
			if len(syms) > remainingSymbols {
				f.Close()
				return nil, fmt.Errorf("frames: symbol count limit")
			}
			remainingSymbols -= len(syms)
			for _, s := range syms {
				if len(s.Name) > 4096 || len(s.Version) > 4096 {
					f.Close()
					return nil, fmt.Errorf("frames: symbol name limit")
				}
				if s.Name == "" || elf.ST_TYPE(s.Info) != elf.STT_FUNC || s.Size == 0 || int(s.Section) >= len(f.Sections) {
					continue
				}
				sec := f.Sections[s.Section]
				if sec.Type == elf.SHT_NOBITS || sec.Flags&elf.SHF_EXECINSTR == 0 || strings.HasPrefix(sec.Name, ".plt") || sec.Name == ".iplt" || s.Value < sec.Addr || s.Value-sec.Addr > sec.Size || s.Size > sec.Size-(s.Value-sec.Addr) {
					continue
				}
				rel := s.Value - sec.Addr
				if sec.Offset > uint64(len(data)) || rel > uint64(len(data))-sec.Offset || s.Size > uint64(len(data))-sec.Offset-rel || s.Value > ^uint64(0)-s.Size {
					continue
				}
				symbol := frameSymbol{s.Name, s.Version, s.Value, s.Size, sec.Offset + rel}
				if !seen[symbol] {
					m.symbols = append(m.symbols, symbol)
					seen[symbol] = true
				}
			}
		}
		f.Close()
		r.mappings = append(r.mappings, m)
	}
	return r, nil
}

// ResolveStack symbolizes at most eight actual PCs against the snapshotted
// mappings. PID/generation and known execution epochs must agree. Unknown
// execution context retains PCs as unresolved. No address is added, reordered
// or adjusted to guess a return address. Caller declarations do not authenticate
// running mappings or frame-pointer coverage.
func (r *FrameResolver) ResolveStack(c StackCapture) (FrameReport, error) {
	if r == nil || r.context.PID == 0 || c.Context.PID != r.context.PID || c.Context.Generation != r.context.Generation {
		return FrameReport{}, fmt.Errorf("frames: process lifetime mismatch")
	}
	if c.Context.ExecID != 0 && c.Context.ExecID != r.context.ExecID {
		return FrameReport{}, fmt.Errorf("frames: execution lifetime mismatch")
	}
	if len(c.PCs) > 8 || len(c.Unavailable) > 4096 || (!c.Requested && (len(c.PCs) != 0 || c.Unavailable != "")) || (c.Unavailable != "" && len(c.PCs) != 0) {
		return FrameReport{}, fmt.Errorf("frames: contradictory or excessive capture")
	}
	out := FrameReport{Context: c.Context, CaptureID: c.ID, Requested: c.Requested, Unavailable: c.Unavailable, Partial: true, Truncated: true, IdentityScope: "caller_mapping_consistency_only", LogicalFrames: "unresolved_not_captured", FramePointerStatus: "unknown", Frames: []Frame{}}
	if c.FramePointersKnown {
		out.FramePointerStatus = "declared_absent"
		if c.FramePointers {
			out.FramePointerStatus = "declared_present"
		}
	}
	if c.Requested && len(c.PCs) == 0 && out.Unavailable == "" {
		out.Unavailable = "empty_bounded_capture"
	}
	for _, pc := range c.PCs {
		if pc == 0 {
			return FrameReport{}, fmt.Errorf("frames: zero captured address")
		}
		frame := Frame{PC: pc, Reason: "unmapped_address"}
		if c.Context.ExecID == 0 {
			frame.Reason = "execution_lifetime_unavailable"
			out.Frames = append(out.Frames, frame)
			continue
		}
		for _, m := range r.mappings {
			if pc < m.start || pc >= m.end {
				continue
			}
			frame.ImageSHA256, frame.DeclaredBuildID = m.digest, m.buildID
			offset := m.offset + (pc - m.start)
			var address uint64
			segments := 0
			for _, p := range m.segments {
				if offset >= p.offset && offset-p.offset < p.size {
					address = p.value + (offset - p.offset)
					segments++
				}
			}
			frame.Reason = "outside_unique_executable_segment"
			if segments != 1 {
				break
			}
			frame.Reason = "no_unique_function_symbol"
			var hit frameSymbol
			matches := 0
			for _, s := range m.symbols {
				if address >= s.value && address-s.value < s.size && s.offset+(address-s.value) == offset {
					hit = s
					matches++
				}
			}
			if matches == 1 {
				frame.Symbol, frame.SymbolVersion, frame.SymbolOffset, frame.Reason = hit.name, hit.version, address-hit.value, ""
			}
			break
		}
		out.Frames = append(out.Frames, frame)
	}
	return out, nil
}
