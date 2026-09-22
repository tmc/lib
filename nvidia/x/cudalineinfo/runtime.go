package cudalineinfo

import (
	"bufio"
	"bytes"
	"debug/elf"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// A Resolver maps the program counters CUPTI reports for a unit back to the
// sites of its attribution. It implements [cuptipcsampling.CodeResolver], so
// a [cuptipcsampling.ProfileBridge] can carry a program's own lines into a
// pprof profile.
//
// CUPTI reports a PC as a kernel name and a function-relative byte offset,
// which is the address nvdisasm prints, so the join needs no relocation.
type Resolver struct {
	Attribution *Attribution

	// CRC, when nonzero, is the CUBIN identity CUPTI reports for the loaded
	// module; a PC from any other module is left unresolved. Zero accepts
	// every module, which is right when one unit is loaded at a time.
	CRC uint64

	// text holds the byte offset of each function's .text section in the
	// CUBIN, so a resolved location can carry an image-absolute offset.
	text map[string]uint64
}

// NewResolver returns a resolver over a. When cubin is the ELF image the
// attribution was compiled to, the .text section offsets are read from it so
// each location's CUBINOffset is absolute within the image; without it the
// offset is function-relative.
func NewResolver(a *Attribution, cubin []byte) (*Resolver, error) {
	r := &Resolver{Attribution: a}
	if len(cubin) == 0 {
		return r, nil
	}
	f, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		return nil, fmt.Errorf("cudalineinfo: parse cubin: %w", err)
	}
	defer f.Close()
	r.text = map[string]uint64{}
	for _, s := range f.Sections {
		if name, ok := strings.CutPrefix(s.Name, ".text."); ok {
			r.text[name] = s.Offset
		}
	}
	return r, nil
}

// ResolvePC locates a sampled PC. A location is resolved when the function
// and offset are in the disassembly, whether or not the instruction names a
// site: an instruction on a generated line resolves to that line, so the
// profile keeps the prologue and epilogue rather than dropping them.
func (r *Resolver) ResolvePC(mod cuptipcsampling.ModuleID, kernelName string, pcOffset uint64) cuptipcsampling.CodeLocation {
	loc := cuptipcsampling.CodeLocation{KernelName: kernelName}
	if r == nil || r.Attribution == nil || r.CRC != 0 && mod.CRC != r.CRC {
		return loc
	}
	a := r.Attribution
	in, ok := a.Instruction(kernelName, pcOffset)
	if !ok {
		return loc
	}
	loc.Resolved = true
	loc.SourceFile = in.SourceFile
	loc.SourceLine = int64(in.SourceLine)
	loc.SASSFile = a.Unit.Name + ".sass"
	loc.SASSLine = int64(a.ordinal(kernelName, pcOffset))
	loc.CUBINOffset = r.text[kernelName] + pcOffset
	if in.InlinedLine > 0 {
		loc.InlinedAt = []cudaptx.Location{{File: in.InlinedFile, Line: in.InlinedLine}}
	}
	if lines := a.PTXLines[Site{File: in.SourceFile, Line: in.SourceLine}]; len(lines) > 0 {
		loc.PTXFile = a.Unit.Name + ".ptx"
		loc.PTXLine = int64(lines[0])
	}
	return loc
}

// HostFrames returns the program frames above a resolved location, leaf
// first, in the form [cuptipcsampling.ProfileBridge] expects: the statement
// that built the site, then each of its recorded callers. A location that
// names no site has no frames.
func (a *Attribution) HostFrames(loc cuptipcsampling.CodeLocation) []cudaprof.Frame {
	s := Site{File: loc.SourceFile, Line: int(loc.SourceLine)}
	l, ok := a.Unit.Location(s)
	if !ok {
		return nil
	}
	frames := []cudaprof.Frame{{FunctionName: l.Function, FileName: l.File, Line: int64(l.Line)}}
	for _, c := range l.Callers {
		frames = append(frames, cudaprof.Frame{FunctionName: c.Function, FileName: c.File, Line: int64(c.Line)})
	}
	return frames
}

// ordinal returns the one-based position of the instruction at offset in
// the named function, or 0. nvdisasm text has no stable line numbers, so the
// ordinal stands in for a SASS line.
// SiteAtLine returns the site of the instruction at a one-based position in
// a function's disassembly, which is the line a profile's SASS frames
// record, and false if no instruction is there or it names no program line.
func (a *Attribution) SiteAtLine(function string, line int) (Site, bool) {
	if a.Disassembly == nil || line <= 0 {
		return Site{}, false
	}
	fn := a.Disassembly.Function(function)
	if fn == nil || line > len(fn.Instructions) {
		return Site{}, false
	}
	return a.SiteAt(function, fn.Instructions[line-1].Address)
}

func (a *Attribution) ordinal(function string, offset uint64) int {
	if a.Disassembly == nil {
		return 0
	}
	fn := a.Disassembly.Function(function)
	if fn == nil {
		return 0
	}
	for i, in := range fn.Instructions {
		if in.Address == offset {
			return i + 1
		}
	}
	return 0
}

// A Profile is a runtime measure of one unit distributed over its sites:
// hardware PC samples, or instructions executed. Sites holds the measure at
// instructions that name a site, Other the measure at instructions that name
// none, and Unresolved the measure at PCs the disassembly does not have.
type Profile struct {
	Measure    string // what is counted, for reports
	Sites      map[Site]int64
	Other      int64
	Unresolved int64
}

// Total returns the whole measure, attributed or not.
func (p Profile) Total() int64 {
	var n int64
	for _, v := range p.Sites {
		n += v
	}
	return n + p.Other + p.Unresolved
}

// CountSamples distributes the PC samples of batches over the sites of a.
// A PC's samples are the sum over its base stall reasons, which inv names;
// the companion reasons CUPTI reports alongside them count a subset of the
// same cycles and are skipped, matching [cuptipcsampling.ProfileBridge].
func CountSamples(a *Attribution, inv cuptipcsampling.ReasonInventory, batches ...cuptipcsampling.Batch) Profile {
	p := Profile{Measure: "samples", Sites: map[Site]int64{}}
	for _, b := range batches {
		if !b.Valid {
			continue
		}
		for _, pc := range b.PCs {
			if pc.Name == "" {
				continue
			}
			var n int64
			for _, r := range pc.Reasons {
				if _, _, _, companion := cuptipcsampling.ClassifyPipelineStall(inv.Lookup(r.Index)); !companion {
					n += int64(r.Samples)
				}
			}
			p.add(a, pc.Name, pc.Offset, n)
		}
	}
	return p
}

// CountExecuted distributes a SASS metric over the sites of a. Each record's
// instances of metricID are summed, so a per-SM granularity and a per-GPU
// one give the same totals.
func CountExecuted(a *Attribution, metricID uint64, records ...cuptisass.Record) Profile {
	p := Profile{Measure: "executed", Sites: map[Site]int64{}}
	for _, rec := range records {
		var n int64
		for _, v := range rec.Values {
			if v.MetricID == metricID {
				n += int64(v.Value)
			}
		}
		p.add(a, rec.FunctionName, uint64(rec.PCOffset), n)
	}
	return p
}

func (p *Profile) add(a *Attribution, function string, offset uint64, n int64) {
	if n == 0 {
		return
	}
	if _, ok := a.Instruction(function, offset); !ok {
		p.Unresolved += n
		return
	}
	if s, ok := a.SiteAt(function, offset); ok {
		p.Sites[s] += n
		return
	}
	p.Other += n
}

// WriteProfile writes one row per site with its static instruction count
// and its share of the measure, followed by the shares that reached no site.
func WriteProfile(w io.Writer, a *Attribution, p Profile) error {
	src := newSourceReader()
	total := p.Total()
	pct := func(n int64) string {
		if total == 0 {
			return "-"
		}
		return fmt.Sprintf("%.1f%%", 100*float64(n)/float64(total))
	}
	bw := bufio.NewWriter(w)
	if a.Unit.Name != "" {
		fmt.Fprintf(bw, "%s: ", a.Unit.Name)
	}
	fmt.Fprintf(bw, "%d %s by program line\n", total, p.Measure)
	tw := tabwriter.NewWriter(bw, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "  SITE\tSOURCE\tSASS\t%s\tSHARE\n", strings.ToUpper(p.Measure))
	sites := profileSites(a, p)
	names := SiteNames(sites)
	for _, s := range sites {
		fmt.Fprintf(tw, "  %s\t%s\t%d\t%d\t%s\n", names[s], src.line(s), len(a.SASS[s]), p.Sites[s], pct(p.Sites[s]))
	}
	tw.Flush()
	fmt.Fprintf(bw, "  elsewhere: %d at instructions naming no program line, %d at unresolved PCs\n", p.Other, p.Unresolved)
	return bw.Flush()
}

// profileSites returns the attribution's sites and any the profile measured
// that the attribution lacks, in file and line order. A profile whose
// frames carry source lines can name lines the disassembly in hand does not.
func profileSites(a *Attribution, p Profile) []Site {
	sites := a.Sites()
	seen := make(map[Site]bool, len(sites))
	for _, s := range sites {
		seen[s] = true
	}
	for s := range p.Sites {
		if !seen[s] {
			seen[s] = true
			sites = append(sites, s)
		}
	}
	sortSites(sites)
	return sites
}
