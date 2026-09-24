package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// runLinesCommand totals a capture's samples by the program line each
// sampled instruction was compiled from, next to the number of instructions
// the line has in the capture's disassembly.
//
//	cudaprof lines [-sample type] [-format profile|rollup|digraph|dot] [-diff other.pprof] [kernel] capture.pprof
//
// -format chooses the report: the per-line profile, a rollup charging each
// line's instructions to its callers as well, or the call graph as a digraph
// or in DOT. The last three read the capture's translation unit, which is
// also what says which files the program wrote, so an instruction that came
// from generated text is charged to elsewhere rather than counted as a
// program line.
const linesUsage = "usage: cudaprof lines [-sample type] [-format profile|rollup|digraph|dot] [-diff other.pprof] [kernel] capture.pprof"

func runLinesCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("lines", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	sample := fs.String("sample", "", "sample type to total (default: the profile's default type, or its per-instruction type when the default is per launch)")
	format := fs.String("format", "profile", "report to write: profile, rollup, digraph, or dot")
	diff := fs.String("diff", "", "compare the capture with this one, line by line")
	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("%s: %w", linesUsage, err)
	}
	switch *format {
	case "profile", "rollup", "digraph", "dot":
	default:
		return fmt.Errorf("unknown -format %q; want profile, rollup, digraph, or dot", *format)
	}
	var kernel, capturePath string
	switch fs.NArg() {
	case 1:
		capturePath = fs.Arg(0)
	case 2:
		kernel, capturePath = fs.Arg(0), fs.Arg(1)
	default:
		return errors.New(linesUsage)
	}
	path, err := filepath.Abs(capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	if *diff == "" {
		return printCaptureLines(os.Stdout, cap, kernel, *sample, *format)
	}
	if *format != "profile" {
		return fmt.Errorf("-diff writes the per-line comparison, so it cannot also write -format %s", *format)
	}
	otherPath, err := filepath.Abs(*diff)
	if err != nil {
		return err
	}
	other, err := readCapturePathLimits(otherPath, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	return diffCaptures(os.Stdout, cap, other, kernel, *sample, filepath.Base(path), filepath.Base(otherPath))
}

func printCaptureLines(w io.Writer, cap *cudacapture.Capture, kernel, sample, format string) error {
	if cap.Profile == nil {
		return errors.New("capture has no profile")
	}
	if format == "profile" {
		var err error
		if sample, err = lineSample(cap.Profile, sample); err != nil {
			return err
		}
	}
	if len(cap.Modules) < 2 {
		return printModuleLines(w, cap, kernel, sample, format)
	}
	// Each module is its own build with its own disassembly, and two can
	// hold a kernel of the same name; only samples labelled with the
	// module's identity are its own. A sample the producer left unlabelled
	// is given the one module that defines its symbol, if one does.
	inferred := inferModules(cap)
	printed := 0
	unlabelled := 0
	for _, s := range cap.Profile.Sample {
		// A copy or a synchronization runs no code and belongs to no
		// module, so only unlabelled kernel work is worth reporting.
		if subtype := s.Label["subtype"]; len(subtype) == 1 && subtype[0] != "kernel" {
			continue
		}
		if len(s.Label["module_id"]) != 1 {
			unlabelled++
		}
	}
	for _, mod := range cap.Modules {
		local := moduleView(cap, mod)
		if kernel != "" {
			if dis, err := captureDisassembly(local); err != nil || findKernel(dis, kernel) == nil {
				continue
			}
		}
		if printed > 0 {
			fmt.Fprintln(w)
		}
		name := mod.Name
		if name == "" {
			name = mod.ID
		}
		fmt.Fprintf(w, "module %s\n", name)
		if err := printModuleLines(w, local, kernel, sample, format); err != nil {
			return fmt.Errorf("module %s: %w", mod.ID, err)
		}
		printed++
	}
	if printed == 0 {
		return fmt.Errorf("kernel %q not in any module's disassembly", kernel)
	}
	if inferred.Inferred > 0 {
		fmt.Fprintf(w, "\nkernel work attributed by symbol: %d records the producer left without a module (labelled module_inferred=symbol)\n", inferred.Inferred)
	}
	if unlabelled > 0 {
		fmt.Fprintf(w, "\nkernel work naming no module: %d records, counted in none of the tables above (%d with a symbol in no module, %d in several)\n", unlabelled, inferred.Absent, inferred.Ambiguous)
	}
	return nil
}

// moduleView is the capture as that one module saw it: the module's own code
// files and the samples labelled with its identity. Two modules can hold a
// kernel of the same name, so only labelled samples are a module's own.
func moduleView(cap *cudacapture.Capture, mod cudacapture.Module) *cudacapture.Capture {
	local := *cap
	local.Code, local.Modules = mod.Code, nil
	// A module that names no unit of its own still belongs to whatever
	// program the capture as a whole was generated from.
	if local.Code.Unit == "" {
		local.Code.Unit = cap.Code.Unit
	}
	local.Profile = cap.Profile.Copy()
	local.Profile.Sample = nil
	for _, s := range cap.Profile.Sample {
		if ids := s.Label["module_id"]; len(ids) == 1 && ids[0] == moduleLabel(mod) {
			local.Profile.Sample = append(local.Profile.Sample, s)
		}
	}
	return &local
}

// moduleLabel is the value a producer writes in a sample's "module_id"
// label for this module: the driver's number, not the name of the stored
// image. A module the census did not number matches no sample.
func moduleLabel(mod cudacapture.Module) string {
	if mod.DriverID == 0 {
		return ""
	}
	return strconv.FormatUint(uint64(mod.DriverID), 10)
}

func printModuleLines(w io.Writer, cap *cudacapture.Capture, kernel, sample, format string) error {
	dis, err := captureDisassembly(cap)
	if err != nil {
		return err
	}
	if kernel != "" {
		fn := findKernel(dis, kernel)
		if fn == nil {
			return fmt.Errorf("kernel %q not in the capture's disassembly", kernel)
		}
		kernel = fn.Name
	}
	ptx, _, _ := findPTX(cap)
	a := cudalineinfo.Attribute(captureUnit(cap, kernel), string(ptx), dis)
	switch format {
	case "rollup":
		// The cumulative view needs the callers the unit records, so a
		// capture that carries no unit has nothing to roll anything up to.
		if len(a.Unit.LineMap) == 0 {
			return errors.New("capture carries no unit, so no line has a caller to roll up to")
		}
		return cudalineinfo.WriteRollup(w, a)
	case "digraph":
		return cudalineinfo.WriteDigraph(w, a)
	case "dot":
		return cudalineinfo.WriteDot(w, a)
	}
	p, err := lineProfile(cap.Profile, a, kernel, sample)
	if err != nil {
		return err
	}
	return cudalineinfo.WriteProfile(w, a, p)
}

// captureUnit is the generated translation unit the capture was built from,
// or, when it carries none, the bare unit naming the kernel under report.
//
// The difference is what can be said about the result. A bare unit knows only
// the kernel's name, so every line has to be taken on trust and a line in
// generated code is indistinguishable from a line in the program. The unit
// knows which lines are its own and which call stack each one came from, so
// the report can roll lines up to their callers and say plainly that a line
// belongs to generated code.
func captureUnit(cap *cudacapture.Capture, kernel string) cudalineinfo.Unit {
	file := cap.Code.Unit
	if file == "" {
		// A capture that named the unit per module and has only one module
		// is naming it for the capture: the module is all there is.
		for _, m := range cap.Modules {
			if m.Code.Unit != "" {
				file = m.Code.Unit
				break
			}
		}
	}
	data, ok := cap.Files[file]
	if !ok {
		return cudalineinfo.Unit{Name: kernel}
	}
	var u cudalineinfo.Unit
	if err := json.Unmarshal(data, &u); err != nil || u.Name == "" {
		return cudalineinfo.Unit{Name: kernel}
	}
	return u
}

// findKernel returns the function named kernel, or else the first whose
// name matches it ignoring case or contains it.
func findKernel(dis *cudasass.Disassembly, kernel string) *cudasass.Function {
	if fn := dis.Function(kernel); fn != nil {
		return fn
	}
	for _, f := range dis.Functions {
		if strings.EqualFold(f.Name, kernel) || strings.Contains(f.Name, kernel) {
			return f
		}
	}
	return nil
}

// captureDisassembly parses the capture's SASS or, when it holds only
// CUBINs, disassembles each of them with nvdisasm, whose line comments
// carry the same file and line the SASS would.
func captureDisassembly(cap *cudacapture.Capture) (*cudasass.Disassembly, error) {
	if sass, ok := findSASS(cap); ok {
		dis, err := cudasass.ParseString(string(sass))
		if err != nil {
			return nil, fmt.Errorf("parse SASS: %w", err)
		}
		return dis, nil
	}
	var cubins [][]byte
	for _, m := range cap.Modules {
		if data := cap.Files[m.Code.CUBIN]; len(data) > 0 {
			cubins = append(cubins, data)
		}
	}
	if len(cubins) == 0 {
		if data, ok := findCUBIN(cap); ok {
			cubins = append(cubins, data)
		}
	}
	if len(cubins) == 0 {
		return nil, errors.New("capture has neither SASS disassembly nor a CUBIN")
	}
	var text []byte
	for _, cubin := range cubins {
		out, err := cudasass.Disassembler{}.Disassemble(context.Background(), cubin, cudasass.Options{PrintCode: true, SourceLines: true})
		if err != nil {
			return nil, fmt.Errorf("disassemble CUBIN: %w", err)
		}
		text = append(text, out.Stdout...)
		text = append(text, '\n')
	}
	dis, err := cudasass.ParseString(string(text))
	if err != nil {
		return nil, fmt.Errorf("parse nvdisasm output: %w", err)
	}
	return dis, nil
}

// lineProfile totals one sample type per site. A sample's site is the one
// its kernel frame names, or, for a sample recorded with only a SASS frame,
// the one the disassembly gives that instruction. Samples in the kernel
// on a file that is not the program's count as elsewhere; samples in a
// function the disassembly lacks are in another module and count as
// foreign; samples with no instruction-level location at all, such as a
// kernel's whole time, count as unresolved. With a kernel, samples in
// other kernels are skipped.
//
// A profile records files by the path the profiler resolved, which need
// not be the path the compiler recorded, so a frame's site that the
// disassembly does not know is matched to the disassembly's site with the
// same base name and line when there is exactly one.
func lineProfile(prof *profile.Profile, a *cudalineinfo.Attribution, kernel, sample string) (cudalineinfo.Profile, error) {
	if sample == "" {
		sample = prof.DefaultSampleType
	}
	idx := -1
	for i, st := range prof.SampleType {
		if sample == "" || st.Type == sample {
			idx = i
			break
		}
	}
	if idx < 0 {
		return cudalineinfo.Profile{}, fmt.Errorf("no sample type %q; the profile has %s", sample, sampleTypeNames(prof))
	}
	byBase := map[cudalineinfo.Site][]cudalineinfo.Site{}
	for _, s := range a.Sites() {
		k := cudalineinfo.Site{File: filepath.Base(s.File), Line: s.Line}
		byBase[k] = append(byBase[k], s)
	}
	p := cudalineinfo.Profile{Measure: prof.SampleType[idx].Type, Sites: map[cudalineinfo.Site]int64{}}
	for _, s := range prof.Sample {
		if idx >= len(s.Value) || s.Value[idx] == 0 {
			continue
		}
		n := s.Value[idx]
		f := deviceFrames(s)
		if f.kernel == "" {
			p.Unresolved += n
			continue
		}
		if kernel != "" && f.kernel != kernel {
			continue
		}
		if a.Disassembly.Function(f.kernel) == nil {
			p.Foreign += n
			continue
		}
		switch {
		case f.site.File != "":
			if !a.Unit.ProgramFile(f.site.File) {
				p.Other += n
				break
			}
			site := f.site
			if _, ok := a.SASS[site]; !ok {
				if same := byBase[cudalineinfo.Site{File: filepath.Base(site.File), Line: site.Line}]; len(same) == 1 {
					site = same[0]
				}
			}
			p.Sites[site] += n
		case f.sassLine > 0:
			if site, ok := a.SiteAtLine(f.kernel, f.sassLine); ok {
				p.Sites[site] += n
			} else {
				p.Other += n
			}
		default:
			p.Unresolved += n
		}
	}
	return p, nil
}

// lineSample names the sample type lines totals: sample, or when it is empty
// the profile's default, or when the default is not measured per instruction
// the first type that is. It refuses a type measured only per launch.
//
// A kernel's device time is measured per launch: its sample's stack is the
// kernel alone, with no instruction under it. Totalled by line, all of it
// lands at unresolved PCs, which reads as a capture whose PCs could not be
// resolved when it holds none to resolve. Activity captures, the injected
// agent's among them, default to kernel_time, and carry PC samples beside it
// only when captured with --pcsampling.
//
// It is resolved against the whole capture, before any split by module: a
// module whose kernels were never sampled still has its launches timed.
func lineSample(prof *profile.Profile, sample string) (string, error) {
	explicit := sample != ""
	if !explicit {
		sample = prof.DefaultSampleType
	}
	idx := -1
	for i, st := range prof.SampleType {
		if sample == "" || st.Type == sample {
			idx = i
			break
		}
	}
	if idx < 0 {
		return "", fmt.Errorf("no sample type %q; the profile has %s", sample, sampleTypeNames(prof))
	}
	carried, located := measured(prof, idx)
	if !carried || located {
		return prof.SampleType[idx].Type, nil
	}
	var pcs []string
	for i, st := range prof.SampleType {
		if carried, located := measured(prof, i); carried && located {
			pcs = append(pcs, st.Type)
		}
	}
	name := prof.SampleType[idx].Type
	switch {
	case len(pcs) == 0:
		return "", fmt.Errorf("%s is measured per launch, not per instruction, so it has no line to total by, "+
			"and this capture holds no per-instruction measure; capture with --pcsampling for one", name)
	case explicit:
		return "", fmt.Errorf("%s is measured per launch, not per instruction, so it has no line to total by; "+
			"this capture's per-instruction measures: %s", name, strings.Join(pcs, ", "))
	}
	return pcs[0], nil
}

// measured reports whether any sample carries sample type idx, and whether
// any that does names an instruction, by a SASS frame or a program line. A
// carrying sample that names none is an unresolved PC of a measure that
// resolves others, or, if no sample names one, a measure taken per launch.
func measured(prof *profile.Profile, idx int) (carried, located bool) {
	for _, s := range prof.Sample {
		if idx >= len(s.Value) || s.Value[idx] == 0 {
			continue
		}
		carried = true
		if f := deviceFrames(s); f.sass || f.site.File != "" {
			return true, true
		}
	}
	return carried, false
}

func sampleTypeNames(prof *profile.Profile) string {
	var names []string
	for _, st := range prof.SampleType {
		names = append(names, st.Type)
	}
	return strings.Join(names, ", ")
}

// frames is what a sample's stack says about the device: the kernel it
// was in, the program line its kernel frame names, and the position of
// its SASS frame in the disassembly.
type frames struct {
	kernel   string
	site     cudalineinfo.Site
	sass     bool // the stack has a SASS frame, even one at no known ordinal
	sassLine int
}

// deviceFrames reads the device frames of a sample, leaf first: the
// kernel's SASS, PTX, and inlined frames, then the kernel frame itself,
// which is the first frame without a tier suffix. Frames above it belong
// to the host.
func deviceFrames(s *profile.Sample) frames {
	var f frames
	for _, loc := range s.Location {
		for _, line := range loc.Line {
			if line.Function == nil {
				continue
			}
			name := line.Function.Name
			switch {
			case strings.HasSuffix(name, ".SASS"):
				if f.kernel == "" {
					f.kernel = strings.TrimSuffix(name, ".SASS")
					f.sass = true
					f.sassLine = int(line.Line)
				}
			case strings.HasSuffix(name, ".PTX"), strings.HasSuffix(name, ".inlined"):
				if f.kernel == "" {
					f.kernel = strings.TrimSuffix(strings.TrimSuffix(name, ".PTX"), ".inlined")
				}
			default:
				if f.kernel == "" {
					f.kernel = name
				}
				file := line.Function.Filename
				if file != "" && line.Line > 0 && !isTierFile(file) {
					f.site = cudalineinfo.Site{File: file, Line: int(line.Line)}
				}
				return f
			}
		}
	}
	return f
}

// isTierFile reports whether a frame's file is a disassembly tier rather
// than program source.
func isTierFile(file string) bool {
	switch strings.ToLower(filepath.Ext(file)) {
	case ".sass", ".ptx", ".cubin":
		return true
	}
	return false
}
