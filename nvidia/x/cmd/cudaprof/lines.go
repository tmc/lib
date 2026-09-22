package main

import (
	"context"
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
//	cudaprof lines [-sample type] [kernel] capture.pprof
func runLinesCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("lines", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	sample := fs.String("sample", "", "sample type to total (default: the profile's default type)")
	if err := fs.Parse(args); err != nil {
		return fmt.Errorf("usage: cudaprof lines [-sample type] [kernel] capture.pprof: %w", err)
	}
	var kernel, capturePath string
	switch fs.NArg() {
	case 1:
		capturePath = fs.Arg(0)
	case 2:
		kernel, capturePath = fs.Arg(0), fs.Arg(1)
	default:
		return errors.New("usage: cudaprof lines [-sample type] [kernel] capture.pprof")
	}
	path, err := filepath.Abs(capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	return printCaptureLines(os.Stdout, cap, kernel, *sample)
}

func printCaptureLines(w io.Writer, cap *cudacapture.Capture, kernel, sample string) error {
	if cap.Profile == nil {
		return errors.New("capture has no profile")
	}
	if len(cap.Modules) < 2 {
		return printModuleLines(w, cap, kernel, sample)
	}
	// Each module is its own build with its own disassembly, and two can
	// hold a kernel of the same name; only samples labelled with the
	// module's identity are its own.
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
		local := *cap
		local.Code, local.Modules = mod.Code, nil
		local.Profile = cap.Profile.Copy()
		local.Profile.Sample = nil
		for _, s := range cap.Profile.Sample {
			if ids := s.Label["module_id"]; len(ids) == 1 && ids[0] == moduleLabel(mod) {
				local.Profile.Sample = append(local.Profile.Sample, s)
			}
		}
		if kernel != "" {
			if dis, err := captureDisassembly(&local); err != nil || findKernel(dis, kernel) == nil {
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
		if err := printModuleLines(w, &local, kernel, sample); err != nil {
			return fmt.Errorf("module %s: %w", mod.ID, err)
		}
		printed++
	}
	if printed == 0 {
		return fmt.Errorf("kernel %q not in any module's disassembly", kernel)
	}
	if unlabelled > 0 {
		fmt.Fprintf(w, "\nkernel work naming no module: %d records, counted in none of the tables above\n", unlabelled)
	}
	return nil
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

func printModuleLines(w io.Writer, cap *cudacapture.Capture, kernel, sample string) error {
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
	a := cudalineinfo.Attribute(cudalineinfo.Unit{Name: kernel}, string(ptx), dis)
	p, err := lineProfile(cap.Profile, a, kernel, sample)
	if err != nil {
		return err
	}
	return cudalineinfo.WriteProfile(w, a, p)
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
// on a file that is not the program's count as elsewhere; samples with no
// instruction-level location at all, such as a kernel's whole time, and
// samples in a function the disassembly lacks, count as unresolved. With a
// kernel, samples in other kernels are skipped.
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
		var names []string
		for _, st := range prof.SampleType {
			names = append(names, st.Type)
		}
		return cudalineinfo.Profile{}, fmt.Errorf("no sample type %q; the profile has %s", sample, strings.Join(names, ", "))
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
			p.Unresolved += n
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

// frames is what a sample's stack says about the device: the kernel it
// was in, the program line its kernel frame names, and the position of
// its SASS frame in the disassembly.
type frames struct {
	kernel   string
	site     cudalineinfo.Site
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
