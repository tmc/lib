package main

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// kernelImage is one CUBIN advise reads kernel resources and SASS from.
type kernelImage struct {
	file  string
	cubin []byte

	// kernels names the kernels this image answers for, or is nil when it
	// answers for every kernel it defines.
	kernels map[string]bool
}

func (img kernelImage) answers(kernel string) bool {
	return img.kernels == nil || img.kernels[kernel]
}

// kernelImages returns the images advise reads per-kernel code from, and for
// each kernel whose code cannot be bound to the image it ran from, why its
// source location is UNREAD.
//
// A capture of one compilation carries one CUBIN, which answers for every
// kernel. A capture of a process carries one image per loaded module, and a
// kernel name can be defined by several of them: ds4 compiles a probe module
// and a model-probe module that each define ds4_rms besides the module that
// runs it. Choosing among them by map order read a different module on each
// run, and when it read the probe it reported the probe's line table for the
// kernel that ran from the other. The journal records the module each launch
// ran from, and an image answers only for the kernels that ran from it. When
// the journal does not say, the first image in module order answers for a
// name several define, and its source location is UNREAD.
func kernelImages(cap *cudacapture.Capture) ([]kernelImage, map[string]string) {
	if data, ok := findCaptureCUBIN(cap); ok {
		return []kernelImage{{file: "capture CUBIN", cubin: data}}, nil
	}
	type candidate struct {
		file   string
		module uint32
	}
	var cands []candidate
	for _, m := range cap.Modules {
		if data := cap.Files[m.Code.CUBIN]; m.Code.CUBIN != "" && len(data) > 0 {
			cands = append(cands, candidate{m.Code.CUBIN, m.DriverID})
		}
	}
	if len(cands) == 0 {
		for name, data := range cap.Files {
			if strings.HasSuffix(name, ".cubin") && len(data) > 0 {
				cands = append(cands, candidate{file: name})
			}
		}
		sort.Slice(cands, func(i, j int) bool { return cands[i].file < cands[j].file })
	}
	switch len(cands) {
	case 0:
		return nil, nil
	case 1:
		return []kernelImage{{file: cands[0].file, cubin: cap.Files[cands[0].file]}}, nil
	}

	unread := make(map[string]string)
	images := make([]kernelImage, len(cands))
	holders := make(map[string][]string) // kernel -> files that answer for or define it
	if ran := kernelModules(cap); len(ran) > 0 {
		for i, c := range cands {
			images[i] = kernelImage{file: c.file, cubin: cap.Files[c.file], kernels: make(map[string]bool)}
			for name, modules := range ran {
				if c.module != 0 && modules[c.module] {
					holders[name] = append(holders[name], c.file)
					if len(holders[name]) == 1 {
						images[i].kernels[name] = true
					}
				}
			}
		}
		for name, files := range holders {
			if len(files) > 1 {
				unread[name] = fmt.Sprintf("kernel ran from %d modules (%s); advise reads the first and charges time by name", len(files), strings.Join(files, ", "))
			}
		}
	} else {
		for i, c := range cands {
			images[i] = kernelImage{file: c.file, cubin: cap.Files[c.file], kernels: make(map[string]bool)}
			infos, _ := cudaoccupancy.ReadKernelInfoBytes(images[i].cubin)
			for _, k := range infos {
				holders[k.Name] = append(holders[k.Name], c.file)
				if len(holders[k.Name]) == 1 {
					images[i].kernels[k.Name] = true
				}
			}
		}
		for name, files := range holders {
			if len(files) > 1 {
				unread[name] = fmt.Sprintf("kernel is defined in %d captured modules (%s) and the journal does not record which one it ran from", len(files), strings.Join(files, ", "))
			}
		}
	}
	var kept []kernelImage
	for _, img := range images {
		if len(img.kernels) > 0 {
			kept = append(kept, img)
		}
	}
	return kept, unread
}

// findCaptureCUBIN returns the capture's own CUBIN, when it names one.
func findCaptureCUBIN(cap *cudacapture.Capture) ([]byte, bool) {
	for _, name := range []string{cap.Code.CUBIN, "kernels.cubin"} {
		if data := cap.Files[name]; name != "" && len(data) > 0 {
			return data, true
		}
	}
	return nil, false
}

// kernelModules returns, for each kernel the journal records a launch of, the
// driver ids of the modules it ran from. It is empty when the capture has no
// readable journal or its records do not name modules.
func kernelModules(cap *cudacapture.Capture) map[string]map[uint32]bool {
	journals, err := requireJournals(cap)
	if err != nil {
		return nil
	}
	records, err := timeline.Decode(journals, controllerMaxJournalRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil
	}
	ran := make(map[string]map[uint32]bool)
	for _, a := range records.Activities {
		if a.Subtype != "kernel" || a.Module == 0 || a.Name == "" {
			continue
		}
		if ran[a.Name] == nil {
			ran[a.Name] = make(map[uint32]bool)
		}
		ran[a.Name][a.Module] = true
	}
	return ran
}

// adviseDisassembly returns the SASS advise reads, holding for each kernel
// the code of the image that answers for it.
func adviseDisassembly(cap *cudacapture.Capture, images []kernelImage) *cudasass.Disassembly {
	if d, err := extractPreJoinedPressure(cap); err == nil && d != nil {
		return d
	}
	var fns []*cudasass.Function
	for _, img := range images {
		d, err := cudasass.DisassembleWithPressure(img.cubin)
		if err != nil {
			continue
		}
		for _, fn := range d.Functions {
			if img.answers(fn.Name) {
				fns = append(fns, fn)
			}
		}
	}
	if len(fns) > 0 {
		return &cudasass.Disassembly{Functions: fns}
	}
	if data, ok := findSASS(cap); ok && len(data) > 0 {
		if d, err := cudasass.Parse(bytes.NewReader(data)); err == nil {
			return d
		}
	}
	return nil
}

// formatPCs lists the first few addresses of insts and counts the rest.
func formatPCs(insts []cudasass.Instruction) string {
	const shown = 3
	var pcs []string
	for i, inst := range insts {
		if i == shown {
			pcs = append(pcs, fmt.Sprintf("+%d more", len(insts)-shown))
			break
		}
		pcs = append(pcs, fmt.Sprintf("0x%04x", inst.Address))
	}
	return strings.Join(pcs, ", ")
}

// markUnreadSources replaces each source location that is not provably from
// the compilation the kernel ran with UNREAD and the reason. A fusion failure
// applies to every kernel, as view applies it to every line of its listing;
// unread names the kernels kernelImages could not bind to one image.
func markUnreadSources(report *AdviceReport, unread map[string]string) {
	for i := range report.Items {
		loc := &report.Items[i].Location
		if loc.Kernel == "" || loc.File == "" {
			continue
		}
		reason, ok := unread[loc.Kernel]
		if report.SourceFusion != "" {
			reason, ok = "the capture's PTX and SASS are not from one compilation", true
		}
		if !ok {
			continue
		}
		loc.File, loc.Line = "", 0
		loc.Source = "UNREAD: " + reason
	}
}
