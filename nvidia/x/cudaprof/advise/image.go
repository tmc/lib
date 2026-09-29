package advise

import (
	"bytes"
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

// kernelImage is one CUBIN advise reads kernel resources and SASS from.
type kernelImage struct {
	file  string
	cubin []byte

	// kernels names the kernels this image answers for, or is nil when it
	// answers for every kernel it defines.
	kernels map[string]bool

	// shares holds, for each kernel this image answers for that also ran
	// from another image, the fraction of the kernel's launch time the
	// journal records from this one.
	shares map[string]float64
}

// module returns the image to name in a finding about kernel: its file when
// the kernel ran from more than one image, and "" otherwise.
func (img kernelImage) module(kernel string) string {
	if _, ok := img.shares[kernel]; ok {
		return img.file
	}
	return ""
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
				if _, ok := modules[c.module]; c.module != 0 && ok {
					images[i].kernels[name] = true
					holders[name] = append(holders[name], c.file)
				}
			}
		}
		for i, c := range cands {
			for name := range images[i].kernels {
				if len(holders[name]) < 2 {
					continue
				}
				if images[i].shares == nil {
					images[i].shares = make(map[string]float64)
				}
				images[i].shares[name] = moduleShare(ran[name], c.module)
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

// kernelResource is the .nv.info resources of a kernel, and the image to name
// in findings about it (see kernelImage.module).
type kernelResource struct {
	cudaoccupancy.KernelResourceInfo
	module string
}

// kernelResources returns the .nv.info resources of each kernel the images
// answer for.
func kernelResources(images []kernelImage) []kernelResource {
	var infos []kernelResource
	for _, img := range images {
		kinfos, err := cudaoccupancy.ReadKernelInfoBytes(img.cubin)
		if err != nil {
			continue
		}
		for _, k := range kinfos {
			if img.answers(k.Name) {
				infos = append(infos, kernelResource{k, img.module(k.Name)})
			}
		}
	}
	return infos
}

// chargeModules adds to kernelTimes the time of each kernel that ran from
// several images, per image, under the key of a finding that names the image
// (see Location.kernelKey). The time is the kernel's, split by the share of
// its launch time the journal records from each image.
func chargeModules(kernelTimes map[string]int64, images []kernelImage) {
	for _, img := range images {
		for name, share := range img.shares {
			key := Location{Kernel: name, Module: img.file}.kernelKey()
			kernelTimes[key] = int64(float64(kernelTimes[name]) * share)
		}
	}
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

// moduleShare returns the fraction of a kernel's launch time that ran from
// module, given the launch time from each module it ran from. Launches with
// no recorded duration split the kernel evenly.
func moduleShare(times map[uint32]int64, module uint32) float64 {
	var total int64
	for _, t := range times {
		total += t
	}
	if total <= 0 {
		return 1 / float64(len(times))
	}
	return float64(times[module]) / float64(total)
}

// kernelModules returns, for each kernel the journal records a launch of, the
// launch time from each module it ran from, keyed by the module's driver id.
// It is empty when the capture has no readable journal or its records do not
// name modules.
func kernelModules(cap *cudacapture.Capture) map[string]map[uint32]int64 {
	journals, err := timeline.Journals(cap)
	if err != nil {
		return nil
	}
	records, err := timeline.Decode(journals, timeline.MaxRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return nil
	}
	ran := make(map[string]map[uint32]int64)
	for _, a := range records.Activities {
		if a.Subtype != "kernel" || a.Module == 0 || a.Name == "" {
			continue
		}
		if ran[a.Name] == nil {
			ran[a.Name] = make(map[uint32]int64)
		}
		ran[a.Name][a.Module] += max(a.End-a.Start, 0)
	}
	return ran
}

// adviseDisassembly returns the SASS advise reads, holding for each kernel
// the code of each image that answers for it, and the image to name in
// findings about each function (see kernelImage.module).
func adviseDisassembly(cap *cudacapture.Capture, images []kernelImage) (*cudasass.Disassembly, map[*cudasass.Function]string) {
	if d, err := capturefiles.Pressure(cap); err == nil && d != nil {
		return d, nil
	}
	// Each image is two nvdisasm runs, and a process capture holds one
	// image per loaded module, so disassemble them concurrently.
	disasms := make([]*cudasass.Disassembly, len(images))
	sem := make(chan struct{}, runtime.GOMAXPROCS(0))
	var wg sync.WaitGroup
	for i, img := range images {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			if d, err := cudasass.DisassembleWithPressure(img.cubin); err == nil {
				disasms[i] = d
			}
		}()
	}
	wg.Wait()
	var fns []*cudasass.Function
	modules := make(map[*cudasass.Function]string)
	for i, d := range disasms {
		if d == nil {
			continue
		}
		for _, fn := range d.Functions {
			if images[i].answers(fn.Name) {
				fns = append(fns, fn)
				if m := images[i].module(fn.Name); m != "" {
					modules[fn] = m
				}
			}
		}
	}
	if len(fns) > 0 {
		return &cudasass.Disassembly{Functions: fns}, modules
	}
	if data, ok := capturefiles.SASS(cap); ok && len(data) > 0 {
		if d, err := cudasass.Parse(bytes.NewReader(data)); err == nil {
			return d, nil
		}
	}
	return nil, nil
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
func markUnreadSources(report *Report, unread map[string]string) {
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
