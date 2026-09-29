package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

const pressureUsage = "usage: cudaprof pressure [kernel] capture.pprof"

func runPressureCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	if len(args) == 0 {
		return errors.New(pressureUsage)
	}

	cmd := args[0]
	var kernel string
	switch {
	case cmd == "pressure" || cmd == "-pressure" || cmd == "--pressure":
		args = args[1:]
	case strings.HasPrefix(cmd, "-pressure="):
		kernel = strings.TrimPrefix(cmd, "-pressure=")
		args = args[1:]
	case strings.HasPrefix(cmd, "--pressure="):
		kernel = strings.TrimPrefix(cmd, "--pressure=")
		args = args[1:]
	}

	var capturePath string
	switch len(args) {
	case 1:
		capturePath = args[0]
	case 2:
		if kernel != "" {
			return errors.New("kernel specified multiple times")
		}
		kernel = args[0]
		capturePath = args[1]
	default:
		return errors.New(pressureUsage)
	}

	path, err := filepath.Abs(capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	return printCapturePressure(os.Stdout, cap, kernel)
}

func printCapturePressure(w io.Writer, cap *cudacapture.Capture, kernel string) error {
	reportMissingModuleCode(w, cap)
	images, err := loadDisassembliesWithPressureForKernel(cap, kernel)
	if err != nil {
		return err
	}
	if len(images) == 0 {
		return errors.New("no functions found in disassembly")
	}

	// The load failure is stated rather than discarded. A source-fusion failure
	// leaves the kernel offsets this table is built from intact, so the table is
	// still correct, but it is the same capture whose source and PTX views are
	// empty and a reader moving between them needs to know why.
	code, fusion, loadErr := capturefiles.LoadCode(cap)
	if fusion != nil {
		fmt.Fprintf(w, "note: %s\n\n", sourceFusionCaveat(fusion))
	} else if loadErr != nil {
		fmt.Fprintf(w, "note: kernel offsets unavailable, resolving addresses from the disassembly alone: %v\n\n", loadErr)
	}
	var targets []struct {
		module string
		disasm *cudasass.Disassembly
		fn     *cudasass.Function
	}
	for _, image := range images {
		for _, fn := range image.disasm.Functions {
			if kernel != "" && !strings.EqualFold(fn.Name, kernel) && !strings.Contains(fn.Name, kernel) {
				continue
			}
			targets = append(targets, struct {
				module string
				disasm *cudasass.Disassembly
				fn     *cudasass.Function
			}{image.module, image.disasm, fn})
		}
	}
	if len(targets) == 0 && kernel != "" {
		return fmt.Errorf("kernel %q not found in disassembly", kernel)
	}
	for i, target := range targets {
		if i > 0 {
			fmt.Fprintln(w)
		}
		fn, disasm := target.fn, target.disasm
		if target.module != "" {
			fmt.Fprintf(w, "Module: %s\n", target.module)
		}
		fmt.Fprintf(w, "Kernel: %s\n", fn.Name)
		stalls := extractStallMetrics(cap, code, disasm)

		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		if stalls.hasStalls {
			fmt.Fprintln(tw, "PC\tSTALL%\tGPR\tPRED\tUGPR\tUPRED\tINSTRUCTION")
		} else {
			fmt.Fprintln(tw, "PC\tGPR\tPRED\tUGPR\tUPRED\tINSTRUCTION")
		}

		kernelTotal := stalls.byKernel[fn.Name]
		if kernelTotal == 0 && len(disasm.Functions) == 1 {
			for _, v := range stalls.byKernel {
				kernelTotal = v
				break
			}
		}

		for _, inst := range fn.Instructions {
			instText := capturefiles.Instruction(inst)
			if stalls.hasStalls {
				var instStall int64
				if pcMap, ok := stalls.byPC[fn.Name]; ok {
					instStall = pcMap[inst.Address]
				} else if len(disasm.Functions) == 1 {
					for _, pcMap := range stalls.byPC {
						instStall = pcMap[inst.Address]
						break
					}
				}
				var pct float64
				if kernelTotal > 0 {
					pct = float64(instStall) / float64(kernelTotal) * 100.0
				}
				fmt.Fprintf(tw, "0x%04x\t%5.1f%%\t%d\t%d\t%d\t%d\t%s\n",
					inst.Address, pct, inst.LiveGPR, inst.LivePred, inst.LiveUGPR, inst.LiveUPred, instText)
			} else {
				fmt.Fprintf(tw, "0x%04x\t%d\t%d\t%d\t%d\t%s\n",
					inst.Address, inst.LiveGPR, inst.LivePred, inst.LiveUGPR, inst.LiveUPred, instText)
			}
		}
		if err := tw.Flush(); err != nil {
			return err
		}
	}

	return nil
}

func loadDisassemblyWithPressure(cap *cudacapture.Capture) (*cudasass.Disassembly, error) {
	images, err := loadDisassembliesWithPressure(cap)
	if err != nil {
		return nil, err
	}
	if len(images) != 1 {
		return nil, errors.New("capture has multiple module images; select a kernel with pressure <kernel> capture.pprof")
	}
	return images[0].disasm, nil
}

type pressureImage struct {
	module string
	disasm *cudasass.Disassembly
}

func loadDisassembliesWithPressure(cap *cudacapture.Capture) ([]pressureImage, error) {
	return loadDisassembliesWithPressureForKernel(cap, "")
}

func loadDisassembliesWithPressureForKernel(cap *cudacapture.Capture, kernel string) ([]pressureImage, error) {
	if cap == nil {
		return nil, errors.New("nil capture")
	}
	if len(cap.Modules) == 0 {
		if disasm, err := capturefiles.Pressure(cap); err == nil && disasm != nil {
			return []pressureImage{{disasm: disasm}}, nil
		}
		cubinData, ok := findCUBIN(cap)
		if !ok || len(cubinData) == 0 {
			return nil, errors.New("capture contains neither CUBIN nor SASS disassembly with register pressure")
		}
		disasm, err := cudasass.DisassembleWithPressure(cubinData)
		if err != nil {
			return nil, fmt.Errorf("disassemble with pressure: %w", err)
		}
		return []pressureImage{{disasm: disasm}}, nil
	}
	var images []pressureImage
	for _, module := range cap.Modules {
		if kernel != "" && !moduleMayContainKernel(cap, module, kernel) {
			continue
		}
		local := *cap
		local.Code = module.Code
		local.Modules = nil
		disasm, err := capturefiles.Pressure(&local)
		if err != nil || disasm == nil {
			cubin := cap.Files[module.Code.CUBIN]
			if len(cubin) == 0 {
				continue
			}
			disasm, err = cudasass.DisassembleWithPressure(cubin)
		}
		if err != nil {
			return nil, fmt.Errorf("module %s: disassemble with pressure: %w", module.ID, err)
		}
		if disasm != nil {
			images = append(images, pressureImage{module: module.ID, disasm: disasm})
		}
	}
	if len(images) == 0 {
		if kernel != "" {
			return nil, fmt.Errorf("kernel %q not found in captured module images", kernel)
		}
		return nil, errors.New("capture contains no module CUBIN or SASS disassembly with register pressure")
	}
	return images, nil
}

// moduleMayContainKernel avoids running nvdisasm on unrelated module images
// when pressure is requested for one named kernel. A byte scan for the
// requested symbol can rule a module out without parsing a large symbol table.
func moduleMayContainKernel(cap *cudacapture.Capture, module cudacapture.Module, kernel string) bool {
	needle := []byte(kernel)
	if sass := cap.Files[module.Code.SASS]; len(sass) != 0 {
		if bytes.Contains(sass, needle) {
			return true
		}
	}
	cubin := cap.Files[module.Code.CUBIN]
	if len(cubin) == 0 {
		return false
	}
	// Kernel identifiers occur in cubin symbol strings. A raw substring scan
	// avoids parsing/allocating the full symbol table of a very large fatbin.
	return bytes.Contains(cubin, needle)
}

func findCUBIN(cap *cudacapture.Capture) ([]byte, bool) {
	if cap.Code.CUBIN != "" {
		if data, ok := cap.Files[cap.Code.CUBIN]; ok && len(data) > 0 {
			return data, true
		}
	}
	if data, ok := cap.Files["kernels.cubin"]; ok && len(data) > 0 {
		return data, true
	}
	for name, data := range cap.Files {
		if strings.HasSuffix(name, ".cubin") && len(data) > 0 {
			return data, true
		}
	}
	return nil, false
}

type kernelStalls struct {
	hasStalls bool
	byPC      map[string]map[uint64]int64
	byKernel  map[string]int64
}

func extractStallMetrics(cap *cudacapture.Capture, code *cudacapture.Code, disasm *cudasass.Disassembly) kernelStalls {
	if cap == nil || cap.Profile == nil || len(cap.Profile.Sample) == 0 {
		return kernelStalls{}
	}

	var stallIndices []int
	for i, st := range cap.Profile.SampleType {
		if strings.HasPrefix(st.Type, "stall_") {
			stallIndices = append(stallIndices, i)
		}
	}
	if len(stallIndices) == 0 {
		return kernelStalls{}
	}

	res := kernelStalls{
		byPC:     make(map[string]map[uint64]int64),
		byKernel: make(map[string]int64),
	}

	for _, s := range cap.Profile.Sample {
		var stalls int64
		for _, idx := range stallIndices {
			stalls += s.Value[idx]
		}
		if stalls == 0 || len(s.Location) == 0 {
			continue
		}
		loc := s.Location[0]
		kName, pc := capturefiles.LocationPC(loc, s, code, disasm)
		if kName == "" {
			continue
		}
		if res.byPC[kName] == nil {
			res.byPC[kName] = make(map[uint64]int64)
		}
		res.byPC[kName][pc] += stalls
		res.byKernel[kName] += stalls
		res.hasStalls = true
	}

	return res
}
