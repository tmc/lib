package main

import (
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
	disasm, err := loadDisassemblyWithPressure(cap)
	if err != nil {
		return err
	}

	var targets []*cudasass.Function
	if kernel != "" {
		fn := disasm.Function(kernel)
		if fn == nil {
			for _, f := range disasm.Functions {
				if strings.EqualFold(f.Name, kernel) || strings.Contains(f.Name, kernel) {
					fn = f
					break
				}
			}
		}
		if fn == nil {
			return fmt.Errorf("kernel %q not found in disassembly", kernel)
		}
		targets = []*cudasass.Function{fn}
	} else {
		targets = disasm.Functions
	}

	if len(targets) == 0 {
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
	stalls := extractStallMetrics(cap, code, disasm)

	for i, fn := range targets {
		if i > 0 {
			fmt.Fprintln(w)
		}
		fmt.Fprintf(w, "Kernel: %s\n", fn.Name)

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
	if cap == nil {
		return nil, errors.New("nil capture")
	}

	// 1. Try reading pre-joined pressure from capture files if present.
	if disasm, err := capturefiles.Pressure(cap); err == nil && disasm != nil {
		return disasm, nil
	}

	// 2. If pre-joined pressure is not present, extract CUBIN and run DisassembleWithPressure.
	cubinData, ok := findCUBIN(cap)
	if ok && len(cubinData) > 0 {
		disasm, err := cudasass.DisassembleWithPressure(cubinData)
		if err != nil {
			return nil, fmt.Errorf("disassemble with pressure: %w", err)
		}
		return disasm, nil
	}

	return nil, errors.New("capture contains neither CUBIN nor SASS disassembly with register pressure")
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
