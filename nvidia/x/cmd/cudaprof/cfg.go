package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func runCFGCommand(rawArgs []string) error {
	args, limits, err := parseReadLimits(rawArgs)
	if err != nil {
		return err
	}
	if len(args) == 0 {
		return errors.New("usage: cudaprof cfg [--bb] [kernel] capture.pprof")
	}

	basicBlocks := false
	cmd := args[0]
	args = args[1:]

	var kernel string
	switch {
	case cmd == "bbcfg" || cmd == "-bbcfg" || cmd == "--bbcfg":
		basicBlocks = true
	case strings.HasPrefix(cmd, "-bbcfg="):
		basicBlocks = true
		kernel = strings.TrimPrefix(cmd, "-bbcfg=")
	case strings.HasPrefix(cmd, "--bbcfg="):
		basicBlocks = true
		kernel = strings.TrimPrefix(cmd, "--bbcfg=")
	case cmd == "cfg" || cmd == "-cfg" || cmd == "--cfg":
		// default hyperblocks
	case strings.HasPrefix(cmd, "-cfg="):
		kernel = strings.TrimPrefix(cmd, "-cfg=")
	case strings.HasPrefix(cmd, "--cfg="):
		kernel = strings.TrimPrefix(cmd, "--cfg=")
	default:
		return fmt.Errorf("unknown cfg command: %s", cmd)
	}

	posArgs := make([]string, 0, len(args))
	for _, a := range args {
		if a == "--bb" || a == "-bb" {
			basicBlocks = true
		} else {
			posArgs = append(posArgs, a)
		}
	}

	var capturePath string
	switch len(posArgs) {
	case 1:
		capturePath = posArgs[0]
	case 2:
		if kernel != "" {
			return errors.New("kernel specified multiple times")
		}
		kernel = posArgs[0]
		capturePath = posArgs[1]
	default:
		return errors.New("usage: cudaprof cfg [--bb] [kernel] capture.pprof")
	}

	path, err := filepath.Abs(capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	return printCaptureCFG(os.Stdout, cap, kernel, basicBlocks)
}

func runInspectCFG(args []string, limits cudacapture.Limits) error {
	basicBlocks := args[0] == "--bbcfg"
	var kernel, capturePath string
	switch len(args) {
	case 2:
		capturePath = args[1]
	case 3:
		kernel = args[1]
		capturePath = args[2]
	default:
		return errors.New("usage: cudaprof inspect [--capture-file-bytes=N] [--capture-total-bytes=N] --cfg|--bbcfg [kernel] capture.pprof")
	}
	path, err := filepath.Abs(capturePath)
	if err != nil {
		return err
	}
	cap, err := readCapturePathLimits(path, limits)
	if err != nil {
		return fmt.Errorf("read capture: %w", err)
	}
	return printCaptureCFG(os.Stdout, cap, kernel, basicBlocks)
}

func printCaptureCFG(w io.Writer, cap *cudacapture.Capture, kernel string, basicBlocks bool) error {
	dotText, err := findCaptureDOT(cap, basicBlocks)
	if err != nil {
		return err
	}

	if kernel != "" {
		extracted, err := cudasass.ExtractDOTGraph(dotText, kernel)
		if err != nil {
			return err
		}
		dotText = extracted
	}

	if !strings.HasSuffix(dotText, "\n") {
		dotText += "\n"
	}
	_, err = io.WriteString(w, dotText)
	return err
}

func findCaptureDOT(cap *cudacapture.Capture, basicBlocks bool) (string, error) {
	if cap == nil {
		return "", errors.New("nil capture")
	}

	if basicBlocks {
		if cap.Code.BBCFG != "" {
			if data, ok := cap.Files[cap.Code.BBCFG]; ok && len(data) > 0 {
				return string(data), nil
			}
		}
		if data, ok := cap.Files["kernels.bbcfg.dot"]; ok && len(data) > 0 {
			return string(data), nil
		}
		for name, data := range cap.Files {
			if strings.HasSuffix(name, ".bbcfg.dot") && len(data) > 0 {
				return string(data), nil
			}
		}
	} else {
		if cap.Code.CFG != "" {
			if data, ok := cap.Files[cap.Code.CFG]; ok && len(data) > 0 {
				return string(data), nil
			}
		}
		if data, ok := cap.Files["kernels.cfg.dot"]; ok && len(data) > 0 {
			return string(data), nil
		}
		for name, data := range cap.Files {
			if strings.HasSuffix(name, ".cfg.dot") && len(data) > 0 {
				return string(data), nil
			}
		}
		for name, data := range cap.Files {
			if strings.HasSuffix(name, ".dot") && len(data) > 0 {
				return string(data), nil
			}
		}
	}

	// If not pre-rendered in capture, check if CUBIN exists to disassemble dynamically
	cubinName := cap.Code.CUBIN
	if cubinName == "" {
		cubinName = "kernels.cubin"
	}
	if cubin, ok := cap.Files[cubinName]; ok && len(cubin) > 0 {
		out, err := cudasass.DisassembleCFG(cubin, basicBlocks)
		if err == nil && len(out) > 0 {
			return out, nil
		}
		return "", fmt.Errorf("no control flow graph embedded and nvdisasm failed: %w", err)
	}

	return "", errors.New("capture does not contain control flow graph (.dot) file")
}
