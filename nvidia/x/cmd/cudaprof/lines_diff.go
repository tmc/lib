package main

import (
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
)

// diffCaptures writes what changed between two captures of the same program,
// line by line: the instructions each line has in each build, what each
// build's samples measured there, and the opcodes one build has and the other
// does not.
//
// The two are compared as builds, not as runs of one build. A line that loses
// instructions has not necessarily lost work, which is why the measured
// columns are beside the static ones rather than instead of them.
func diffCaptures(w io.Writer, base, target *cudacapture.Capture, kernel, sample, baseLabel, targetLabel string) error {
	b, err := captureSide(base, kernel, sample, baseLabel)
	if err != nil {
		return fmt.Errorf("%s: %w", baseLabel, err)
	}
	t, err := captureSide(target, kernel, sample, targetLabel)
	if err != nil {
		return fmt.Errorf("%s: %w", targetLabel, err)
	}
	return cudalineinfo.WriteDiff(w, b, t)
}

// captureSide is one capture as a side of a comparison: the attribution of
// the module holding the kernel, and what that module's samples measured.
func captureSide(cap *cudacapture.Capture, kernel, sample, label string) (cudalineinfo.Side, error) {
	if cap.Profile == nil {
		return cudalineinfo.Side{}, errors.New("capture has no profile")
	}
	sample, err := lineSample(cap.Profile, sample)
	if err != nil {
		return cudalineinfo.Side{}, err
	}
	local, err := kernelModule(cap, kernel)
	if err != nil {
		return cudalineinfo.Side{}, err
	}
	dis, err := captureDisassembly(local)
	if err != nil {
		return cudalineinfo.Side{}, err
	}
	if kernel != "" {
		fn := findKernel(dis, kernel)
		if fn == nil {
			return cudalineinfo.Side{}, fmt.Errorf("kernel %q not in the capture's disassembly", kernel)
		}
		kernel = fn.Name
	}
	ptx, _, _ := findPTX(local)
	a := cudalineinfo.Attribute(captureUnit(local, kernel), string(ptx), dis)
	p, err := lineProfile(local.Profile, a, kernel, sample)
	if err != nil {
		return cudalineinfo.Side{}, err
	}
	return cudalineinfo.Side{Attribution: a, Label: label, Profile: &p}, nil
}

// kernelModule is the capture as a single build: itself when it holds at most
// one module, and otherwise the view of the one module whose disassembly
// holds the kernel. A capture of several builds compares as none of them
// until the reader says which kernel is meant.
func kernelModule(cap *cudacapture.Capture, kernel string) (*cudacapture.Capture, error) {
	if len(cap.Modules) < 2 {
		return cap, nil
	}
	if kernel == "" {
		return nil, fmt.Errorf("capture holds %d modules; name the kernel to compare", len(cap.Modules))
	}
	for _, mod := range cap.Modules {
		local := moduleView(cap, mod)
		dis, err := captureDisassembly(local)
		if err != nil || findKernel(dis, kernel) == nil {
			continue
		}
		return local, nil
	}
	return nil, fmt.Errorf("kernel %q not in any module's disassembly", kernel)
}
