package advise

import (
	"bytes"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// capturePTXASReport returns the ptxas -v report a capture holds, from a file
// whose name says it is one ("kernels.ptxas.txt", "ptxas.log"), or nil.
//
// A compile-only capture of a target with no device has no cubin this
// package can always read and often only the text the compiler printed; the
// report gives the same per-kernel resources .nv.info does.
func capturePTXASReport(cap *cudacapture.Capture) *ptxasinfo.Report {
	var names []string
	for name := range cap.Files {
		if strings.Contains(strings.ToLower(name), "ptxas") {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	for _, name := range names {
		r, err := ptxasinfo.Parse(bytes.NewReader(cap.Files[name]))
		if err == nil && len(r.Functions()) > 0 {
			return r
		}
	}
	return nil
}

// ptxasResources returns the resources of each entry function in a ptxas
// report, for the target arch when it names one.
//
// ptxas reports the shared memory the kernel declares. The cubin's
// .nv.shared section, which [cudaoccupancy.KernelResource] documents, also
// holds the driver's reservation (1 KiB on sm_121a), so occupancy computed
// from this report can be one block per SM optimistic where shared memory
// binds.
func ptxasResources(r *ptxasinfo.Report, arch string) []kernelResource {
	if r == nil {
		return nil
	}
	var out []kernelResource
	for _, t := range r.Targets {
		if arch != "" && t.Arch != "" && !strings.EqualFold(t.Arch, arch) {
			continue
		}
		for _, f := range t.Functions {
			if !f.Entry {
				continue
			}
			out = append(out, kernelResource{KernelResourceInfo: cudaoccupancy.KernelResourceInfo{
				Name:           f.Name,
				Registers:      f.Registers,
				RegCount:       f.Registers,
				SharedMemBytes: f.SharedMem,
				FrameSizeBytes: f.StackFrame,
				FrameSize:      f.StackFrame,
				MaxRegCount:    f.MaxRegisters,
			}})
		}
	}
	return out
}

// ptxasArch returns the single target a ptxas report compiled for, or "".
func ptxasArch(r *ptxasinfo.Report) string {
	if r == nil {
		return ""
	}
	arch := ""
	for _, t := range r.Targets {
		if t.Arch == "" || len(t.Functions) == 0 {
			continue
		}
		if arch != "" && arch != t.Arch {
			return ""
		}
		arch = t.Arch
	}
	return arch
}
