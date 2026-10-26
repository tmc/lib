package cudahostemu

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// CLCHeaderFile is the filename of the Blackwell clusterlaunchcontrol host
// emulation C++ header.
const CLCHeaderFile = "clc.h"

//go:embed clc.h
var clcHeader string

// CLCHeader returns the C++ header implementing Blackwell sm_100a
// clusterlaunchcontrol (try_cancel / query_cancel) host emulation and async
// mbarrier hazard detection.
func CLCHeader() string {
	return clcHeader
}

var reCLCGlobalKernel = regexp.MustCompile(`__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)`)

// RunWithCLC compiles and runs src on the host CPU with Blackwell sm_100a
// clusterlaunchcontrol work-stealing emulation configured for activeCTAs
// resident CTAs and totalTiles logical work tiles.
//
// Resident CTAs 0 .. activeCTAs-1 execute with block dimensions block and
// sharedBytes of dynamic shared memory, stealing unlaunched tile IDs
// activeCTAs .. totalTiles-1 via eg_emu_clc_try_cancel_async. Any pointer
// buffer Arg with non-empty Data is updated in place with the buffer's final
// contents after the launch. If a thread queries a clusterlaunchcontrol
// response before awaiting its mbarrier, RunWithCLC returns an error.
func RunWithCLC(ctx context.Context, src string, activeCTAs, totalTiles int, block Dim3, sharedBytes int, args ...Arg) error {
	if activeCTAs <= 0 {
		return fmt.Errorf("cudahostemu: activeCTAs must be positive, got %d", activeCTAs)
	}
	if totalTiles <= 0 {
		return fmt.Errorf("cudahostemu: totalTiles must be positive, got %d", totalTiles)
	}

	dir, err := os.MkdirTemp("", "cudahostemu-clc-*")
	if err != nil {
		return fmt.Errorf("clc temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, CLCHeaderFile), []byte(clcHeader), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", CLCHeaderFile, err)
	}

	launchStmt, bufs, argToBuf, err := buildCLCLaunch(src, activeCTAs, totalTiles, block, sharedBytes, args)
	if err != nil {
		return err
	}

	var fullSrc strings.Builder
	fullSrc.WriteString("#include \"" + CLCHeaderFile + "\"\n")
	fullSrc.WriteString("#line 1 \"source\"\n")
	fullSrc.WriteString(src)

	res, err := Run(ctx, dir, Program{
		Source:  fullSrc.String(),
		Launch:  launchStmt,
		Buffers: bufs,
	})
	if err != nil {
		return err
	}

	for argIdx, bufName := range argToBuf {
		if out, ok := res[bufName]; ok && len(args[argIdx].Data) > 0 {
			copy(args[argIdx].Data, out)
		}
	}
	return nil
}

func buildCLCLaunch(src string, activeCTAs, totalTiles int, block Dim3, sharedBytes int, args []Arg) (string, []Buffer, map[int]string, error) {
	var bufs []Buffer
	var callArgs []string
	argToBuf := make(map[int]string)

	for i, a := range args {
		if a.Expr != "" && a.CType == "" && a.Data == nil && a.Size == 0 {
			callArgs = append(callArgs, a.Expr)
			continue
		}
		name := a.Name
		if name == "" {
			name = fmt.Sprintf("cudahostemu_clc_arg_%d", i)
		}
		ctype := a.CType
		if ctype == "" {
			ctype = "uint8_t"
		}
		data := a.Data
		if len(data) == 0 {
			sz := a.Size
			if sz <= 0 {
				sz = 4096
			}
			data = make([]byte, sz)
		}
		bufs = append(bufs, Buffer{Name: name, CType: ctype, Data: data})
		callArgs = append(callArgs, name)
		argToBuf[i] = name
	}

	kernelName, params := findCLCKernelSignature(src)
	if kernelName == "" {
		return "", nil, nil, fmt.Errorf("cudahostemu: no __global__ kernel found in source")
	}

	if len(args) == 0 && len(params) > 0 {
		for i, p := range params {
			if strings.Contains(p, "*") {
				name := fmt.Sprintf("cudahostemu_clc_auto_%d", i)
				ctype := extractBaseCType(p)
				bufs = append(bufs, Buffer{Name: name, CType: ctype, Data: make([]byte, 65536)})
				callArgs = append(callArgs, name)
			} else {
				callArgs = append(callArgs, "{}")
			}
		}
	}

	bx := block.X
	if bx <= 0 {
		bx = 1
	}
	by := block.Y
	if by <= 0 {
		by = 1
	}
	bz := block.Z
	if bz <= 0 {
		bz = 1
	}
	smem := sharedBytes
	if smem < 0 {
		smem = 0
	}

	argSuffix := ""
	if len(callArgs) > 0 {
		argSuffix = ", " + strings.Join(callArgs, ", ")
	}
	launchStmt := fmt.Sprintf(
		"eg_emu_clc_init(%d, %d); eg_emu_launch(dim3(%d, 1, 1), dim3(%d, %d, %d), %d, %s%s); eg_emu_clc_check()",
		activeCTAs, totalTiles, activeCTAs, bx, by, bz, smem, kernelName, argSuffix,
	)
	return launchStmt, bufs, argToBuf, nil
}

func findCLCKernelSignature(src string) (string, []string) {
	matches := reCLCGlobalKernel.FindAllStringSubmatch(src, -1)
	for _, m := range matches {
		name := m[1]
		rawParams := strings.TrimSpace(m[2])
		if rawParams == "" || rawParams == "void" {
			return name, nil
		}
		parts := strings.Split(rawParams, ",")
		params := make([]string, 0, len(parts))
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p != "" {
				params = append(params, p)
			}
		}
		return name, params
	}
	return "", nil
}
