package cudahostemu

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// PDLHeaderFile is the filename of the Programmatic Dependent Launch emulation header.
const PDLHeaderFile = "pdl.h"

//go:embed pdl.h
var pdlHeaderText string

// PDLHeader returns the C++ header text implementing adversarial Hopper/Blackwell
// Programmatic Dependent Launch (griddepcontrol.launch_dependents / griddepcontrol.wait)
// emulation.
func PDLHeader() string {
	return pdlHeaderText
}

// PDLStage configures one stage in a multi-kernel Programmatic Dependent Launch
// chain executed by [RunPDLChain].
type PDLStage struct {
	Kernel      string   // __global__ function name (e.g. "producer" or "gemm<16>"), or raw C++ launch statement
	Grid        Dim3     // grid dimensions in blocks (zero components default to 1)
	Block       Dim3     // block dimensions in threads (zero components default to 1)
	Cluster     int      // cluster size in blocks along X (0 or 1 means no cluster)
	SharedBytes int      // dynamic shared memory bytes per block
	StageID     uint32   // PDL stage ID (0 defaults to uint32(stageIndex + 1))
	Buffer      string   // optional intermediate buffer name produced by this stage
	Buffers     []string // optional intermediate buffer names produced by this stage
	Args        []string // optional explicit C++ argument expressions for this stage
}

// PDLViolation describes a single Programmatic Dependent Launch ordering or
// synchronization violation detected during host emulation.
type PDLViolation struct {
	StageID uint32
	CTA     int
	Thread  int
	Detail  string
}

// PDLViolationError is returned by [RunPDLChain] when one or more PDL ordering
// violations occur during emulation.
type PDLViolationError struct {
	Violations []PDLViolation
}

func (e *PDLViolationError) Error() string {
	if e == nil || len(e.Violations) == 0 {
		return "cudahostemu: pdl ordering violation"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "cudahostemu: %d PDL ordering violation(s): ", len(e.Violations))
	for i, v := range e.Violations {
		if i > 0 {
			b.WriteString("; ")
		}
		fmt.Fprintf(&b, "[stage %d cta %d] %s", v.StageID, v.CTA, v.Detail)
	}
	return b.String()
}

var rePDLKernel = regexp.MustCompile(`__global__\s+void\s+(?:__cluster_dims__\s*\([^)]*\)\s*)*(?:__launch_bounds__\s*\([^)]*\)\s*)*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)`)

// RunPDLChain compiles and executes a sequence of Programmatic Dependent Launch
// stages on the host CPU with adversarial buffer poisoning enabled.
//
// Intermediate buffers produced by stage i and consumed by stage i+1 are
// registered with eg_emu_pdl_register_buffer and poisoned with 0xff (NaN) until
// the producer triggers eg_pdl_trigger / eg_emu_pdl_launch_dependents and the
// consumer calls eg_pdl_wait / eg_emu_pdl_wait. Final buffer contents are copied
// back into any non-empty Data slices in args. If any PDL ordering violation is
// observed, RunPDLChain returns a [*PDLViolationError].
func RunPDLChain(ctx context.Context, src string, stages []PDLStage, args ...Arg) error {
	if len(stages) == 0 {
		return fmt.Errorf("cudahostemu: empty PDL stage chain")
	}
	dir, err := os.MkdirTemp("", "cudahostemu-pdl-*")
	if err != nil {
		return fmt.Errorf("pdl temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, PDLHeaderFile), []byte(pdlHeaderText), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", PDLHeaderFile, err)
	}

	violationFile := filepath.Join(dir, "pdl_violations.tsv")
	launchStmt, bufs, bufArgIdx, err := buildPDLChainLaunch(src, stages, args)
	if err != nil {
		return err
	}

	var fullSrc strings.Builder
	fmt.Fprintf(&fullSrc, "#define EG_EMU_PDL_VIOLATION_FILE %q\n", violationFile)
	if strings.Contains(src, "EG_PRELUDE") || strings.Contains(src, "eg_pdl_wait()") {
		fullSrc.WriteString("#define EG_HAS_PRELUDE 1\n")
	}
	fullSrc.WriteString("#include \"" + PDLHeaderFile + "\"\n")
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

	for bi, b := range bufs {
		ai := bufArgIdx[bi]
		if ai >= 0 && ai < len(args) && len(args[ai].Data) > 0 {
			if out, ok := res[b.Name]; ok {
				copy(args[ai].Data, out)
			}
		}
	}

	violations, err := readPDLViolationFile(violationFile)
	if err != nil {
		return err
	}
	if len(violations) > 0 {
		return &PDLViolationError{Violations: violations}
	}
	return nil
}

func buildPDLChainLaunch(src string, stages []PDLStage, args []Arg) (string, []Buffer, []int, error) {
	var (
		bufs      []Buffer
		bufArgIdx []int
		argNames  []string
		bufSizes  = make(map[string]int)
		isBufArg  = make(map[string]bool)
	)
	for i, a := range args {
		if a.Expr != "" && a.CType == "" && a.Data == nil && a.Size == 0 {
			argNames = append(argNames, a.Expr)
			continue
		}
		name := a.Name
		if name == "" {
			name = fmt.Sprintf("cudahostemu_arg_%d", i)
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
		} else {
			data = slices.Clone(data)
		}
		bufs = append(bufs, Buffer{Name: name, CType: ctype, Data: data})
		bufArgIdx = append(bufArgIdx, i)
		argNames = append(argNames, name)
		bufSizes[name] = len(data)
		isBufArg[name] = true
	}

	type stageResolved struct {
		callExpr     string
		usedBufs     []string
		nonConstBufs []string
	}
	resolved := make([]stageResolved, len(stages))

	for si, st := range stages {
		if strings.Contains(st.Kernel, "(") {
			resolved[si] = stageResolved{callExpr: st.Kernel}
			for _, b := range bufs {
				if strings.Contains(st.Kernel, b.Name) {
					resolved[si].usedBufs = append(resolved[si].usedBufs, b.Name)
					resolved[si].nonConstBufs = append(resolved[si].nonConstBufs, b.Name)
				}
			}
			continue
		}

		baseName := st.Kernel
		if idx := strings.IndexByte(baseName, '<'); idx >= 0 {
			baseName = strings.TrimSpace(baseName[:idx])
		}
		foundName, params := findPDLKernelSignature(src, baseName)
		kernelExpr := st.Kernel
		if kernelExpr == "" {
			if foundName == "" {
				return "", nil, nil, fmt.Errorf("cudahostemu: stage %d has empty kernel name", si)
			}
			kernelExpr = foundName
		}

		var callArgs []string
		var usedBufs []string
		var nonConstBufs []string

		if len(st.Args) > 0 {
			callArgs = slices.Clone(st.Args)
			for pi, ca := range callArgs {
				if isBufArg[ca] {
					usedBufs = append(usedBufs, ca)
					if pi >= len(params) || !isConstPtrParam(params[pi]) {
						nonConstBufs = append(nonConstBufs, ca)
					}
				}
			}
		} else if len(params) > 0 && len(argNames) > 0 {
			// Match kernel parameter names to named Args when possible; fall back
			// to positional matching.
			allNamedMatch := true
			namedCall := make([]string, len(params))
			for pi, p := range params {
				pname := extractParamIdent(p)
				matched := ""
				for _, an := range argNames {
					if pname != "" && pname == an {
						matched = an
						break
					}
				}
				if matched == "" {
					allNamedMatch = false
					break
				}
				namedCall[pi] = matched
			}
			if allNamedMatch {
				callArgs = namedCall
			} else if len(params) <= len(argNames) {
				callArgs = slices.Clone(argNames[:len(params)])
			} else {
				callArgs = slices.Clone(argNames)
			}
			for pi, ca := range callArgs {
				if isBufArg[ca] {
					usedBufs = append(usedBufs, ca)
					if pi >= len(params) || !isConstPtrParam(params[pi]) {
						nonConstBufs = append(nonConstBufs, ca)
					}
				}
			}
		} else {
			callArgs = slices.Clone(argNames)
			for _, ca := range callArgs {
				if isBufArg[ca] {
					usedBufs = append(usedBufs, ca)
					nonConstBufs = append(nonConstBufs, ca)
				}
			}
		}

		cluster := st.Cluster
		if cluster <= 0 {
			cluster = 1
		}
		gx := st.Grid.X
		if gx <= 0 {
			gx = cluster
		}
		gy := st.Grid.Y
		if gy <= 0 {
			gy = 1
		}
		gz := st.Grid.Z
		if gz <= 0 {
			gz = 1
		}
		bx := st.Block.X
		if bx <= 0 {
			bx = 1
		}
		by := st.Block.Y
		if by <= 0 {
			by = 1
		}
		bz := st.Block.Z
		if bz <= 0 {
			bz = 1
		}
		smem := st.SharedBytes
		if smem < 0 {
			smem = 0
		}

		argSuffix := ""
		if len(callArgs) > 0 {
			argSuffix = ", " + strings.Join(callArgs, ", ")
		}
		var call string
		if cluster > 1 {
			call = fmt.Sprintf("eg_emu_launch_cluster(dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %d, %s%s)",
				gx, gy, gz, bx, by, bz, smem, cluster, kernelExpr, argSuffix)
		} else {
			call = fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %s%s)",
				gx, gy, gz, bx, by, bz, smem, kernelExpr, argSuffix)
		}
		resolved[si] = stageResolved{
			callExpr:     call,
			usedBufs:     usedBufs,
			nonConstBufs: nonConstBufs,
		}
	}

	var stmts []string
	// Register PDL buffers before launching the chain.
	for si := 0; si < len(stages)-1; si++ {
		st := stages[si]
		sid := st.StageID
		if sid == 0 {
			sid = uint32(si + 1)
		}
		var stageBufs []string
		if st.Buffer != "" {
			stageBufs = append(stageBufs, st.Buffer)
		}
		for _, bn := range st.Buffers {
			if bn != "" && !slices.Contains(stageBufs, bn) {
				stageBufs = append(stageBufs, bn)
			}
		}
		if len(stageBufs) == 0 {
			// Infer intermediate buffers written by stage si and read by stage si+1.
			nextUsed := resolved[si+1].usedBufs
			nextOut := resolved[si+1].nonConstBufs
			for _, bname := range resolved[si].nonConstBufs {
				if slices.Contains(nextUsed, bname) {
					// If both stages take the same buffer list (e.g. [buf, out]),
					// prefer buffers that are not the final output of stage si+1.
					if len(resolved[si].nonConstBufs) > 1 && len(nextOut) > 0 && bname == nextOut[len(nextOut)-1] {
						continue
					}
					stageBufs = append(stageBufs, bname)
				}
			}
			if len(stageBufs) == 0 && len(resolved[si].nonConstBufs) > 0 {
				stageBufs = append(stageBufs, resolved[si].nonConstBufs[0])
			} else if len(stageBufs) == 0 && len(bufs) > 0 {
				stageBufs = append(stageBufs, bufs[0].Name)
			}
		}
		for _, bname := range stageBufs {
			if sz, ok := bufSizes[bname]; ok && sz > 0 {
				stmts = append(stmts, fmt.Sprintf("eg_emu_pdl_register_buffer(%s, %d, %du)", bname, sz, sid))
			}
		}
	}

	for _, r := range resolved {
		stmts = append(stmts, r.callExpr)
	}
	stmts = append(stmts, "eg_emu_pdl_flush_violations()")
	return strings.Join(stmts, ";\n    "), bufs, bufArgIdx, nil
}

func findPDLKernelSignature(src, wantName string) (string, []string) {
	matches := rePDLKernel.FindAllStringSubmatch(src, -1)
	for _, m := range matches {
		name := m[1]
		if wantName != "" && name != wantName {
			continue
		}
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
	return wantName, nil
}

func extractParamIdent(param string) string {
	if idx := strings.IndexByte(param, '='); idx >= 0 {
		param = strings.TrimSpace(param[:idx])
	}
	fields := strings.FieldsFunc(param, func(r rune) bool {
		return r == ' ' || r == '\t' || r == '\n' || r == '*' || r == '&'
	})
	if len(fields) == 0 {
		return ""
	}
	return fields[len(fields)-1]
}

func isConstPtrParam(param string) bool {
	if !strings.Contains(param, "*") {
		return false
	}
	idx := strings.IndexByte(param, '*')
	prefix := param[:idx]
	for _, f := range strings.Fields(prefix) {
		if f == "const" {
			return true
		}
	}
	return false
}

func readPDLViolationFile(path string) ([]PDLViolation, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read pdl violations: %w", err)
	}
	var out []PDLViolation
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 4)
		if len(parts) < 4 {
			continue
		}
		sid, _ := strconv.ParseUint(parts[0], 10, 32)
		cta, _ := strconv.Atoi(parts[1])
		tid, _ := strconv.Atoi(parts[2])
		out = append(out, PDLViolation{
			StageID: uint32(sid),
			CTA:     cta,
			Thread:  tid,
			Detail:  parts[3],
		})
	}
	return out, nil
}
