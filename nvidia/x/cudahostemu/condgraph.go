package cudahostemu

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// CondGraphHeaderFile is the filename of the CUDA Conditional Graph emulation header.
const CondGraphHeaderFile = "condgraph.h"

//go:embed condgraph.h
var condGraphHeaderText string

// CondGraphHeader returns the C++ header text implementing CUDA 12.4+/13.x
// cudaGraphConditionalHandle and cudaGraphSetConditional device-side emulation.
func CondGraphHeader() string {
	return condGraphHeaderText
}

var (
	// ErrCondGraphRunawayLoop is returned by [RunConditionalWhile] when the
	// conditional WHILE body never sets the conditional handle to 0 within
	// MaxIterations, or when RequireExplicitUpdate is set and a body iteration
	// fails to call cudaGraphSetConditional.
	ErrCondGraphRunawayLoop = errors.New("cudahostemu: conditional graph WHILE loop exceeded MaxIterations or failed to update handle")

	// ErrCondGraphDataRace is returned by [RunConditionalWhile] when
	// DetectDataRace is enabled and multiple threads update the same
	// conditional handle within a single body iteration.
	ErrCondGraphDataRace = errors.New("cudahostemu: data race on cudaGraphConditionalHandle across threads")
)

// CondGraphStep configures one kernel step inside a conditional WHILE or IF
// graph body executed by [RunConditionalWhile].
type CondGraphStep struct {
	Kernel      string   // __global__ function name, or raw C++ launch statement
	Grid        Dim3     // grid dimensions in blocks (zero components default to 1)
	Block       Dim3     // block dimensions in threads (zero components default to 1)
	Cluster     int      // cluster size in blocks along X (0 or 1 means no cluster)
	SharedBytes int      // dynamic shared memory bytes per block
	Args        []string // optional explicit C++ argument expressions for this step
}

// CondWhileConfig configures a CUDA Conditional WHILE graph node execution in
// [RunConditionalWhile].
type CondWhileConfig struct {
	Handle                uint64 // conditional handle identifier (0 defaults to 1)
	InitialCondition      uint32 // initial handle condition (1 to enter loop, 0 to skip)
	MaxIterations         int    // runaway loop guard (<= 0 defaults to 1024)
	RequireExplicitUpdate bool   // fail with ErrCondGraphRunawayLoop if an iteration does not call cudaGraphSetConditional
	DetectDataRace        bool   // fail with ErrCondGraphDataRace if multiple threads write the handle in one iteration
}

// CondWhileResult summarizes the execution of a CUDA Conditional WHILE loop
// under [RunConditionalWhile].
type CondWhileResult struct {
	IterationsExecuted int      // number of body iterations executed
	FinalCondition     uint32   // final condition value stored in the handle
	HandleUpdateCount  int      // total number of cudaGraphSetConditional calls across all iterations
	IterationTrace     []uint32 // condition value at the end of each executed iteration
}

// CondHandleArg returns a scalar [Arg] binding a named or positional
// cudaGraphConditionalHandle parameter.
func CondHandleArg(name string, handle uint64) Arg {
	if handle == 0 {
		handle = 1
	}
	return Arg{
		Name: name,
		Expr: fmt.Sprintf("((cudaGraphConditionalHandle)%dULL)", handle),
	}
}

var reCondGraphKernel = regexp.MustCompile(`__global__\s+void\s+(?:__cluster_dims__\s*\([^)]*\)\s*)*(?:__launch_bounds__\s*\([^)]*\)\s*)*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)`)

// RunConditionalWhile compiles and executes a conditional WHILE graph body on
// the host CPU across iterations without host round-trips between iterations
// until a kernel inside body calls cudaGraphSetConditional(handle, 0) or
// cfg.MaxIterations trips [ErrCondGraphRunawayLoop].
//
// Final buffer contents are copied back into any non-empty Data slices in args.
func RunConditionalWhile(ctx context.Context, src string, cfg CondWhileConfig, body []CondGraphStep, args ...Arg) (*CondWhileResult, error) {
	if len(body) == 0 {
		return nil, fmt.Errorf("cudahostemu: empty conditional graph body")
	}
	if cfg.Handle == 0 {
		cfg.Handle = 1
	}
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 1024
	}

	dir, err := os.MkdirTemp("", "cudahostemu-condgraph-*")
	if err != nil {
		return nil, fmt.Errorf("condgraph temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, CondGraphHeaderFile), []byte(condGraphHeaderText), 0o644); err != nil {
		return nil, fmt.Errorf("write %s: %w", CondGraphHeaderFile, err)
	}

	traceFile := filepath.Join(dir, "condgraph_trace.tsv")
	launchStmt, bufs, bufArgIdx, err := buildCondWhileLaunch(src, cfg, body, args, traceFile)
	if err != nil {
		return nil, err
	}

	var fullSrc strings.Builder
	fullSrc.WriteString("#include \"" + CondGraphHeaderFile + "\"\n")
	fullSrc.WriteString("#line 1 \"source\"\n")
	fullSrc.WriteString(src)

	res, err := Run(ctx, dir, Program{
		Source:  fullSrc.String(),
		Launch:  launchStmt,
		Buffers: bufs,
	})
	if err != nil {
		return nil, err
	}

	for bi, b := range bufs {
		ai := bufArgIdx[bi]
		if ai >= 0 && ai < len(args) && len(args[ai].Data) > 0 {
			if out, ok := res[b.Name]; ok {
				copy(args[ai].Data, out)
			}
		}
	}

	return parseCondWhileTrace(traceFile)
}

func buildCondWhileLaunch(src string, cfg CondWhileConfig, body []CondGraphStep, args []Arg, traceFile string) (string, []Buffer, []int, error) {
	var (
		bufs      []Buffer
		bufArgIdx []int
		argExprs  []string
		namedExpr = make(map[string]string)
	)
	handleExpr := fmt.Sprintf("((cudaGraphConditionalHandle)%dULL)", cfg.Handle)
	namedExpr["cond_handle"] = handleExpr
	namedExpr["handle"] = handleExpr

	for i, a := range args {
		if a.Expr != "" && a.CType == "" && a.Data == nil && a.Size == 0 {
			argExprs = append(argExprs, a.Expr)
			if a.Name != "" {
				namedExpr[a.Name] = a.Expr
			}
			continue
		}
		name := a.Name
		if name == "" {
			name = fmt.Sprintf("cudahostemu_cg_arg_%d", i)
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
		argExprs = append(argExprs, name)
		namedExpr[name] = name
	}

	bodyCalls := make([]string, 0, len(body))
	for si, st := range body {
		if strings.Contains(st.Kernel, "(") {
			bodyCalls = append(bodyCalls, st.Kernel)
			continue
		}
		baseName := st.Kernel
		if idx := strings.IndexByte(baseName, '<'); idx >= 0 {
			baseName = strings.TrimSpace(baseName[:idx])
		}
		foundName, params := findCondGraphKernelSignature(src, baseName)
		kernelExpr := st.Kernel
		if kernelExpr == "" {
			if foundName == "" {
				return "", nil, nil, fmt.Errorf("cudahostemu: conditional step %d has empty kernel name", si)
			}
			kernelExpr = foundName
		}

		var callArgs []string
		if len(st.Args) > 0 {
			for _, ca := range st.Args {
				if expr, ok := namedExpr[ca]; ok {
					callArgs = append(callArgs, expr)
				} else {
					callArgs = append(callArgs, ca)
				}
			}
		} else if len(params) > 0 {
			allNamed := true
			namedCall := make([]string, len(params))
			for pi, p := range params {
				pname := extractParamIdent(p)
				if expr, ok := namedExpr[pname]; ok {
					namedCall[pi] = expr
					continue
				}
				if strings.Contains(p, "cudaGraphConditionalHandle") {
					namedCall[pi] = handleExpr
					continue
				}
				allNamed = false
				break
			}
			if allNamed {
				callArgs = namedCall
			} else if len(params) <= len(argExprs) {
				callArgs = slices.Clone(argExprs[:len(params)])
			} else {
				callArgs = slices.Clone(argExprs)
				for len(callArgs) < len(params) {
					p := params[len(callArgs)]
					if strings.Contains(p, "cudaGraphConditionalHandle") {
						callArgs = append(callArgs, handleExpr)
					} else {
						callArgs = append(callArgs, "{}")
					}
				}
			}
		} else {
			callArgs = slices.Clone(argExprs)
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
		if cluster > 1 {
			bodyCalls = append(bodyCalls, fmt.Sprintf(
				"eg_emu_launch_cluster(dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %d, %s%s)",
				gx, gy, gz, bx, by, bz, smem, cluster, kernelExpr, argSuffix,
			))
		} else {
			bodyCalls = append(bodyCalls, fmt.Sprintf(
				"eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %s%s)",
				gx, gy, gz, bx, by, bz, smem, kernelExpr, argSuffix,
			))
		}
	}

	reqUpdate := 0
	if cfg.RequireExplicitUpdate {
		reqUpdate = 1
	}
	detectRace := "false"
	if cfg.DetectDataRace {
		detectRace = "true"
	}

	var b strings.Builder
	b.WriteString("{\n")
	b.WriteString("        eg_condgraph_reset();\n")
	fmt.Fprintf(&b, "        eg_condgraph_set_detect_race(%s);\n", detectRace)
	fmt.Fprintf(&b, "        const cudaGraphConditionalHandle cg_handle = %s;\n", handleExpr)
	fmt.Fprintf(&b, "        eg_condgraph_init(cg_handle, %du);\n", cfg.InitialCondition)
	b.WriteString("        std::vector<unsigned int> cg_trace;\n")
	b.WriteString("        int cg_iters = 0;\n")
	b.WriteString("        int cg_runaway = 0;\n")
	b.WriteString("        int cg_missing_update = 0;\n")
	fmt.Fprintf(&b, "        const int cg_max_iters = %d;\n", cfg.MaxIterations)
	fmt.Fprintf(&b, "        const int cg_req_update = %d;\n", reqUpdate)
	b.WriteString("        while (eg_condgraph_get(cg_handle) != 0u) {\n")
	b.WriteString("            if (cg_iters >= cg_max_iters) {\n")
	b.WriteString("                cg_runaway = 1;\n")
	b.WriteString("                break;\n")
	b.WriteString("            }\n")
	b.WriteString("            eg_condgraph_begin_iter(cg_handle);\n")
	for _, call := range bodyCalls {
		fmt.Fprintf(&b, "            %s;\n", call)
	}
	b.WriteString("            ++cg_iters;\n")
	b.WriteString("            cg_trace.push_back(eg_condgraph_get(cg_handle));\n")
	b.WriteString("            if (cg_req_update && eg_condgraph_iter_updates(cg_handle) == 0) {\n")
	b.WriteString("                cg_missing_update = 1;\n")
	b.WriteString("                break;\n")
	b.WriteString("            }\n")
	b.WriteString("        }\n")
	fmt.Fprintf(&b, "        eg_condgraph_write_trace(%q, cg_iters, eg_condgraph_get(cg_handle), eg_condgraph_update_count(cg_handle), cg_runaway, cg_missing_update, eg_condgraph_race_count(cg_handle), cg_trace);\n", traceFile)
	b.WriteString("    }")

	return b.String(), bufs, bufArgIdx, nil
}

func findCondGraphKernelSignature(src, wantName string) (string, []string) {
	matches := reCondGraphKernel.FindAllStringSubmatch(src, -1)
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

func parseCondWhileTrace(path string) (*CondWhileResult, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read condgraph trace: %w", err)
	}
	res := &CondWhileResult{}
	var runaway, missingUpdate, raceCount int
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			continue
		}
		switch parts[0] {
		case "iterations":
			res.IterationsExecuted, _ = strconv.Atoi(parts[1])
		case "final_condition":
			v, _ := strconv.ParseUint(parts[1], 10, 32)
			res.FinalCondition = uint32(v)
		case "update_count":
			res.HandleUpdateCount, _ = strconv.Atoi(parts[1])
		case "runaway":
			runaway, _ = strconv.Atoi(parts[1])
		case "missing_update":
			missingUpdate, _ = strconv.Atoi(parts[1])
		case "race_count":
			raceCount, _ = strconv.Atoi(parts[1])
		case "trace":
			for _, tok := range parts[1:] {
				if tok == "" {
					continue
				}
				v, _ := strconv.ParseUint(tok, 10, 32)
				res.IterationTrace = append(res.IterationTrace, uint32(v))
			}
		}
	}
	if raceCount > 0 {
		return res, fmt.Errorf("%w: %d conflicting thread writes to handle", ErrCondGraphDataRace, raceCount)
	}
	if runaway != 0 || missingUpdate != 0 {
		if missingUpdate != 0 {
			return res, fmt.Errorf("%w: body iteration %d did not call cudaGraphSetConditional", ErrCondGraphRunawayLoop, res.IterationsExecuted)
		}
		return res, fmt.Errorf("%w: condition remained %d after %d iterations", ErrCondGraphRunawayLoop, res.FinalCondition, res.IterationsExecuted)
	}
	return res, nil
}
