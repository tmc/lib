package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/internal/cudaview"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// recompileSources reports one source per module the capture retained, falling
// back to the capture's top-level artifacts when it records no modules.
// recorded supplies the options a module was built with, keyed by module ID and
// then by module name; a capture that records none leaves every module's
// options assumed.
func recompileSources(cap *cudacapture.Capture, arch string, recorded map[string][]string) []*cudaview.RecompileSource {
	if cap == nil {
		return nil
	}
	text := func(name string) string {
		if name == "" {
			return ""
		}
		return string(cap.Files[name])
	}
	build := func(id, name string, code cudacapture.CodeFiles) *cudaview.RecompileSource {
		source := text(code.Source)
		if strings.TrimSpace(source) == "" {
			return nil
		}
		src := &cudaview.RecompileSource{
			ID:     id,
			Name:   name,
			File:   code.Source,
			Arch:   arch,
			Source: source,
			PTX:    text(code.PTX),
			SASS:   text(code.SASS),
		}
		if src.Name == "" {
			src.Name = code.Source
		}
		for _, key := range []string{id, name, code.Source} {
			if options, ok := recorded[key]; ok && key != "" {
				src.Options, src.Recorded = options, true
				break
			}
		}
		if !src.Recorded {
			src.Options = assumedOptions(arch, source)
		}
		src.Suggested = suggestedOptions(src.Options)
		return src
	}

	var sources []*cudaview.RecompileSource
	for _, mod := range cap.Modules {
		if src := build(mod.ID, mod.Name, mod.Code); src != nil {
			sources = append(sources, src)
		}
	}
	if len(sources) > 0 {
		return sources
	}
	code := cap.Code
	if code.Source == "" {
		code.Source = "kernels.cu"
	}
	if code.PTX == "" {
		code.PTX = "kernels.ptx"
	}
	if code.SASS == "" {
		code.SASS = "kernels.sass"
	}
	if src := build("", code.Source, code); src != nil {
		return []*cudaview.RecompileSource{src}
	}
	return nil
}

// suggestedOptions is what the panel offers for the second side: the module's
// own options plus --use_fast_math, except where the module already compiles
// with it, which would compare a variant against itself.
func suggestedOptions(options []string) []string {
	suggested := append([]string(nil), options...)
	for _, opt := range options {
		if opt == "--use_fast_math" {
			return suggested
		}
	}
	return append(suggested, "--use_fast_math")
}

// find reports the source with the given module ID or name, or the first one
// when id is empty.
func findRecompileSource(sources []*cudaview.RecompileSource, id string) *cudaview.RecompileSource {
	if len(sources) == 0 {
		return nil
	}
	if id == "" {
		return sources[0]
	}
	for _, src := range sources {
		if src.ID == id || src.Name == id {
			return src
		}
	}
	return nil
}

// captureVariantLabel names the variant built from the capture's own artifacts
// rather than from a recompilation. It is also the option string that selects
// it in a /api/recompile request.
const captureVariantLabel = "capture"

// An nvrtcVariant is one side of a comparison: either the artifacts the capture
// retained or a recompilation of its source under Options.
type nvrtcVariant struct {
	Label   string       `json:"label"`
	Options []string     `json:"options,omitempty"`
	PTX     string       `json:"ptx,omitempty"`
	SASS    string       `json:"sass,omitempty"`
	Log     string       `json:"log,omitempty"`
	Error   string       `json:"error,omitempty"`
	Hint    string       `json:"hint,omitempty"`
	Stats   variantStats `json:"stats"`
}

// variantStats are static instruction counts, taken from the text of one
// variant. A static count is not a cost: a kernel whose loops stay rolled
// executes an instruction many times, and a kernel that unrolls trades count
// for iterations. Read these to see which instructions the compiler chose,
// then measure time to learn what it was worth.
type variantStats struct {
	PTXInstructions  int            `json:"ptx_instructions"`
	SASSInstructions int            `json:"sass_instructions"`
	Kernels          map[string]int `json:"kernels,omitempty"`
	PTXOpcodes       map[string]int `json:"ptx_opcodes,omitempty"`
	SASSOpcodes      map[string]int `json:"sass_opcodes,omitempty"`
}

// A variantDelta is one row of the comparison table.
type variantDelta struct {
	Group string `json:"group"`
	Name  string `json:"name"`
	A     int    `json:"a"`
	B     int    `json:"b"`
	Delta int    `json:"delta"`
}

// recompileResponse is the body of /api/recompile.
type recompileResponse struct {
	Module  string         `json:"module,omitempty"`
	A       *nvrtcVariant  `json:"a"`
	B       *nvrtcVariant  `json:"b"`
	Deltas  []variantDelta `json:"deltas,omitempty"`
	Note    string         `json:"note,omitempty"`
	Caveat  string         `json:"caveat"`
	Elapsed string         `json:"elapsed,omitempty"`
}

// staticCountCaveat travels with every comparison so the counts are never read
// as a speedup.
const staticCountCaveat = "Static instruction counts. Rolled loops make them misleading, and a memory-bound kernel is unmoved by arithmetic it no longer issues. Measure device time before claiming a win."

// A variantCompiler compiles CUDA C++ source under NVRTC options. Tests
// substitute one; compileNVRTCVariant is the implementation that calls NVRTC.
type variantCompiler func(name, src string, options []string) *nvrtcVariant

// compileNVRTCVariant compiles src under options and disassembles the resulting
// CUBIN. A compilation or disassembly failure is reported in the variant, not
// as an error: the viewer shows the compiler log beside the other variant.
func compileNVRTCVariant(name, src string, options []string) *nvrtcVariant {
	v := &nvrtcVariant{Label: strings.Join(options, " "), Options: options}
	if v.Label == "" {
		v.Label = "(no options)"
	}
	// WithNoCache: the cache disagrees with a fresh compile. Measured against
	// NVRTC 13.0 on ds4_math.cu, --ftz=true, --prec-div=false, --prec-sqrt=false
	// and --extra-device-vectorization each produce a CUBIN bit-identical to the
	// one compiled without them, reproducibly across processes -- they are
	// no-ops on that source. Served from the cache they instead returned a CUBIN
	// no fresh compile produces, and the panel reported them removing 24 FCHK
	// and 8 MUFU.RCP. A comparison that invents a difference is worse than a
	// slow one, so every variant here is compiled.
	opts := []nvrtcx.Option{nvrtcx.WithName(name), nvrtcx.WithNoCache()}
	for _, flag := range options {
		opts = append(opts, nvrtcx.WithFlag(flag))
	}
	mod, err := nvrtcx.Compile(src, opts...)
	if err != nil {
		v.Error = err.Error()
		v.Hint = missingIncludeHint(v.Error)
		return v
	}
	v.PTX = mod.PTX()
	v.Log = mod.Log()
	if cubin := mod.CUBIN(); len(cubin) > 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		out, err := (cudasass.Disassembler{}).Disassemble(ctx, cubin, cudasass.Options{Mode: cudasass.Text, PrintCode: true})
		if err != nil {
			v.Error = fmt.Sprintf("disassemble: %v", err)
		} else {
			v.SASS = string(out.Stdout)
		}
	}
	v.Stats = newVariantStats(v.PTX, v.SASS)
	return v
}

// captureVariant reports the capture's own artifacts as a variant.
func captureVariant(src *cudaview.RecompileSource) *nvrtcVariant {
	v := &nvrtcVariant{Label: captureVariantLabel, PTX: src.PTX, SASS: src.SASS}
	if src.Recorded {
		v.Options = src.Options
	}
	v.Stats = newVariantStats(v.PTX, v.SASS)
	return v
}

// newVariantStats counts instructions in one variant's PTX and SASS text.
func newVariantStats(ptx, sass string) variantStats {
	s := variantStats{PTXOpcodes: countPTXOpcodes(ptx)}
	for _, n := range s.PTXOpcodes {
		s.PTXInstructions += n
	}
	s.SASSOpcodes, s.Kernels = countSASSOpcodes(sass)
	for _, n := range s.SASSOpcodes {
		s.SASSInstructions += n
	}
	return s
}

// countPTXOpcodes counts PTX instructions by opcode, keeping the full opcode
// with its modifiers so div.rn.f32 and div.approx.f32 stay distinct.
func countPTXOpcodes(ptx string) map[string]int {
	counts := make(map[string]int)
	for _, line := range strings.Split(ptx, "\n") {
		if i := strings.Index(line, "//"); i >= 0 {
			line = line[:i]
		}
		line = strings.TrimSpace(line)
		// An instruction ends in a semicolon; a directive begins with a dot,
		// a label with a dollar sign, and a parameter list or body brace has
		// no opcode at all.
		if !strings.HasSuffix(line, ";") || strings.HasPrefix(line, ".") || strings.HasPrefix(line, "$") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		if strings.HasPrefix(fields[0], "@") {
			fields = fields[1:]
			if len(fields) == 0 {
				continue
			}
		}
		opcode := strings.TrimRight(fields[0], ";,")
		if opcode == "" || strings.HasSuffix(opcode, ":") {
			continue
		}
		counts[opcode]++
	}
	return counts
}

// countSASSOpcodes counts SASS instructions by opcode and by kernel, using the
// same parser the SASS tier is built from.
func countSASSOpcodes(sass string) (opcodes, kernels map[string]int) {
	opcodes, kernels = make(map[string]int), make(map[string]int)
	if strings.TrimSpace(sass) == "" {
		return opcodes, kernels
	}
	disasm, err := cudasass.ParseString(sass)
	if err != nil || disasm == nil {
		return opcodes, kernels
	}
	for _, fn := range disasm.Functions {
		for _, inst := range fn.Instructions {
			if inst.Opcode == "" {
				continue
			}
			opcodes[inst.Opcode]++
			kernels[fn.Name]++
		}
	}
	return opcodes, kernels
}

// incomparable reports why two variants cannot be compared, or "" if they can.
// A variant that failed to compile has no instructions, not zero of them: every
// count would read as a decrease, and a compile error would render as the
// largest improvement the panel can show.
func incomparable(a, b *nvrtcVariant) string {
	if a == nil || b == nil {
		return "recompile produced no result"
	}
	for _, side := range []struct {
		name string
		v    *nvrtcVariant
	}{{"A", a}, {"B", b}} {
		if side.v.Error != "" {
			return side.name + " did not compile, so there is nothing to compare: " + side.v.Error
		}
	}
	return ""
}

// diffVariants reports the counts that differ between two variants, totals
// first, then per-kernel SASS instructions, then opcodes ordered by the size of
// the change. Counts equal on both sides are left out.
func diffVariants(a, b *nvrtcVariant) []variantDelta {
	if incomparable(a, b) != "" {
		return nil
	}
	rows := []variantDelta{
		{Group: "total", Name: "PTX instructions", A: a.Stats.PTXInstructions, B: b.Stats.PTXInstructions},
		{Group: "total", Name: "SASS instructions", A: a.Stats.SASSInstructions, B: b.Stats.SASSInstructions},
	}
	rows = append(rows, changedCounts("kernel", a.Stats.Kernels, b.Stats.Kernels)...)
	rows = append(rows, changedCounts("sass", a.Stats.SASSOpcodes, b.Stats.SASSOpcodes)...)
	rows = append(rows, changedCounts("ptx", a.Stats.PTXOpcodes, b.Stats.PTXOpcodes)...)
	for i := range rows {
		rows[i].Delta = rows[i].B - rows[i].A
	}
	return rows
}

// changedCounts returns one row per name whose count differs, largest change
// first and ties broken by name.
func changedCounts(group string, a, b map[string]int) []variantDelta {
	names := make(map[string]bool, len(a)+len(b))
	for name := range a {
		names[name] = true
	}
	for name := range b {
		names[name] = true
	}
	var rows []variantDelta
	for name := range names {
		if a[name] == b[name] {
			continue
		}
		rows = append(rows, variantDelta{Group: group, Name: name, A: a[name], B: b[name]})
	}
	sort.Slice(rows, func(i, j int) bool {
		di, dj := abs(rows[i].B-rows[i].A), abs(rows[j].B-rows[j].A)
		if di != dj {
			return di > dj
		}
		return rows[i].Name < rows[j].Name
	})
	return rows
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// parseVariantOptions splits one side's option string. The empty string and
// captureVariantLabel both select the capture's own artifacts, reported by ok
// being false.
func parseVariantOptions(s string) (options []string, recompile bool) {
	s = strings.TrimSpace(s)
	if s == "" || s == captureVariantLabel {
		return nil, false
	}
	return strings.Fields(s), true
}

// newRecompileHandler serves /api/recompile. Parameter module selects which of
// the capture's modules to recompile; parameters a and b each hold a
// space-separated NVRTC option list, or "capture" for the capture's own
// artifacts.
func newRecompileHandler(sources []*cudaview.RecompileSource, compile variantCompiler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		src := findRecompileSource(sources, r.URL.Query().Get("module"))
		if src == nil || src.Source == "" {
			http.Error(w, "recompile: capture retained no CUDA C++ source for this module", http.StatusNotFound)
			return
		}
		start := time.Now()
		side := func(param string) *nvrtcVariant {
			options, recompile := parseVariantOptions(r.URL.Query().Get(param))
			if !recompile {
				return captureVariant(src)
			}
			return compile(src.File, src.Source, options)
		}
		resp := recompileResponse{Module: src.Name, A: side("a"), B: side("b"), Caveat: staticCountCaveat}
		if resp.Note = incomparable(resp.A, resp.B); resp.Note == "" {
			resp.Deltas = diffVariants(resp.A, resp.B)
		}
		resp.Elapsed = time.Since(start).Round(time.Millisecond).String()
		// A comparison carries the full PTX and SASS of both variants, tens of
		// megabytes, and a slow reader takes longer to drain it than the
		// server's ordinary write timeout allows. Without this the client sees
		// a truncated body under a 200.
		if rc := http.NewResponseController(w); rc != nil {
			rc.SetWriteDeadline(time.Now().Add(10 * time.Minute))
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Printf("recompile: write response: %v", err)
		}
	}
}

// assumedOptions are the options the panel starts from for a module whose real
// ones the capture did not record. They mirror what ds4-min passes, plus an
// include path when the source includes a header: NVRTC searches no directory
// unless told to, and reports "no directories in search list" for a source nvcc
// compiles without complaint, because nvcc knows its own toolkit include
// directory and NVRTC does not.
func assumedOptions(arch, source string) []string {
	var options []string
	if arch != "" {
		options = append(options, "--gpu-architecture="+arch)
	}
	options = append(options, "--generate-line-info")
	if strings.Contains(source, "#include") {
		if dir := cudaIncludeDir(); dir != "" {
			options = append(options, "--include-path="+dir)
		}
	}
	return options
}

// missingIncludeHint explains an NVRTC failure whose only cause is that no
// include path was passed.
func missingIncludeHint(log string) string {
	if !strings.Contains(log, "no directories in search list") {
		return ""
	}
	hint := "NVRTC searches no directory by default. Add --include-path=<CUDA toolkit include dir>"
	if dir := cudaIncludeDir(); dir != "" {
		hint = "NVRTC searches no directory by default. Add --include-path=" + dir
	}
	return hint
}

// cudaIncludeDir reports a CUDA toolkit include directory on this host, or the
// empty string when none is found. It prefers the toolkit the environment
// names.
func cudaIncludeDir() string {
	var roots []string
	for _, env := range []string{"CUDA_HOME", "CUDA_PATH"} {
		if root := os.Getenv(env); root != "" {
			roots = append(roots, root)
		}
	}
	roots = append(roots, "/usr/local/cuda")
	for _, root := range roots {
		candidates, _ := filepath.Glob(filepath.Join(root, "targets", "*", "include"))
		candidates = append(candidates, filepath.Join(root, "include"))
		for _, dir := range candidates {
			if _, err := os.Stat(filepath.Join(dir, "cuda_runtime.h")); err == nil {
				return dir
			}
		}
	}
	return ""
}
