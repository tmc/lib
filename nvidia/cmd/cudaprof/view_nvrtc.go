package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// A recompileSource is the CUDA C++ source a capture retained, together with
// the artifacts the capture itself holds. The viewer recompiles the source
// under caller-chosen NVRTC options and compares the result against those
// artifacts, so a capture taken without --use_fast_math can be read beside the
// same source compiled with it.
type recompileSource struct {
	File     string   `json:"file"`
	Arch     string   `json:"arch,omitempty"`
	Options  []string `json:"options,omitempty"`
	Recorded bool     `json:"recorded"`
	// Suggested is the option list the panel offers for the second side:
	// Options plus --use_fast_math.
	Suggested []string `json:"suggested,omitempty"`

	Source string `json:"-"`
	PTX    string `json:"-"`
	SASS   string `json:"-"`
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
	A       *nvrtcVariant  `json:"a"`
	B       *nvrtcVariant  `json:"b"`
	Deltas  []variantDelta `json:"deltas,omitempty"`
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
	opts := []nvrtcx.Option{nvrtcx.WithName(name)}
	for _, flag := range options {
		opts = append(opts, nvrtcx.WithFlag(flag))
	}
	mod, err := nvrtcx.Compile(src, opts...)
	if err != nil {
		v.Error = err.Error()
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
func captureVariant(src *recompileSource) *nvrtcVariant {
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

// diffVariants reports the counts that differ between two variants, totals
// first, then per-kernel SASS instructions, then opcodes ordered by the size of
// the change. Counts equal on both sides are left out.
func diffVariants(a, b *nvrtcVariant) []variantDelta {
	if a == nil || b == nil {
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

// newRecompileHandler serves /api/recompile. Parameters a and b each hold a
// space-separated NVRTC option list, or "capture" for the capture's own
// artifacts.
func newRecompileHandler(src *recompileSource, compile variantCompiler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if src == nil || src.Source == "" {
			http.Error(w, "recompile: capture retained no CUDA C++ source", http.StatusNotFound)
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
		resp := recompileResponse{A: side("a"), B: side("b"), Caveat: staticCountCaveat}
		resp.Deltas = diffVariants(resp.A, resp.B)
		resp.Elapsed = time.Since(start).Round(time.Millisecond).String()
		json.NewEncoder(w).Encode(resp)
	}
}

// defaultRecompileOptions are the options the panel starts from when the
// capture did not record the ones it was built with. They mirror the options
// ds4-min passes today; the panel marks them as assumed.
func defaultRecompileOptions(arch string) []string {
	if arch == "" {
		return []string{"--generate-line-info"}
	}
	return []string{"--gpu-architecture=" + arch, "--generate-line-info"}
}
