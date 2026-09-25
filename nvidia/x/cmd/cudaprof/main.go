// Cudaprof records a CUDA program's device timeline with CUPTI and reads the
// resulting capture: kernel totals and source through upstream pprof, PTX and
// NVIDIA disassembly, idle gaps, a per-op comparison of two captures, the
// verdict of an ABBA timing campaign, and a Perfetto trace.
//
// Capture requires Linux and the CUPTI agent built by go generate. Every
// other command reads a capture file and requires neither a GPU nor CUDA.
//
// Run "cudaprof help" for the commands, and see README.md for the workflow.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/google/pprof/driver"
	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
	"github.com/tmc/lib/nvidia/x/tileir"
)

type exampleMetadata struct {
	Device     string          `json:"device"`
	Arch       string          `json:"architecture"`
	Elements   int             `json:"elements"`
	Passes     int             `json:"passes"`
	Rounds     int             `json:"fma_rounds"`
	Verified   int             `json:"verified_results"`
	MaxError   float64         `json:"max_absolute_error"`
	NCCL       *ncclMeta       `json:"nccl,omitempty"`
	PCSampling json.RawMessage `json:"pc_sampling,omitempty"`
}

type ncclMeta struct {
	Mode     string                 `json:"mode"`
	Ranks    int                    `json:"ranks"`
	Verified map[string]map[int]int `json:"verified_by_rank"`
}

func main() {
	log.SetFlags(0)
	if err := run(); err != nil {
		var exitErr *exitCodeError
		if errors.As(err, &exitErr) {
			log.Print(exitErr)
			os.Exit(exitErr.Code)
		}
		log.Fatal(err)
	}
}

type exitCodeError struct {
	Code int
	Err  error
}

func (e *exitCodeError) Error() string { return e.Err.Error() }
func (e *exitCodeError) Unwrap() error { return e.Err }

func run() error {
	if len(os.Args) == 1 || len(os.Args) == 2 && isHelpFlag(os.Args[1]) {
		fmt.Print(usage)
		return nil
	}
	if cmd, ok := lookupCommand(os.Args[1]); ok {
		return cmd.invoke(os.Args[1:])
	}
	path, err := filepath.Abs(os.Args[len(os.Args)-1])
	if err != nil {
		return err
	}

	// Determine if path is a file or a directory
	var cap *cudacapture.Capture
	var isTempDir bool
	var extractDir string

	fi, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("verify capture: %w", err)
	}

	if fi.IsDir() {
		cap, err = cudacapture.ReadDir(path)
		if err != nil {
			return fmt.Errorf("verify capture: %w", err)
		}
		extractDir = path
	} else if isBundlePath(path) {
		cap, err = readBundleCapturePath(path, cudacapture.Limits{})
		if err != nil {
			return fmt.Errorf("verify capture: %w", err)
		}
		// Materialize embedded files into a temporary directory under ~/tmp for upstream pprof
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		parent := filepath.Join(home, "tmp")
		if err := os.MkdirAll(parent, 0755); err != nil {
			return err
		}
		td, err := os.MkdirTemp(parent, "cudaprof-")
		if err != nil {
			return err
		}
		isTempDir = true
		extractDir = td
		defer func() {
			if isTempDir {
				os.RemoveAll(extractDir)
			}
		}()

		for name, data := range cap.Files {
			if err := os.WriteFile(filepath.Join(extractDir, name), data, 0600); err != nil {
				return err
			}
		}
	} else {
		f, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("open capture file: %w", err)
		}
		defer f.Close()

		cap, err = cudacapture.Read(f)
		if err != nil {
			// If reading embedded capture fails, try reading as directory next to it or as bundle
			dir := filepath.Dir(path)
			if dCap, dErr := cudacapture.ReadDir(dir); dErr == nil {
				cap = dCap
				extractDir = dir
			} else if bCap, bErr := readBundleCapturePath(path, cudacapture.Limits{}); bErr == nil {
				cap = bCap
			} else {
				return fmt.Errorf("verify capture: %w", err)
			}
		} else {
			// Materialize embedded files into a temporary directory under ~/tmp for upstream pprof
			home, err := os.UserHomeDir()
			if err != nil {
				return err
			}
			parent := filepath.Join(home, "tmp")
			if err := os.MkdirAll(parent, 0755); err != nil {
				return err
			}
			td, err := os.MkdirTemp(parent, "cudaprof-")
			if err != nil {
				return err
			}
			isTempDir = true
			extractDir = td
			defer func() {
				if isTempDir {
					os.RemoveAll(extractDir)
				}
			}()

			for name, data := range cap.Files {
				if err := os.WriteFile(filepath.Join(extractDir, name), data, 0600); err != nil {
					return err
				}
			}
		}
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		// A source-fusion failure invalidates the source tier and nothing else.
		// The kernel ranking a reader comes here for is built from the CUBIN
		// sections and the disassembly, so refusing the whole profile made the
		// tool unusable exactly when it was most needed: on a fresh profile of
		// code that was just changed. Degrade to the intact tier and say so on
		// stderr, where it travels with the ranking rather than sitting in a
		// document someone has to remember to read.
		var fusion *cudacapture.SourceFusionError
		if errors.As(err, &fusion) && code != nil {
			fmt.Fprintf(os.Stderr, "cudaprof: %v\n", fusion)
			fmt.Fprintf(os.Stderr, "cudaprof: rendering the kernel-level ranking without source attribution; source and PTX views of this capture are unavailable\n")
		} else {
			// A v3 capture can be useful as an ordinary pprof profile even when
			// source, PTX, CUBIN, or SASS was unavailable. Object navigation then
			// fails explicitly, while duration and typed profile views remain usable.
			if len(cap.CodeUnavailable) == 0 {
				return err
			}
			code = &cudacapture.Code{}
		}
	}
	tool, err := newObjectTool(cap, code, extractDir)
	if err != nil && len(cap.CodeUnavailable) == 0 {
		return err
	}
	if tool == nil {
		tool = &objectTool{images: make(map[string]*objectImage)}
	}
	var limits cudacapture.Limits
	fetch := &fetcher{path: path, dir: extractDir, cap: cap, tool: tool, limits: limits}
	args, allowUnknownCoverage := stripCoverageFlag(os.Args[1:])
	if handled, err := retainedDisasm(os.Stdout, args, cap.Profile, tool); handled {
		return err
	}
	// Refuse an artifact that records its own failure before handing it to
	// pprof, which would render it as a valid, empty profile and exit 0.
	if err := preflightCapture(cap); err != nil {
		return err
	}
	// State what the shares below are shares of, and refuse to rank by a total
	// the profile cannot relate to elapsed time.
	if err := preflightCoverage(os.Stderr, cap, args, allowUnknownCoverage); err != nil {
		return err
	}
	// Ordered after the coverage statement so a selected column that the
	// statement does not describe is named directly beneath it.
	if err := preflightSampleIndex(os.Stderr, cap, args); err != nil {
		return err
	}
	reportProvenance(os.Stderr, cap)
	os.Args = append(os.Args[:1], args...)
	return driver.PProf(&driver.Options{Obj: tool, Fetch: fetch})
}

func pack(output, input string) (err error) {
	if filepath.Base(input) != "gpu.pprof" {
		return fmt.Errorf("pack requires a capture directory's gpu.pprof")
	}
	dir := filepath.Dir(input)
	cap, err := cudacapture.ReadDir(dir)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	err = cudacapture.Write(f, cap)
	err = errors.Join(err, f.Close())
	if err != nil {
		os.Remove(output)
	}
	return err
}

// extract unpacks a capture under explicit limits. A zero Limits keeps the
// package defaults, so an artifact written under a raised producer budget must
// be asked for by name on both sides.
func extract(dir, input string, limits cudacapture.Limits) (err error) {
	f, err := os.Open(input)
	if err != nil {
		return err
	}
	defer f.Close()

	cap, err := cudacapture.ReadLimits(f, limits)
	if err != nil {
		return err
	}
	return cudacapture.WriteDirLimits(dir, cap, limits)
}

type fetcher struct {
	path, dir string
	cap       *cudacapture.Capture
	tool      *objectTool
	limits    cudacapture.Limits
}

func (f *fetcher) Fetch(src string, duration, timeout time.Duration) (*profile.Profile, string, error) {
	path, err := filepath.Abs(src)
	if err != nil {
		return nil, "", err
	}
	if path != f.path {
		file, err := os.Open(path)
		if err != nil {
			return nil, "", err
		}
		defer file.Close()
		if secCap, err := cudacapture.ReadLimits(file, f.limits); err == nil && secCap.Profile != nil {
			p := secCap.Profile.Copy()
			resolveSourceLocations(p, secCap.Files, f.dir)
			return p, path, nil
		}
		if _, err := file.Seek(0, 0); err == nil {
			if p, err := profile.Parse(file); err == nil {
				resolveSourceLocations(p, nil, f.dir)
				return p, path, nil
			}
		}
		return nil, "", fmt.Errorf("read profile %s", path)
	}
	p := f.cap.Profile.Copy()
	for _, fn := range p.Function {
		name := filepath.Base(fn.Filename)
		if _, ok := f.cap.Files[name]; ok {
			fn.Filename = filepath.Join(f.dir, name)
		}
	}
	resolveSourceLocations(p, f.cap.Files, f.dir)
	if f.cap.Code.CUBIN != "" || f.cap.Code.PTX != "" || f.cap.Code.TileIR != "" {
		for _, m := range p.Mapping {
			base := filepath.Base(m.File)
			if base != filepath.Base(f.cap.Code.CUBIN) && base != filepath.Base(f.cap.Code.PTX) && base != filepath.Base(f.cap.Code.TileIR) {
				continue
			}
			m.File = filepath.Join(f.dir, base)
		}
	}
	if f.tool != nil && f.tool.ptx != nil {
		f.tool.mapPTX(p)
	}
	if f.tool != nil && f.tool.tileir != nil {
		f.tool.mapTileIR(p)
	}

	if f.cap.Code.TileIR != "" && len(f.cap.Files[f.cap.Code.TileIR]) > 0 {
		tileData := f.cap.Files[f.cap.Code.TileIR]
		var mod *tileir.Module
		if tileir.IsTileIRBytecode(tileData) {
			if bm, err := tileir.DecodeBytecodeBytes(tileData); err == nil {
				mod = bm.ToModule()
			}
		} else {
			mod, _ = tileir.ParseBytes(tileData)
		}
		if mod != nil {
			tmaOps, mmaOps := 0, 0
			for _, fn := range mod.Functions {
				for _, op := range fn.Operations {
					if op.IsTMA() {
						tmaOps++
					}
					if op.IsMMA() {
						mmaOps++
					}
				}
			}
			p.Comments = append(p.Comments,
				fmt.Sprintf("Tile IR (%s): %d functions, %d TMA operations, %d Tensor Core MMAs",
					f.cap.Code.TileIR, len(mod.Functions), tmaOps, mmaOps))
		}
	}

	if remarks, rName, _ := capturefiles.Remarks(f.cap); len(remarks) > 0 {
		var tmaFails, tcMMAs, tcFails int
		for _, r := range remarks {
			if r.IsTMAFailure() {
				tmaFails++
			}
			if r.IsTensorCoreMMA() {
				tcMMAs++
			}
			if r.IsTensorCoreFailure() {
				tcFails++
			}
		}
		p.Comments = append(p.Comments,
			fmt.Sprintf("Compiler remarks (%s): %d total (%d Tensor Core MMA, %d TMA failures, %d Tensor Core fallbacks)",
				rName, len(remarks), tcMMAs, tmaFails, tcFails))
		for _, r := range remarks {
			if r.IsTMAFailure() {
				p.Comments = append(p.Comments,
					fmt.Sprintf("WARNING: TMA failure at %s:%d: %s (reason: %s)", r.File, r.Line, r.Remark, r.Reason))
			}
		}
	}

	var meta exampleMetadata
	if len(f.cap.Metadata) > 0 {
		if err := json.Unmarshal(f.cap.Metadata, &meta); err == nil && meta.Device != "" {
			p.Comments = append(p.Comments,
				fmt.Sprintf("%s (%s): %d elements, %d FMA rounds, %d passes; all results checked", meta.Device, meta.Arch, meta.Elements, meta.Rounds, meta.Passes),
			)
			if meta.NCCL != nil {
				p.Comments = append(p.Comments,
					fmt.Sprintf("NCCL mode: %s across %d ranks; all collective results checked", meta.NCCL.Mode, meta.NCCL.Ranks),
				)
			}
		}
	}

	if meta.PCSampling != nil {
		p.Comments = append(p.Comments,
			"Profile includes hardware PC samples collected via CUPTI and attributed to SASS, PTX and source.",
			"SASS comes from the captured CUBIN's nvdisasm output. SASS coordinates are CUBIN file offsets; PTX coordinates are text byte offsets, not device PCs.")
	} else {
		p.Comments = append(p.Comments,
			"Profile values describe recorded scopes. SASS entry and PTX source rows, when present, are navigation anchors, not instruction costs or PC samples.",
			"SASS comes from the captured CUBIN's nvdisasm output. SASS coordinates are CUBIN file offsets; PTX coordinates are text byte offsets, not device PCs.")
	}
	return p, "", nil
}

const explainUsage = "usage: cudaprof explain <ptx-instruction-or-keyword>"

func runExplainCommand(args []string) error {
	if len(args) == 0 {
		return errors.New(explainUsage)
	}
	stmt := strings.Join(args, " ")
	info := cudaptx.Explain(stmt)
	if info != nil && (info.Doc != nil || len(info.Modifiers) > 0) {
		fmt.Print(info.Markdown())
		return nil
	}
	doc := cudaptx.LookupDoc(stmt)
	if doc != nil {
		h := &cudaptx.HoverInfo{
			Opcode:    doc.Name,
			Mnemonic:  doc.Name,
			Doc:       doc,
			MinPTX:    doc.MinPTX,
			MinTarget: doc.MinTarget,
		}
		fmt.Print(h.Markdown())
		return nil
	}
	return fmt.Errorf("no PTX documentation found for %q", stmt)
}

// sourceCandidate is a file that might define a profiled function, named so
// that a resolved location can be reported against a stable path.
type sourceCandidate struct {
	path    string
	content []byte
}

// captureSources returns the capture's embedded sources in a fixed order. The
// map iterates randomly, which previously let two runs of the same capture
// attribute one function to two different files.
func captureSources(files map[string][]byte) []sourceCandidate {
	names := make([]string, 0, len(files))
	for name := range files {
		switch filepath.Ext(name) {
		case ".cu", ".cpp", ".c", ".go":
			names = append(names, name)
		}
	}
	sort.Strings(names)
	out := make([]sourceCandidate, 0, len(names))
	for _, name := range names {
		out = append(out, sourceCandidate{path: name, content: files[name]})
	}
	return out
}

// neighbouringSources returns the .cu files stored alongside the capture. The
// process working directory is deliberately not searched: a file that merely
// happens to be in the directory cudaprof was launched from has no established
// relationship to the compiled artifact.
func neighbouringSources(dir string) []sourceCandidate {
	if dir == "" {
		return nil
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.cu"))
	sort.Strings(matches)
	var out []sourceCandidate
	for _, m := range matches {
		data, err := os.ReadFile(m)
		if err != nil {
			continue
		}
		out = append(out, sourceCandidate{path: m, content: data})
	}
	return out
}

// uniqueDefinition returns the single candidate that defines name. If several
// do, the name does not identify a location and ok is false: reporting the
// first match would attribute samples to whichever file happened to sort or
// hash first, which is the guess this function exists to refuse.
func uniqueDefinition(candidates []sourceCandidate, name string) (path string, line int, ok bool) {
	for _, c := range candidates {
		n := findFunctionLine(string(c.content), name)
		if n == 0 {
			continue
		}
		if ok {
			return "", 0, false
		}
		path, line, ok = c.path, n, true
	}
	return path, line, ok
}

func resolveSourceLocations(p *profile.Profile, files map[string][]byte, dir string) {
	if p == nil {
		return
	}
	for _, fn := range p.Function {
		if fn.Filename != "" && filepath.IsAbs(fn.Filename) && fn.StartLine > 0 {
			continue
		}
		// Search the capture's own files first: those are the sources the
		// artifact was built from. Only if none of them defines the function
		// do we look at .cu files sitting next to the capture.
		if path, line, ok := uniqueDefinition(captureSources(files), fn.Name); ok {
			if dir != "" {
				fn.Filename = filepath.Join(dir, path)
			} else {
				fn.Filename = path
			}
			fn.StartLine = int64(line)
			continue
		}
		if path, line, ok := uniqueDefinition(neighbouringSources(dir), fn.Name); ok {
			if abs, err := filepath.Abs(path); err == nil {
				path = abs
			}
			fn.Filename = path
			fn.StartLine = int64(line)
		}
		// Otherwise the location stays unset: unknown, rather than a guess.
	}
	for _, loc := range p.Location {
		for i := range loc.Line {
			if loc.Line[i].Function != nil && loc.Line[i].Line == 0 && loc.Line[i].Function.StartLine > 0 {
				loc.Line[i].Line = loc.Line[i].Function.StartLine
			}
		}
	}
}

// findFunctionLine returns the 1-based line in content that defines name, or 0
// if content does not appear to define it. Reporting a line requires evidence
// that name is defined there: a comment mentioning the function, a forward
// declaration, and a call site all name it without being its source location,
// and attributing samples to one of those misplaces every line in the profile.
// Callers must treat 0 as unknown rather than falling back to a guess.
func findFunctionLine(content, name string) int {
	if name == "" {
		return 0
	}
	for i, line := range strings.Split(content, "\n") {
		// Drop a trailing line comment before anything else: the ';' test
		// below runs on the line's last byte, so a declaration written as
		// "void saxpy(int n);   // forward declaration" no longer ends in ';'
		// and would read as a definition.
		if j := strings.Index(line, "//"); j >= 0 {
			line = line[:j]
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "*") || strings.HasPrefix(trimmed, "/*") {
			continue
		}
		// A line ending in ';' declares or calls the function; it does not
		// define it. A definition ends with '{', or continues onto the next
		// line with its parameter list still open.
		if strings.HasSuffix(trimmed, ";") {
			continue
		}
		if !namesFunction(trimmed, name) {
			continue
		}
		return i + 1
	}
	return 0
}

// namesFunction reports whether line applies the call syntax to exactly the
// identifier name. Matching a bare substring would accept my_saxpy for saxpy.
func namesFunction(line, name string) bool {
	for off := 0; ; {
		i := strings.Index(line[off:], name)
		if i < 0 {
			return false
		}
		i += off
		off = i + len(name)
		if i > 0 && isIdentByte(line[i-1]) {
			continue
		}
		rest := strings.TrimLeft(line[off:], " \t")
		if strings.HasPrefix(rest, "(") {
			return true
		}
	}
}

func isIdentByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

// preflightCapture refuses an artifact that records its own failure. A failed
// capture still writes a well-formed pprof -- a couple of kilobytes with zero
// activity records -- and pprof renders it as an empty but valid profile and
// exits 0. Nothing on that path says the numbers are missing rather than
// small, which is the mistake the artifact invites. Reading it fails closed;
// the status is still reachable through advise, which reports the failure as a
// finding rather than pretending to measure.
func preflightCapture(cap *cudacapture.Capture) error {
	reason, failed := capturefiles.Failure(cap)
	if !failed {
		return nil
	}
	return fmt.Errorf("this capture records its own failure: %s\n"+
		"\tIt carries no measurement, and an empty profile is not a fast one.\n"+
		"\tUse 'cudaprof advise' to see the failure as a finding, or re-run the capture.", reason)
}
