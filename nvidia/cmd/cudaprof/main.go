// Cudaprof opens a portable CUDA capture with upstream pprof and a reader for
// the capture's PTX and NVIDIA disassembly. Viewing requires neither a GPU nor CUDA.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/google/pprof/driver"
	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
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
	if len(os.Args) > 1 && os.Args[1] == "capture" {
		return runCaptureCommand(os.Args[2:])
	}
	if len(os.Args) > 1 && os.Args[1] == "inspect" {
		return runInspectCommand(os.Args[2:])
	}
	if len(os.Args) > 1 && (os.Args[1] == "cfg" || os.Args[1] == "-cfg" || strings.HasPrefix(os.Args[1], "-cfg=") || strings.HasPrefix(os.Args[1], "--cfg")) {
		return runCFGCommand(os.Args[1:])
	}
	if len(os.Args) > 1 && (os.Args[1] == "bbcfg" || os.Args[1] == "-bbcfg" || strings.HasPrefix(os.Args[1], "-bbcfg=") || strings.HasPrefix(os.Args[1], "--bbcfg")) {
		return runCFGCommand(os.Args[1:])
	}
	if len(os.Args) > 1 && os.Args[1] == "-extract" {
		args, limits, err := parseReadLimits(os.Args[2:])
		if err != nil {
			return err
		}
		if len(args) != 2 {
			return fmt.Errorf("usage: cudaprof -extract [--capture-file-bytes=N] [--capture-total-bytes=N] new-directory capture.pprof")
		}
		return extract(args[0], args[1], limits)
	}
	if len(os.Args) > 1 && os.Args[1] == "-pack" {
		if len(os.Args) != 4 {
			return fmt.Errorf("usage: cudaprof -pack output.pprof path/to/gpu.pprof")
		}
		return pack(os.Args[2], os.Args[3])
	}
	if len(os.Args) == 1 || len(os.Args) == 2 && slices.Contains([]string{"-h", "-help", "--help"}, os.Args[1]) {
		fmt.Println("Usage: cudaprof [pprof flags] capture.pprof\n\n  -http=127.0.0.1:8087  open interactive pprof\n  -no_browser          serve without opening a browser\n  -top                 show kernel totals\n  -list=compute        show annotated source\n  -disasm=compute      show PTX and SASS\n  -cfg[=kernel]        print control flow graph (DOT format)\n  -bbcfg[=kernel]      print basic block control flow graph (DOT format)\n\nTo embed an existing capture's files:\n  cudaprof -pack output.pprof path/to/gpu.pprof\n\nTo extract source, assembly, control flow graphs, and Go trace:\n  cudaprof -extract new-directory capture.pprof\n\nTo view or render control flow graphs:\n  cudaprof cfg [kernel] capture.pprof | dot -Tsvg -o graph.svg\n  cudaprof inspect --cfg [kernel] capture.pprof")
		return nil
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
	} else {
		f, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("open capture file: %w", err)
		}
		defer f.Close()

		cap, err = cudacapture.Read(f)
		if err != nil {
			// If reading embedded capture fails, try reading as directory next to it
			dir := filepath.Dir(path)
			if dCap, dErr := cudacapture.ReadDir(dir); dErr == nil {
				cap = dCap
				extractDir = dir
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
		// A v3 capture can be useful as an ordinary pprof profile even when
		// source, PTX, CUBIN, or SASS was unavailable. Object navigation then
		// fails explicitly, while duration and typed profile views remain usable.
		if len(cap.CodeUnavailable) == 0 {
			return err
		}
		code = &cudacapture.Code{}
	}
	tool, err := newObjectTool(cap, code, extractDir)
	if err != nil && len(cap.CodeUnavailable) == 0 {
		return err
	}
	if tool == nil {
		tool = &objectTool{images: make(map[string]*objectImage)}
	}
	fetch := &fetcher{path: path, dir: extractDir, cap: cap, tool: tool}
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
}

func (f *fetcher) Fetch(src string, duration, timeout time.Duration) (*profile.Profile, string, error) {
	path, err := filepath.Abs(src)
	if err != nil {
		return nil, "", err
	}
	if path != f.path {
		return nil, "", fmt.Errorf("expected the verified profile path")
	}
	p := f.cap.Profile.Copy()
	for _, fn := range p.Function {
		name := filepath.Base(fn.Filename)
		if _, ok := f.cap.Files[name]; ok {
			fn.Filename = filepath.Join(f.dir, name)
		}
	}
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
			"CUDA event intervals measure whole launches. SASS entry and PTX source rows are navigation anchors, not instruction costs or PC samples.",
			"SASS comes from the captured CUBIN's nvdisasm output. SASS coordinates are CUBIN file offsets; PTX coordinates are text byte offsets, not device PCs.")
	}
	return p, "", nil
}
