// Metalprof opens a portable Metal capture with upstream pprof and an inspector
// for the capture's MSL source, intermediate representation (AIR), and metrics.
// Viewing requires neither a Mac nor an Apple GPU.
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
	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

type exampleMetadata struct {
	Device         string  `json:"device"`
	Elements       int     `json:"elements"`
	Passes         int     `json:"passes"`
	Rounds         int     `json:"fma_rounds"`
	Verified       int     `json:"verified_results"`
	MaxError       float64 `json:"max_absolute_error"`
	TimingMode     string  `json:"timing_mode"`
	TimingReq      string  `json:"timing_requested"`
	TimingActual   string  `json:"timing_actual"`
	TimingScope    string  `json:"timing_scope"`
	FallbackReason string  `json:"timing_fallback_reason"`
}

func main() {
	log.SetFlags(0)
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	if len(os.Args) > 1 && os.Args[1] == "-extract" {
		if len(os.Args) != 4 {
			return fmt.Errorf("usage: metalprof -extract new-directory capture.pprof")
		}
		return extract(os.Args[2], os.Args[3])
	}
	if len(os.Args) > 1 && os.Args[1] == "-pack" {
		if len(os.Args) != 4 {
			return fmt.Errorf("usage: metalprof -pack output.pprof path/to/gpu.pprof")
		}
		return pack(os.Args[2], os.Args[3])
	}
	if len(os.Args) == 1 || len(os.Args) == 2 && slices.Contains([]string{"-h", "-help", "--help"}, os.Args[1]) {
		fmt.Println("Usage: metalprof [pprof flags] capture.pprof\n\n  -http=127.0.0.1:8090  open interactive pprof\n  -no_browser          serve without opening a browser\n  -top                 show kernel totals\n  -list=vector_add     show annotated source\n  -profile=live|replay select profile origin (default: live)\n  -observations        list attached typed observations\n\nTo embed an existing capture's files:\n  metalprof -pack output.pprof path/to/gpu.pprof\n\nTo extract source and artifacts:\n  metalprof -extract new-directory capture.pprof")
		return nil
	}

	selectedProfile := "live"
	showObservations := false
	var filteredArgs []string
	filteredArgs = append(filteredArgs, os.Args[0])

	for i := 1; i < len(os.Args)-1; i++ {
		arg := os.Args[i]
		if strings.HasPrefix(arg, "-profile=") {
			selectedProfile = strings.TrimPrefix(arg, "-profile=")
		} else if arg == "-observations" {
			showObservations = true
		} else {
			filteredArgs = append(filteredArgs, arg)
		}
	}
	lastArg := os.Args[len(os.Args)-1]
	filteredArgs = append(filteredArgs, lastArg)

	path, err := filepath.Abs(lastArg)
	if err != nil {
		return err
	}

	var cap *metalcapture.Capture
	var isTempDir bool
	var extractDir string

	fi, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("verify capture: %w", err)
	}

	if fi.IsDir() {
		cap, err = metalcapture.ReadDir(path)
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

		cap, err = metalcapture.Read(f)
		if err != nil {
			dir := filepath.Dir(path)
			if dCap, dErr := metalcapture.ReadDir(dir); dErr == nil {
				cap = dCap
				extractDir = dir
			} else {
				return fmt.Errorf("verify capture: %w", err)
			}
		}
	}

	if showObservations {
		if len(cap.Observations) == 0 {
			fmt.Println("No typed observations recorded in capture.")
			return nil
		}
		fmt.Printf("=== %d Typed Observation(s) ===\n", len(cap.Observations))
		for _, obs := range cap.Observations {
			nonAdditive := ""
			if obs.NonAdditive {
				nonAdditive = " [non-additive]"
			}
			valStr := fmt.Sprintf("%10.2f", obs.Value.FloatVal)
			if obs.Value.IsInt {
				valStr = fmt.Sprintf("%10d", obs.Value.IntVal)
			}
			idDetails := ""
			if obs.Pipeline != "" {
				idDetails += " pipeline: " + obs.Pipeline
			}
			if obs.DispatchID != nil {
				idDetails += fmt.Sprintf(" dispatch: #%d", *obs.DispatchID)
			}
			fmt.Printf("  [%s] %-26s %-16s %s %-12s (scope: %s, kind: %s%s)%s\n",
				obs.Origin, obs.ID, obs.Name, valStr, obs.Unit, obs.Scope, obs.Kind, idDetails, nonAdditive)
		}
		return nil
	}

	if extractDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		parent := filepath.Join(home, "tmp")
		if err := os.MkdirAll(parent, 0755); err != nil {
			return err
		}
		td, err := os.MkdirTemp(parent, "metalprof-")
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

	os.Args = filteredArgs
	fetch := &fetcher{path: path, dir: extractDir, cap: cap, selected: selectedProfile}
	return driver.PProf(&driver.Options{
		Fetch: fetch,
	})
}

func pack(output, input string) (err error) {
	if filepath.Base(input) != "gpu.pprof" {
		return fmt.Errorf("pack requires a capture directory's gpu.pprof")
	}
	dir := filepath.Dir(input)
	cap, err := metalcapture.ReadDir(dir)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return err
	}
	err = metalcapture.Write(f, cap)
	err = errors.Join(err, f.Close())
	if err != nil {
		os.Remove(output)
	}
	return err
}

func extract(dir, input string) (err error) {
	f, err := os.Open(input)
	if err != nil {
		return err
	}
	defer f.Close()

	cap, err := metalcapture.Read(f)
	if err != nil {
		return err
	}
	return metalcapture.WriteDir(dir, cap)
}

type fetcher struct {
	path, dir string
	cap       *metalcapture.Capture
	selected  string
}

func (f *fetcher) Fetch(src string, duration, timeout time.Duration) (*profile.Profile, string, error) {
	path, err := filepath.Abs(src)
	if err != nil {
		return nil, "", err
	}
	if path != f.path {
		return nil, "", fmt.Errorf("expected the verified profile path")
	}

	var p *profile.Profile
	if f.selected == "replay" {
		replayBytes, ok := f.cap.Files["replay.pprof"]
		if !ok {
			return nil, "", fmt.Errorf("replay profile selected (-profile=replay), but capture contains no enriched replay.pprof (run trace with -enrich)")
		}
		rp, err := profile.ParseData(replayBytes)
		if err != nil {
			return nil, "", fmt.Errorf("parse replay.pprof: %w", err)
		}
		p = rp
	} else {
		p = f.cap.Profile.Copy()
	}

	for _, fn := range p.Function {
		if fn.Filename == "" {
			// Do not automatically attribute unknown functions to the capture's MSL source.
			// Without an explicit verified source mapping or compilation manifest,
			// functions remain source-unmapped.
			continue
		}
		name := filepath.Base(fn.Filename)
		if _, ok := f.cap.Files[name]; !ok {
			return nil, "", fmt.Errorf("unbundled source %q", fn.Filename)
		}
		fn.Filename = filepath.Join(f.dir, name)
	}

	if f.selected != "replay" && len(f.cap.Metadata) > 0 {
		var meta exampleMetadata
		if err := json.Unmarshal(f.cap.Metadata, &meta); err == nil && meta.Device != "" {
			timingDesc := meta.TimingMode
			if meta.TimingActual != "" {
				timingDesc = fmt.Sprintf("%s (actual: %s, scope: %s)", meta.TimingReq, meta.TimingActual, meta.TimingScope)
				if meta.FallbackReason != "" {
					timingDesc += fmt.Sprintf(" [fallback: %s]", meta.FallbackReason)
				}
			}
			p.Comments = append(p.Comments,
				fmt.Sprintf("%s: %d elements, %d FMA rounds, %d passes; all results checked (timing: %s)",
					meta.Device, meta.Elements, meta.Rounds, meta.Passes, timingDesc),
			)
		}
	}

	if f.selected != "replay" {
		p.Comments = append(p.Comments,
			"Metal event intervals measure whole launches or command buffers. Source lines are navigation anchors, not instruction-level hotness.")
	}
	return p, "", nil
}
