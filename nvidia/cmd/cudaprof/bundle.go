package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

// FlightRecorderBundle holds diagnostic artifacts extracted in-memory from a
// flight recorder archive (.tar or .tar.gz).
type FlightRecorderBundle struct {
	Manifest cudaflightrecorder.Manifest
	GPUPprof []byte
	GoTrace  []byte
	Files    map[string][]byte
}

// ReadFlightRecorderBundle reads a diagnostic bundle archive from r. It transparently
// decompresses gzip data when present.
func ReadFlightRecorderBundle(r io.Reader) (*FlightRecorderBundle, error) {
	dr, closer, err := openDecompressedReader(r)
	if err != nil {
		return nil, fmt.Errorf("read bundle: %w", err)
	}
	defer closer.Close()

	bundle, err := cudaflightrecorder.ReadBundle(dr)
	if err != nil {
		return nil, fmt.Errorf("read bundle: %w", err)
	}

	fb := &FlightRecorderBundle{
		Manifest: bundle.Manifest,
		Files:    bundle.Files,
	}
	if fb.Files == nil {
		fb.Files = make(map[string][]byte)
	}

	if data, ok := fb.Files["gpu.pprof"]; ok {
		fb.GPUPprof = data
	}
	if data, ok := fb.Files["go.trace"]; ok {
		fb.GoTrace = data
	} else if data, ok := fb.Files["trace.out"]; ok {
		fb.GoTrace = data
	}

	if _, ok := fb.Files["manifest.json"]; !ok {
		if manifestBytes, err := json.MarshalIndent(fb.Manifest, "", "  "); err == nil {
			fb.Files["manifest.json"] = manifestBytes
		}
	}

	return fb, nil
}

// ReadFlightRecorderBundlePath reads a diagnostic bundle from the filesystem.
func ReadFlightRecorderBundlePath(path string) (*FlightRecorderBundle, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open bundle: %w", err)
	}
	defer f.Close()
	return ReadFlightRecorderBundle(f)
}

// ToCapture converts the in-memory bundle into a cudacapture.Capture.
func (b *FlightRecorderBundle) ToCapture() (*cudacapture.Capture, error) {
	if b == nil {
		return nil, errors.New("nil bundle")
	}

	var prof *profile.Profile
	var baseCap *cudacapture.Capture

	gpuBytes := b.GPUPprof
	if len(gpuBytes) == 0 && len(b.Files["gpu.pprof"]) > 0 {
		gpuBytes = b.Files["gpu.pprof"]
	}

	if len(gpuBytes) > 0 {
		if c, err := cudacapture.Read(bytes.NewReader(gpuBytes)); err == nil {
			baseCap = c
			prof = c.Profile
		} else if p, err := profile.ParseData(gpuBytes); err == nil {
			prof = p
		}
	}
	if prof == nil {
		prof = &profile.Profile{}
	}

	manifestBytes := b.Files["manifest.json"]
	if len(manifestBytes) == 0 {
		manifestBytes, _ = json.MarshalIndent(b.Manifest, "", "  ")
	}

	cap := &cudacapture.Capture{
		Profile:  prof,
		Files:    make(map[string][]byte, len(b.Files)),
		Metadata: manifestBytes,
	}

	if baseCap != nil {
		cap.Code = baseCap.Code
		cap.Modules = baseCap.Modules
		cap.CodeUnavailable = baseCap.CodeUnavailable
		cap.Descriptors = baseCap.Descriptors
		cap.Observations = baseCap.Observations
		for k, v := range baseCap.Files {
			cap.Files[k] = v
		}
	}

	for k, v := range b.Files {
		cap.Files[k] = v
	}

	inferred := inferCodeFiles(cap.Files)
	if cap.Code.Source == "" {
		cap.Code.Source = inferred.Source
	}
	if cap.Code.PTX == "" {
		cap.Code.PTX = inferred.PTX
	}
	if cap.Code.CUBIN == "" {
		cap.Code.CUBIN = inferred.CUBIN
	}
	if cap.Code.SASS == "" {
		cap.Code.SASS = inferred.SASS
	}
	if cap.Code.TileIR == "" {
		cap.Code.TileIR = inferred.TileIR
	}
	if cap.Code.CFG == "" {
		cap.Code.CFG = inferred.CFG
	}
	if cap.Code.BBCFG == "" {
		cap.Code.BBCFG = inferred.BBCFG
	}

	if _, ok := cap.Files["capture-status.json"]; !ok {
		// capture_success qualifies the target generation, so it stays true
		// here for the same reason the live writer leaves it true on a lossy
		// run. run_completeness is the field that answers whether the run
		// itself qualified, and publishing it by the shared rule is what keeps
		// a bundle's success boolean from standing alone: without it a bundle
		// that dropped every record but one read exactly like a clean one.
		dropped := uint64(b.Manifest.GPUDropped)
		statusData, _ := json.Marshal(map[string]any{
			"status": map[string]any{
				"capture_success":     true,
				"collector_finalized": true,
				"transport_ready":     true,
				"activity_records":    uint64(b.Manifest.GPURecords),
				"dropped_records":     dropped,
				"run_completeness":    runCompletenessOf(true, true, 0, dropped),
				"outcome":             b.Manifest.Reason,
			},
		})
		cap.Files["capture-status.json"] = statusData
	}

	return cap, nil
}

func readBundleCapturePath(path string, limits cudacapture.Limits) (*cudacapture.Capture, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stat bundle: %w", err)
	}
	maxInput := limits.InputBytes
	if maxInput == 0 {
		maxInput = cudacapture.MaxCaptureInputBytes
	}
	if fi.Size() > maxInput {
		return nil, fmt.Errorf("bundle file %q exceeds input size limit %d", path, maxInput)
	}

	bundle, err := ReadFlightRecorderBundlePath(path)
	if err != nil {
		return nil, err
	}
	return bundle.ToCapture()
}

func isBundlePath(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasSuffix(lower, ".tar.gz") || strings.HasSuffix(lower, ".tgz") || strings.HasSuffix(lower, ".tar")
}

func openDecompressedReader(r io.Reader) (io.Reader, io.Closer, error) {
	br := bufio.NewReader(r)
	magic, err := br.Peek(2)
	if err == nil && len(magic) == 2 && magic[0] == 0x1f && magic[1] == 0x8b {
		gz, err := gzip.NewReader(br)
		if err != nil {
			return nil, nil, fmt.Errorf("open gzip reader: %w", err)
		}
		return gz, gz, nil
	}
	return br, io.NopCloser(br), nil
}

// inferCodeFiles picks one artifact of each kind out of a bundle's files. A
// name of the form kernels.<ext> wins; otherwise the first in sorted order is
// taken, so that two runs over the same bundle agree. Iterating the map
// directly made the choice depend on Go's map ordering, which could pair one
// run's PTX with another run's disassembly.
//
// Nothing here establishes that the selected artifacts came from the same
// compilation: a bundle assembled by hand can hold a PTX and a SASS built from
// different sources, and their file and line records will fuse without
// complaint. Artifacts recorded by a capture are produced together and do not
// have this problem.
func inferCodeFiles(files map[string][]byte) cudacapture.CodeFiles {
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)

	pick := func(dst *string, canonical string, match func(string) bool) {
		for _, name := range names {
			if !match(name) {
				continue
			}
			if name == canonical {
				*dst = name
				return
			}
			if *dst == "" {
				*dst = name
			}
		}
	}
	hasSuffix := func(suffixes ...string) func(string) bool {
		return func(name string) bool {
			for _, s := range suffixes {
				if strings.HasSuffix(name, s) {
					return true
				}
			}
			return false
		}
	}

	var code cudacapture.CodeFiles
	pick(&code.Source, "kernels.cu", hasSuffix(".cu", ".cpp"))
	pick(&code.PTX, "kernels.ptx", hasSuffix(".ptx"))
	pick(&code.CUBIN, "kernels.cubin", hasSuffix(".cubin"))
	pick(&code.SASS, "kernels.sass", hasSuffix(".sass"))
	pick(&code.TileIR, "kernels.tileir", hasSuffix(".tileir", ".mlir"))
	pick(&code.BBCFG, "kernels.bbcfg.dot", hasSuffix(".bbcfg.dot"))
	pick(&code.CFG, "kernels.cfg.dot", func(name string) bool {
		return strings.HasSuffix(name, ".cfg.dot") ||
			strings.HasSuffix(name, ".dot") && !strings.HasSuffix(name, ".bbcfg.dot")
	})
	return code
}
