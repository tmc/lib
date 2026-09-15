package cuptibench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

const integerPolicy = "scale=1; rounding=none; finite nonnegative integers below 2^53 only"

type exportManifest struct {
	Version    int
	Conversion string
	Profiles   []exportProfile
	Results    []*Result
	Coverage   []exportCoverage
}

type exportProfile struct {
	Path        string
	SampleTypes []cudaprof.SampleType
	Captures    []string
}

type exportCoverage struct {
	Capture  string
	Exported []string
	Excluded []string
	Note     string
}

type exportPartition struct {
	builder  *cudaprof.Builder
	schema   []cudaprof.SampleType
	captures []string
}

func validateExportResult(r *Result) error {
	if r == nil || r.Operations <= 0 {
		return fmt.Errorf("cuptibench: invalid profile result")
	}
	if r.Aggregate != nil {
		a := r.Aggregate
		if a.Aggregate != nil {
			return fmt.Errorf("cuptibench: nested aggregate capture")
		}
		if r.Metadata["mode"] != "kernels" || a.Metadata["mode"] != "whole" {
			return fmt.Errorf("cuptibench: aggregate must be a whole capture associated with kernel attribution")
		}
		if a.Metadata["capture_id"] == r.Metadata["capture_id"] || r.Metadata["aggregate_capture_id"] != a.Metadata["capture_id"] {
			return fmt.Errorf("cuptibench: aggregate requires a distinct linked capture identity")
		}
		if a.Operations != r.Operations || a.Metadata["benchmark"] != r.Metadata["benchmark"] || a.Metadata["run_id"] != r.Metadata["run_id"] {
			return fmt.Errorf("cuptibench: aggregate workload identity differs")
		}
		if err := validateExportResult(a); err != nil {
			return fmt.Errorf("cuptibench: aggregate: %w", err)
		}
	}
	q := r.Quality
	if !q.AllPassesSubmitted || !q.Decoded || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStarted != q.PassesStopped {
		return fmt.Errorf("cuptibench: cannot export incomplete or lossy capture")
	}
	for _, key := range []string{"capture_id", "benchmark", "run_id", "mode"} {
		if r.Metadata[key] == "" {
			return fmt.Errorf("cuptibench: missing profile metadata %q", key)
		}
	}
	ids := make(map[string]bool)
	for _, v := range r.Ranges {
		if v.ID == "" || ids[v.ID] {
			return fmt.Errorf("cuptibench: missing or duplicate range ID %q", v.ID)
		}
		ids[v.ID] = true
		seen := make(map[string]bool)
		for _, m := range v.Metrics {
			if m.Name == "" || seen[m.Name] {
				return fmt.Errorf("cuptibench: missing or duplicate metric %q", m.Name)
			}
			seen[m.Name] = true
			if math.IsNaN(m.Value) || math.IsInf(m.Value, 0) {
				return fmt.Errorf("cuptibench: nonfinite metric %q", m.Name)
			}
		}
	}
	parents := make(map[string]string)
	for _, v := range r.Ranges {
		if v.ParentID != "" && !ids[v.ParentID] {
			return fmt.Errorf("cuptibench: unknown parent %q", v.ParentID)
		}
		parents[v.ID] = v.ParentID
	}
	for id := range ids {
		seen := make(map[string]bool)
		for at := id; at != ""; at = parents[at] {
			if seen[at] {
				return fmt.Errorf("cuptibench: cyclic range tree")
			}
			seen[at] = true
		}
	}
	return nil
}

func integerCounter(v float64) (int64, error) {
	// At 2^53 a rounded odd integer is indistinguishable from its neighbour.
	// Refusing the boundary as well avoids claiming that provenance was retained.
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v >= 1<<53 || math.Trunc(v) != v {
		return 0, fmt.Errorf("cuptibench: counter %g violates profile conversion policy (%s)", v, integerPolicy)
	}
	return int64(v), nil
}

func profileRanges(r *Result) ([]Range, exportCoverage, error) {
	cov := exportCoverage{Capture: r.Metadata["capture_id"]}
	parent := make(map[string]bool)
	for _, v := range r.Ranges {
		parent[v.ParentID] = true
	}
	var selected []Range
	for _, v := range r.Ranges {
		keep := false
		switch r.Metadata["mode"] {
		case "whole":
			keep = v.Kind == "operation" && v.ParentID == ""
		case "regions":
			keep = v.Kind == "region" && !parent[v.ID]
		case "kernels":
			keep = v.Kind == "kernel"
		default:
			return nil, cov, fmt.Errorf("cuptibench: unknown profile mode %q", r.Metadata["mode"])
		}
		if keep {
			selected = append(selected, v)
			cov.Exported = append(cov.Exported, v.ID)
		} else {
			cov.Excluded = append(cov.Excluded, v.ID)
		}
	}
	if r.Metadata["mode"] == "regions" {
		cov.Note = "terminal regions only; inclusive ancestors excluded; work outside terminal regions is not attributed"
	}
	return selected, cov, nil
}

func buildExport(path string, results []*Result) (map[string][]byte, error) {
	if len(results) == 0 {
		return nil, fmt.Errorf("cuptibench: profile requested but no benchmark results recorded")
	}
	manifest := exportManifest{Version: 1, Conversion: integerPolicy, Results: results}
	var partitions []*exportPartition
	bySchema := make(map[string]*exportPartition)
	seenCapture := make(map[string]bool)
	for _, r := range results {
		if err := validateExportResult(r); err != nil {
			return nil, err
		}
		capture := r.Metadata["capture_id"]
		if seenCapture[capture] {
			return nil, fmt.Errorf("cuptibench: duplicate capture identity %q", capture)
		}
		seenCapture[capture] = true
		if a := r.Aggregate; a != nil {
			id := a.Metadata["capture_id"]
			if seenCapture[id] {
				return nil, fmt.Errorf("cuptibench: duplicate aggregate capture identity %q", id)
			}
			seenCapture[id] = true
			cov := exportCoverage{Capture: id, Note: "associated whole-operation capture retained in JSON only; not additive to kernel samples"}
			for _, v := range a.Ranges {
				cov.Excluded = append(cov.Excluded, v.ID)
			}
			manifest.Coverage = append(manifest.Coverage, cov)
		}
		byID := make(map[string]Range)
		for _, v := range r.Ranges {
			byID[v.ID] = v
		}
		ranges, cov, err := profileRanges(r)
		if err != nil {
			return nil, err
		}
		for _, v := range ranges {
			var schema []cudaprof.SampleType
			var values []int64
			for _, metric := range v.Metrics {
				if !metric.Additive || metric.Unit == "" || metric.Unit == "unknown" {
					continue
				}
				value, err := integerCounter(metric.Value)
				if err != nil {
					return nil, fmt.Errorf("metric %s range %s: %w", metric.Name, v.ID, err)
				}
				schema = append(schema, cudaprof.SampleType{Name: metric.Name, Unit: metric.Unit})
				values = append(values, value)
			}
			if len(schema) == 0 {
				cov.Exported = slices.DeleteFunc(cov.Exported, func(id string) bool { return id == v.ID })
				cov.Excluded = append(cov.Excluded, v.ID)
				cov.Note += "; ranges without verified additive metrics remain in JSON only"
				continue
			}
			key, _ := json.Marshal(schema)
			part := bySchema[string(key)]
			if part == nil {
				part = &exportPartition{builder: cudaprof.NewBuilder(), schema: schema}
				if err := part.builder.SetDeclaredSampleTypes(schema...); err != nil {
					return nil, err
				}
				if err := part.builder.SetDefaultSampleType(schema[0].Name); err != nil {
					return nil, err
				}
				part.builder.AddComment("cuptibench: separate captures and views are experiments, not additive pieces of one decode pass")
				part.builder.AddComment(integerPolicy)
				bySchema[string(key)] = part
				partitions = append(partitions, part)
			}
			if !slices.Contains(part.captures, capture) {
				part.captures = append(part.captures, capture)
				if r.Metadata["mode"] == "regions" {
					part.builder.AddComment("cuptibench: capture " + capture + ": terminal-region projection may omit work outside leaves; its total is not the whole-operation total; inclusive root retained in JSON")
				}
			}
			labels := maps.Clone(r.Metadata)
			labels["profile-operations"] = strconv.Itoa(r.Operations)
			labels["range-id"] = v.ID
			labels["range-kind"] = v.Kind
			labels["range-name"] = v.Name
			if v.NativeName != "" {
				labels["native-range-name"] = v.NativeName
			}
			labels["profile-conversion"] = integerPolicy
			var frames []cudaprof.Frame
			var regionPath []string
			for at := v; at.ID != ""; at = byID[at.ParentID] {
				frames = append(frames, cudaprof.Frame{FunctionName: at.Name})
				if at.Kind == "region" {
					regionPath = append(regionPath, at.Name)
				}
			}
			slices.Reverse(regionPath)
			if len(regionPath) != 0 {
				labels["region_path"] = strings.Join(regionPath, "/")
			}
			// The benchmark is a distinct host frame and label, never a kernel alias.
			if v.Kind == "kernel" {
				labels["kernel"] = v.Name
			}
			frames = append(frames, cudaprof.Frame{FunctionName: r.Metadata["benchmark"]})
			if err := part.builder.AddDeclaredSample(values, frames, labels); err != nil {
				return nil, err
			}
		}
		manifest.Coverage = append(manifest.Coverage, cov)
	}
	if len(partitions) == 0 {
		return nil, fmt.Errorf("cuptibench: no verified additive metrics available for pprof export")
	}
	files := make(map[string][]byte)
	for i, part := range partitions {
		name := path
		if i > 0 {
			name = fmt.Sprintf("%s.part%d", path, i+1)
		}
		var buf bytes.Buffer
		if err := part.builder.WriteGzip(&buf); err != nil {
			return nil, err
		}
		files[name] = buf.Bytes()
		manifest.Profiles = append(manifest.Profiles, exportProfile{Path: name, SampleTypes: part.schema, Captures: part.captures})
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, err
	}
	files[path+".json"] = append(data, '\n')
	return files, nil
}

func packagePath() (string, error) {
	info, ok := debug.ReadBuildInfo()
	if !ok || info.Path == "" || info.Path == "command-line-arguments" {
		return "", fmt.Errorf("cuptibench: test package identity unavailable in build information")
	}
	return strings.TrimSuffix(info.Path, ".test"), nil
}

func expandProfilePath(path, pkg string) (string, error) {
	if !strings.Contains(path, "{package}") {
		return path, nil
	}
	if pkg == "" {
		return "", fmt.Errorf("cuptibench: {package} requires verified test package identity")
	}
	// Full UTF-8 hex encoding is reversible and collision-free, unlike replacing
	// path separators with underscores or truncating a package name.
	return strings.ReplaceAll(path, "{package}", hex.EncodeToString([]byte(pkg))), nil
}

func writeExport(path string, results []*Result) error {
	if strings.Contains(path, "{package}") {
		pkg, err := packagePath()
		if err != nil {
			return err
		}
		path, err = expandProfilePath(path, pkg)
		if err != nil {
			return err
		}
	}
	files, err := buildExport(path, results)
	if err != nil {
		return err
	}
	return publishExport(files, path+".json")
}

// publishExport stages every file before installing any destination. Link is an
// atomic no-replace publication of a sibling temporary file, unlike Rename on
// Unix, which can overwrite a racing writer. Publish the manifest last: its
// presence is the completion marker for this collection of profile files.
func publishExport(files map[string][]byte, manifest string) (err error) {
	paths := slices.Sorted(maps.Keys(files))
	paths = slices.DeleteFunc(paths, func(p string) bool { return p == manifest })
	paths = append(paths, manifest)
	var staged, published []string
	defer func() {
		for _, p := range staged {
			_ = os.Remove(p)
		}
		if err != nil {
			for _, p := range published {
				_ = os.Remove(p)
			}
		}
	}()
	temps := make(map[string]string)
	for _, p := range paths {
		if _, e := os.Lstat(p); e == nil {
			return fmt.Errorf("cuptibench: output already exists: %s", p)
		} else if !os.IsNotExist(e) {
			return e
		}
		// A short prefix also supports long destination basenames.
		sum := sha256.Sum256([]byte(filepath.Base(p)))
		f, e := os.CreateTemp(filepath.Dir(p), fmt.Sprintf(".cuptibench-%x-", sum[:4]))
		if e != nil {
			return e
		}
		staged = append(staged, f.Name())
		temps[p] = f.Name()
		_, e = f.Write(files[p])
		if e == nil {
			e = f.Sync()
		}
		closeErr := f.Close()
		if e != nil {
			return e
		}
		if closeErr != nil {
			return closeErr
		}
	}
	for _, p := range paths {
		if e := os.Link(temps[p], p); e != nil {
			return fmt.Errorf("cuptibench: publish %s: %w", p, e)
		}
		published = append(published, p)
	}
	return nil
}
