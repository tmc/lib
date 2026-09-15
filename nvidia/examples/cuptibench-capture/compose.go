package cuptibenchcapture

import (
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"regexp"
	"slices"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cuptibench"
)

// Export is what one cuptibench -cuptiprofile destination produced: the pprof
// profile and the JSON results file written beside it. Name is the profile's
// path as cuptibench recorded it, which pairs the two files.
type Export struct {
	Name    string
	Profile []byte
	Results []byte
}

// Artifacts are the code artifacts of the module the benchmark loaded. Every
// role is required: a capture missing one of them is not offline readable.
type Artifacts struct {
	// Prefix names the files inside the capture, as Prefix+".cu" and so on.
	Prefix string
	Source []byte
	PTX    []byte
	CUBIN  []byte
	SASS   []byte
	// ImageSHA256 is the hex digest of the image handed to the driver, taken
	// from the same bytes at load time. It identifies the module generation;
	// the kernel names in the profile do not.
	ImageSHA256 string
	// Name is the driver-visible module name, if the loader has one.
	Name string
	// Generation distinguishes repeated loads of the same source.
	Generation uint64
	// Kernels are kernel names the benchmark executed. Kernel names carried by
	// the profile are checked in addition to these.
	Kernels []string
}

// benchResults is the subset of cuptibench's exported JSON this package reads.
// The field names are cuptibench's, which writes them without JSON tags.
type benchResults struct {
	Version  int
	Profiles []struct {
		Path     string
		Captures []string
	}
	Results []*cuptibench.Result
}

var prefixPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

// Compose packages e and a as a capture. It fails rather than publishing a
// capture whose counters, code and provenance do not agree: a lossy or failed
// collection, a profile paired with another run's results, artifacts whose
// digest is not the loaded image, or a kernel the retained code does not define.
func Compose(e Export, a Artifacts) (*cudacapture.Capture, error) {
	if !prefixPattern.MatchString(a.Prefix) {
		return nil, fmt.Errorf("cuptibenchcapture: artifact prefix %q is not a plain name", a.Prefix)
	}
	for _, role := range []struct {
		name string
		data []byte
	}{{"source", a.Source}, {"ptx", a.PTX}, {"cubin", a.CUBIN}, {"sass", a.SASS}} {
		if len(role.data) == 0 {
			return nil, fmt.Errorf("cuptibenchcapture: missing %s artifact", role.name)
		}
	}
	if a.ImageSHA256 == "" {
		return nil, errors.New("cuptibenchcapture: missing loaded image digest")
	}
	prof, err := profile.ParseData(e.Profile)
	if err != nil {
		return nil, fmt.Errorf("cuptibenchcapture: parse counter profile: %w", err)
	}
	var results benchResults
	if err := json.Unmarshal(e.Results, &results); err != nil {
		return nil, fmt.Errorf("cuptibenchcapture: parse cuptibench results: %w", err)
	}
	// An unrecognized export schema is not repackaged; its fields may mean
	// something this composition does not check.
	if results.Version != 1 {
		return nil, fmt.Errorf("cuptibenchcapture: unsupported cuptibench export version %d", results.Version)
	}
	captures, err := pairProfile(results, e.Name)
	if err != nil {
		return nil, err
	}
	for _, r := range results.Results {
		if err := checkComplete(r); err != nil {
			return nil, err
		}
	}
	if err := checkProfileCaptures(prof, captures); err != nil {
		return nil, err
	}

	files := map[string][]byte{
		a.Prefix + ".cu":              a.Source,
		a.Prefix + ".ptx":             a.PTX,
		a.Prefix + ".cubin":           a.CUBIN,
		a.Prefix + ".sass":            a.SASS,
		a.Prefix + ".cuptibench.json": e.Results,
	}
	code := cudacapture.CodeFiles{
		Source: a.Prefix + ".cu",
		PTX:    a.Prefix + ".ptx",
		CUBIN:  a.Prefix + ".cubin",
		SASS:   a.Prefix + ".sass",
	}
	moduleID := "sha256:" + a.ImageSHA256
	c := &cudacapture.Capture{
		Profile: prof,
		Files:   files,
		Code:    code,
		Modules: []cudacapture.Module{{
			ID:         moduleID,
			Generation: a.Generation,
			Name:       a.Name,
			Code:       code,
		}},
	}
	kernels := append(slices.Clone(a.Kernels), profileKernels(prof)...)
	slices.Sort(kernels)
	kernels = slices.Compact(kernels)
	if _, err := cudacapture.VerifyModule(c, moduleID, a.ImageSHA256, kernels); err != nil {
		return nil, fmt.Errorf("cuptibenchcapture: %w", err)
	}

	c.Descriptors, c.Observations, err = observe(results.Results)
	if err != nil {
		return nil, err
	}
	c.Metadata, err = describe(e, a, results, kernels)
	if err != nil {
		return nil, err
	}
	return c, nil
}

// pairProfile reports the capture identities the named profile carries. A
// cuptibench export can hold several profiles with different sample schemas,
// so composing the wrong one with these results would attribute counters to
// captures the profile does not contain.
func pairProfile(results benchResults, name string) ([]string, error) {
	if name == "" {
		return nil, errors.New("cuptibenchcapture: export has no profile name")
	}
	var found []string
	for _, p := range results.Profiles {
		if path.Base(p.Path) == path.Base(name) {
			if found != nil {
				return nil, fmt.Errorf("cuptibenchcapture: cuptibench results name %q more than once", name)
			}
			found = p.Captures
		}
	}
	if found == nil {
		return nil, fmt.Errorf("cuptibenchcapture: cuptibench results do not name profile %q", name)
	}
	return found, nil
}

// checkComplete refuses a result cuptibench itself would not have exported.
// It repeats the quality gate at x/cuptibench/export.go so a failed or partial
// collection cannot reach a capture through hand-assembled files.
func checkComplete(r *cuptibench.Result) error {
	if r == nil || r.Operations <= 0 {
		return errors.New("cuptibenchcapture: result records no operations")
	}
	q := r.Quality
	if !q.AllPassesSubmitted || !q.Decoded || q.DroppedRanges != 0 || q.PassesStarted == 0 || q.PassesStarted != q.PassesStopped {
		return fmt.Errorf("cuptibenchcapture: capture %q is incomplete or lossy: %d/%d passes, decoded=%v, submitted=%v, dropped=%d",
			r.Metadata["capture_id"], q.PassesStopped, q.PassesStarted, q.Decoded, q.AllPassesSubmitted, q.DroppedRanges)
	}
	for _, key := range []string{"capture_id", "benchmark", "run_id", "mode"} {
		if r.Metadata[key] == "" {
			return fmt.Errorf("cuptibenchcapture: result is missing provenance %q", key)
		}
	}
	if r.Aggregate != nil {
		if err := checkComplete(r.Aggregate); err != nil {
			return err
		}
	}
	return nil
}

// checkProfileCaptures requires every sample to name a capture the results
// describe, so the packaged profile and the packaged provenance are one run.
func checkProfileCaptures(p *profile.Profile, captures []string) error {
	for _, s := range p.Sample {
		values := s.Label["capture_id"]
		if len(values) == 0 {
			return errors.New("cuptibenchcapture: profile sample has no capture identity")
		}
		for _, v := range values {
			if !slices.Contains(captures, v) {
				return fmt.Errorf("cuptibenchcapture: profile sample names capture %q, which the results do not list for this profile", v)
			}
		}
	}
	return nil
}

// profileKernels returns the kernel names the profile attributes samples to.
func profileKernels(p *profile.Profile) []string {
	var names []string
	for _, s := range p.Sample {
		names = append(names, s.Label["kernel"]...)
	}
	return names
}
