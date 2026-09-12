package cudacapture

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"

	"github.com/google/pprof/profile"
)

// RankCapture supplies an original recording and its explicitly declared
// experiment identity. Session and Run identify the shared experiment, not the
// independent launcher session stored in each recording. The caller must verify
// that the recordings belong to that experiment; timestamps and names cannot
// establish this relationship. Recording is preserved byte for byte.
type RankCapture struct {
	Session   string
	Run       string
	Rank      uint32
	Recording []byte
}

// MergeRanks combines additive profiles from ranks in one explicitly identified
// session and run. It preserves original recordings, files, metadata, typed
// observations and code roles under rank namespaces. Samples retain their
// device and stream labels and gain capture.session, capture.run and
// capture.rank labels. Existing conflicting labels are rejected.
//
// The result is an aggregate, not a synchronized timeline. Its timestamps and
// duration are unset. Typed values are never added, and replay runs must be
// supplied separately. The caller must ensure the sample types are additive.
func MergeRanks(inputs []RankCapture) (*Capture, error) {
	if len(inputs) == 0 {
		return nil, errors.New("cudacapture: no ranks to merge")
	}
	inputs = slices.Clone(inputs)
	slices.SortFunc(inputs, func(a, b RankCapture) int {
		if a.Rank < b.Rank {
			return -1
		}
		if a.Rank > b.Rank {
			return 1
		}
		return 0
	})
	session, run := inputs[0].Session, inputs[0].Run
	if session == "" || run == "" {
		return nil, errors.New("cudacapture: merge requires session and run identities")
	}
	out := &Capture{
		Profile: &profile.Profile{}, Files: make(map[string][]byte),
		CodeUnavailable: map[string]string{},
	}
	for _, role := range []string{"source", "ptx", "cubin", "sass"} {
		out.CodeUnavailable[role] = "rank-specific availability is retained in original recordings and modules"
	}
	type rankManifest struct {
		Rank        uint32            `json:"rank"`
		Recording   string            `json:"recording"`
		SHA256      string            `json:"sha256"`
		Metadata    json.RawMessage   `json:"metadata,omitempty"`
		Unavailable map[string]string `json:"code_unavailable,omitempty"`
	}
	manifest := struct {
		Kind    string         `json:"kind"`
		Session string         `json:"session"`
		Run     string         `json:"run"`
		Ranks   []rankManifest `json:"ranks"`
	}{Kind: "same-run-rank-aggregate", Session: session, Run: run}
	var total int64
	var totals []int64
	addFile := func(name string, data []byte) error {
		if _, exists := out.Files[name]; exists {
			return fmt.Errorf("cudacapture: duplicate aggregate file %q", name)
		}
		if len(data) > MaxCaptureFileBytes || total+int64(len(data)) > MaxCaptureTotalBytes || len(out.Files) >= MaxCaptureFiles {
			return errors.New("cudacapture: aggregate exceeds file limits")
		}
		out.Files[name] = bytes.Clone(data)
		total += int64(len(data))
		return nil
	}
	for i, input := range inputs {
		if input.Session != session || input.Run != run {
			return nil, errors.New("cudacapture: cannot merge different sessions or runs")
		}
		if i > 0 && input.Rank == inputs[i-1].Rank {
			return nil, fmt.Errorf("cudacapture: duplicate rank %d", input.Rank)
		}
		if len(input.Recording) > MaxCaptureFileBytes {
			return nil, errors.New("cudacapture: original recording exceeds attachment limit")
		}
		c, err := Read(bytes.NewReader(input.Recording))
		if err != nil {
			return nil, fmt.Errorf("cudacapture: rank %d: %w", input.Rank, err)
		}
		prefix := "rank" + strconv.FormatUint(uint64(input.Rank), 10) + "-"
		recording := prefix + "original.pprof"
		if err := addFile(recording, input.Recording); err != nil {
			return nil, err
		}
		// The file prefix is distinct from the original-recording name even if
		// a source capture already contains an attachment called original.pprof.
		fileName := func(name string) string {
			if name == "" {
				return ""
			}
			return prefix + "file-" + name
		}
		for name, data := range c.Files {
			if err := addFile(fileName(name), data); err != nil {
				return nil, err
			}
		}
		code := func(v CodeFiles) CodeFiles {
			return CodeFiles{Source: fileName(v.Source), PTX: fileName(v.PTX), CUBIN: fileName(v.CUBIN), SASS: fileName(v.SASS)}
		}
		if c.Code != (CodeFiles{}) {
			out.Modules = append(out.Modules, Module{ID: prefix + "primary", Code: code(c.Code)})
		}
		for _, module := range c.Modules {
			module.ID = prefix + "module-" + module.ID
			module.Code = code(module.Code)
			out.Modules = append(out.Modules, module)
		}
		for _, descriptor := range c.Descriptors {
			descriptor.ID = prefix + descriptor.ID
			out.Descriptors = append(out.Descriptors, descriptor)
		}
		for _, observation := range c.Observations {
			if observation.RunID != "" && observation.RunID != run {
				return nil, fmt.Errorf("cudacapture: rank %d observation belongs to another run", input.Rank)
			}
			observation.DescriptorID = prefix + observation.DescriptorID
			if observation.EntityID != "" {
				observation.EntityID = prefix + observation.EntityID
			}
			observation.Value = bytes.Clone(observation.Value)
			out.Observations = append(out.Observations, observation)
		}
		p := c.Profile.Copy()
		if i == 0 {
			out.Profile.SampleType = p.SampleType
			out.Profile.PeriodType = p.PeriodType
			out.Profile.Period = p.Period
			out.Profile.DefaultSampleType = p.DefaultSampleType
			totals = make([]int64, len(p.SampleType))
		} else if !sameProfileTypes(out.Profile, p) {
			return nil, errors.New("cudacapture: incompatible rank profile types")
		}
		for _, sample := range p.Sample {
			for j, value := range sample.Value {
				if value > 0 && totals[j] > math.MaxInt64-value || value < 0 && totals[j] < math.MinInt64-value {
					return nil, errors.New("cudacapture: aggregate sample total overflows int64")
				}
				totals[j] += value
			}
			if sample.Label == nil {
				sample.Label = make(map[string][]string)
			}
			for key, value := range map[string]string{"capture.session": session, "capture.run": run, "capture.rank": strconv.FormatUint(uint64(input.Rank), 10)} {
				if old, ok := sample.Label[key]; ok && (len(old) != 1 || old[0] != value) {
					return nil, fmt.Errorf("cudacapture: conflicting %s label", key)
				}
				if _, ok := sample.NumLabel[key]; ok {
					return nil, fmt.Errorf("cudacapture: conflicting numeric %s label", key)
				}
				sample.Label[key] = []string{value}
			}
		}
		// Copy preserves pointer relationships. Assign fresh IDs without merging
		// same-name code from different modules, ranks or address spaces.
		for _, m := range p.Mapping {
			m.ID = uint64(len(out.Profile.Mapping) + 1)
			out.Profile.Mapping = append(out.Profile.Mapping, m)
		}
		for _, f := range p.Function {
			f.ID = uint64(len(out.Profile.Function) + 1)
			out.Profile.Function = append(out.Profile.Function, f)
		}
		for _, l := range p.Location {
			l.ID = uint64(len(out.Profile.Location) + 1)
			out.Profile.Location = append(out.Profile.Location, l)
		}
		out.Profile.Sample = append(out.Profile.Sample, p.Sample...)
		manifest.Ranks = append(manifest.Ranks, rankManifest{Rank: input.Rank, Recording: recording, SHA256: fmt.Sprintf("%x", sha256.Sum256(input.Recording)), Metadata: bytes.Clone(c.Metadata), Unavailable: c.CodeUnavailable})
	}
	var err error
	out.Metadata, err = json.Marshal(manifest)
	if err != nil {
		return nil, err
	}
	if err := validateCapture(out, Limits{}.withDefaults()); err != nil {
		return nil, err
	}
	return out, nil
}

func sameProfileTypes(a, b *profile.Profile) bool {
	if len(a.SampleType) != len(b.SampleType) || a.Period != b.Period || a.DefaultSampleType != b.DefaultSampleType {
		return false
	}
	for i := range a.SampleType {
		if *a.SampleType[i] != *b.SampleType[i] {
			return false
		}
	}
	if a.PeriodType == nil || b.PeriodType == nil {
		return a.PeriodType == b.PeriodType
	}
	return *a.PeriodType == *b.PeriodType
}
