package advise

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// WithShareEvidence returns a copy with classified profile-share evidence.
// known must reflect the producer's presence information: the legacy numeric
// field alone cannot distinguish a missing value from a measured zero.
// basis must identify the actual numerator, denominator and time channel.
// An observed profile share is Derived from the stated denominator; suspected
// bottleneck attribution remains Inferred. No result is a measured cause.
func (item Item) WithShareEvidence(source cudaevidence.Source, known bool, basis string) (Item, error) {
	var record cudaevidence.Record
	record.Source = source
	record.Unit = "percent"
	if !known || item.EvidenceLevel == EvidenceUnmeasured {
		record.Reason = "profile share unavailable"
	} else {
		if strings.TrimSpace(basis) == "" {
			return Item{}, fmt.Errorf("classify profile share: missing denominator basis")
		}
		if err := source.Identity.Validate(); err != nil {
			return Item{}, fmt.Errorf("classify profile share: %w", err)
		}
		if item.Location.Kernel == "" || item.Location.Kernel != source.Identity.Entry {
			return Item{}, fmt.Errorf("classify profile share: kernel identity mismatch")
		}
		if math.IsNaN(item.ShareOfProfile) || math.IsInf(item.ShareOfProfile, 0) || item.ShareOfProfile < 0 || item.ShareOfProfile > 100 {
			return Item{}, fmt.Errorf("classify profile share: invalid percentage")
		}
		switch item.EvidenceLevel {
		case EvidenceObserved:
			record.Class = cudaevidence.Derived
			record.Basis = basis
		case EvidenceSuspected:
			record.Class = cudaevidence.Inferred
			record.Basis = basis + "; profile attribution interpreted as a possible bottleneck"
		default:
			return Item{}, fmt.Errorf("classify profile share: unknown evidence level")
		}
		number := item.ShareOfProfile
		record.Number = &number
	}
	value, err := cudaevidence.New(record)
	if err != nil {
		return Item{}, fmt.Errorf("classify profile share: %w", err)
	}
	item.ClassifiedShare = &value
	return item, nil
}

// classifyProfileShares runs after ranking/grouping. It independently checks
// the population; a legacy scalar, including zero, is never presence evidence.
func classifyProfileShares(cap *cudacapture.Capture, report *Report, supplied *cudaevidence.Source) {
	counts, total, channel, reason := profileSharePopulation(cap)
	source := cudaevidence.Source{Producer: "retained-profile", Locator: "gpu.pprof"}
	if supplied != nil {
		source = *supplied
		if reason == "" {
			reason = profileShareIdentity(cap, source)
		}
	} else if reason == "" {
		reason = "artifact, compiler and workload source identity unavailable"
	}
	var classify func([]Item)
	classify = func(items []Item) {
		for i := range items {
			item := &items[i]
			why := reason
			n, present := counts[item.Location.Kernel]
			if why == "" && (item.Location.Kernel != source.Identity.Entry || item.Location.Module != "") {
				why = "finding has no unique supplied kernel identity"
			}
			if why == "" && (!present || item.EvidenceLevel == EvidenceUnmeasured) {
				why = "kernel share population unavailable"
			}
			share := 0.0
			if total > 0 {
				share = float64(n) / float64(total) * 100
			}
			if why == "" && item.ShareOfProfile != 0 && math.Abs(share-item.ShareOfProfile) > 1e-9 {
				why = "finding uses a different numerator or denominator"
			}
			if why == "" {
				basis := fmt.Sprintf("retained %s nanoseconds for exact kernel entry / all retained kernel nanoseconds: %d/%d * 100; journal/profile per-kernel duration totals agree; caller-supplied compiler association, not hardware authentication", channel, n, total)
				candidate := *item
				candidate.ShareOfProfile = share
				typed, err := candidate.WithShareEvidence(source, true, basis)
				if err == nil {
					item.ClassifiedShare = typed.ClassifiedShare
				} else {
					why = err.Error()
				}
			}
			if why != "" {
				value, _ := cudaevidence.New(cudaevidence.Record{Unit: "percent", Source: source, Reason: why})
				item.ClassifiedShare = &value
			}
			classify(item.Patterns)
		}
	}
	classify(report.Items)
}

func profileSharePopulation(cap *cudacapture.Capture) (map[string]int64, int64, string, string) {
	if cap.Profile == nil {
		return nil, 0, "", "profile unavailable"
	}
	p := cap.Profile
	index, channel := -1, ""
	for i, t := range p.SampleType {
		if t != nil && (t.Type == "kernel_time" || t.Type == "gpu_time") {
			if index >= 0 {
				return nil, 0, "", "ambiguous kernel-time channels"
			}
			if t.Unit != "nanoseconds" && t.Unit != "ns" {
				return nil, 0, "", "kernel-time unit is not nanoseconds"
			}
			index, channel = i, t.Type
		}
	}
	if index < 0 {
		return nil, 0, "", "kernel-time population unavailable; sample counts are not device time"
	}
	counts := make(map[string]int64)
	var total int64
	for _, s := range p.Sample {
		if s == nil || index >= len(s.Value) || s.Value[index] < 0 {
			return nil, 0, channel, "invalid kernel-time sample"
		}
		n := s.Value[index]
		name := extractKernelNameFromSample(s)
		if len(s.Label["kernel"]) > 1 || len(s.Label["name"]) > 1 ||
			len(s.Label["kernel"]) == 1 && len(s.Label["name"]) == 1 && s.Label["kernel"][0] != s.Label["name"][0] {
			return nil, 0, channel, "ambiguous profile kernel entry"
		}
		if subtype := s.Label["subtype"]; len(subtype) > 1 || len(subtype) == 1 && subtype[0] != "kernel" {
			if n != 0 {
				return nil, 0, channel, "nonkernel sample carries kernel time"
			}
			continue
		}
		if name == "" {
			return nil, 0, channel, "unnamed kernel-time sample"
		}
		if n > math.MaxInt64-total {
			return nil, 0, channel, "kernel-time population overflows"
		}
		total += n
		counts[name] += n
	}
	if total <= 0 {
		return counts, total, channel, "kernel-time denominator is zero"
	}
	status, err := profileShareStatus(cap)
	if err != nil {
		return counts, total, channel, err.Error()
	}
	var success bool
	var complete string
	var dropped, unqualified uint64
	if json.Unmarshal(status["capture_success"], &success) != nil || !success ||
		json.Unmarshal(status["run_completeness"], &complete) != nil || complete != "complete" ||
		bytes.Equal(bytes.TrimSpace(status["dropped_records"]), []byte("null")) || json.Unmarshal(status["dropped_records"], &dropped) != nil || dropped != 0 {
		return counts, total, channel, "capture completeness or loss coverage unavailable"
	}
	if raw, ok := status["unqualified_generations"]; ok && (bytes.Equal(bytes.TrimSpace(raw), []byte("null")) || json.Unmarshal(raw, &unqualified) != nil || unqualified != 0) {
		return counts, total, channel, "capture has unqualified or unknown generations"
	}
	journals, err := timeline.Journals(cap)
	if err != nil {
		return counts, total, channel, "journal population unavailable: " + err.Error()
	}
	records, err := timeline.Decode(journals, timeline.MaxRecords, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return counts, total, channel, "journal population unavailable: " + err.Error()
	}
	actual := make(map[string]int64)
	for _, a := range records.Activities {
		if a.Subtype != "kernel" {
			continue
		}
		if a.Name == "" || a.Start < 0 || a.End < a.Start || a.End-a.Start > math.MaxInt64-actual[a.Name] {
			return counts, total, channel, "invalid journal kernel population"
		}
		actual[a.Name] += a.End - a.Start
	}
	if len(actual) != len(counts) {
		return counts, total, channel, "journal/profile kernel populations differ"
	}
	for name, n := range counts {
		if v, ok := actual[name]; !ok || v != n {
			return counts, total, channel, "journal/profile kernel populations differ"
		}
	}
	return counts, total, channel, ""
}

func profileShareIdentity(cap *cudacapture.Capture, source cudaevidence.Source) string {
	id := source.Identity
	if cap.Identity != nil {
		if cap.Identity.Version != "v1" || !cap.Identity.Complete {
			return "capture identity is unsupported or incomplete"
		}
		if cap.Identity.WorkloadHash != "" && cap.Identity.WorkloadHash != id.WorkloadKey {
			return "capture workload contradicts supplied identity"
		}
	}
	// A retained code image can veto the supplied image association. Absent code
	// cannot authenticate it, and that limitation remains in the numeric basis.
	var images [][]byte
	if name := cap.Code.CUBIN; name != "" {
		images = append(images, cap.Files[name])
	}
	if len(cap.Modules) != 0 {
		for _, module := range cap.Modules {
			if name := module.Code.CUBIN; name != "" {
				images = append(images, cap.Files[name])
			}
		}
	}
	if len(images) != 0 {
		if len(images) != 1 || len(images[0]) == 0 || fmt.Sprintf("%x", sha256.Sum256(images[0])) != id.ArtifactDigest {
			return "retained image contradicts or cannot uniquely bind supplied identity"
		}
	}
	if reason := profileShareTargets(cap, id.Target); reason != "" {
		return reason
	}
	// Reuse the existing strict manifest reader when the capture retained one.
	status, err := profileShareStatus(cap)
	if err != nil {
		return err.Error()
	}
	if raw, ok := status["capture_plan_manifest"]; ok {
		manifest, err := cudaprof.ReadCaptureManifest(bytes.NewReader(raw), 64<<20)
		if err != nil || manifest.Plan.Identity != id {
			return "capture plan identity contradicts supplied source"
		}
		if !manifest.Complete() {
			return "capture plan provider evidence incomplete"
		}
	}

	return ""
}

// profileShareTargets checks every retained source of target identity, rather
// than the first architecture useful to the legacy resource estimator. A matching
// listing cannot erase a conflicting metadata, module or provenance target.
func profileShareTargets(cap *cudacapture.Capture, supplied string) string {
	// ParseArch provides sm_/compute_ aliases and preserves feature suffixes.
	// Match cudaptx's existing all-letter target suffix rule; ParseArch alone
	// accepts any nondigit tail, including punctuation that is not an SM target.
	parse := func(target string) (int, int, string, error) {
		canonical := cudapipeline.CanonicalArch(target)
		rest, ok := strings.CutPrefix(canonical, "sm_")
		if !ok {
			rest, ok = strings.CutPrefix(canonical, "compute_")
		}
		// cudaptx targetArchitecture requires digits immediately after the prefix.
		// In particular, do not let ParseArch's successive TrimPrefix calls accept
		// a nested spelling such as sm_compute_70 as a new architecture alias.
		if !ok || rest == "" || rest[0] < '0' || rest[0] > '9' {
			return 0, 0, "", fmt.Errorf("invalid target architecture")
		}
		m, n, suffix, err := cudakernel.ParseArch(canonical)
		if err != nil {
			return 0, 0, "", err
		}
		for _, c := range suffix {
			if c < 'a' || c > 'z' {
				return 0, 0, "", fmt.Errorf("invalid target suffix")
			}
		}
		return m, n, suffix, nil
	}
	major, minor, suffix, err := parse(supplied)
	if err != nil {
		return "supplied target identity unavailable"
	}
	check := func(target string, baseOnly bool) string {
		m, n, s, err := parse(target)
		if err != nil {
			return "retained target identity unavailable"
		}
		if m != major || n != minor || (!baseOnly && s != suffix) {
			return "retained target contradicts supplied identity"
		}
		return ""
	}
	if cap.Identity != nil && cap.Identity.ComputeCap != "" {
		if reason := check(cap.Identity.ComputeCap, false); reason != "" {
			return reason
		}
	}
	// These are the same architecture keys used by findArchInMap, plus the
	// serialized ComparisonIdentity compute_cap field. Visit all values and array
	// elements; JSON/map ordering must never select which fact is authoritative.
	var visit func(any) string
	visit = func(value any) string {
		switch value := value.(type) {
		case map[string]any:
			keys := make([]string, 0, len(value))
			for key := range value {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				switch strings.ToLower(key) {
				case "arch", "architecture", "target", "compute_capability", "computecapability", "compute_cap", "chip", "chip_name", "sm":
					if target, ok := value[key].(string); ok && target != "" {
						if reason := check(target, false); reason != "" {
							return reason
						}
					} else if value[key] != nil && value[key] != "" {
						return "retained target identity unavailable"
					}
				}
				if reason := visit(value[key]); reason != "" {
					return reason
				}
			}
		case []any:
			for _, child := range value {
				if reason := visit(child); reason != "" {
					return reason
				}
			}
		}
		return ""
	}
	readJSON := func(data []byte) string {
		if len(data) == 0 || len(data) > 64<<20 {
			return "retained target metadata unavailable or oversized"
		}
		d := json.NewDecoder(bytes.NewReader(data))
		if err := profileShareJSONKeys(d, 0); err != nil {
			return "retained target metadata unavailable: " + err.Error()
		}
		if _, err := d.Token(); err != io.EOF {
			return "retained target metadata has trailing JSON"
		}
		d = json.NewDecoder(bytes.NewReader(data))
		d.UseNumber()
		var value any
		if err := d.Decode(&value); err != nil {
			return "retained target metadata unavailable"
		}
		return visit(value)
	}
	if len(cap.Metadata) != 0 {
		if reason := readJSON(cap.Metadata); reason != "" {
			return reason
		}
	}
	for _, name := range []string{"capture-provenance.json", "capture-status.json", "status.json", "manifest.json", "provenance.json"} {
		if data, ok := cap.Files[name]; ok {
			if reason := readJSON(data); reason != "" {
				return reason
			}
		}
	}
	// Include every mapped or conventionally named listing, including modules;
	// capturefiles.PTX/SASS intentionally select only one for the legacy analyzer.
	roles := make(map[string]map[string]bool)
	add := func(name, role string) {
		if name != "" {
			if roles[name] == nil {
				roles[name] = make(map[string]bool)
			}
			roles[name][role] = true
		}
	}
	code := func(c cudacapture.CodeFiles) { add(c.PTX, "ptx"); add(c.SASS, "sass"); add(c.CUBIN, "cubin") }
	code(cap.Code)
	for _, module := range cap.Modules {
		code(module.Code)
	}
	for name := range cap.Files {
		for _, role := range []string{"ptx", "sass", "cubin"} {
			if strings.HasSuffix(name, "."+role) {
				add(name, role)
			}
		}
	}
	names := make([]string, 0, len(roles))
	for name := range roles {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		data := cap.Files[name]
		if len(data) == 0 {
			return "retained target listing unavailable"
		}
		if roles[name]["ptx"] {
			mod, err := cudaptx.ParseBytes(data)
			if err != nil || mod == nil {
				return "retained PTX target unavailable"
			}
			// Require one actual declaration; the legacy Target field selects the
			// last declaration and cannot establish uniqueness.
			if len(mod.TargetDeclarations) != 1 {
				return "retained PTX target declarations ambiguous or unavailable"
			}
			target, _, _ := strings.Cut(mod.TargetDeclarations[0].Target, ",")
			if reason := check(target, false); reason != "" {
				return reason
			}
		}
		if roles[name]["sass"] {
			disasm, err := cudasass.Parse(bytes.NewReader(data))
			if err != nil || disasm == nil {
				return "retained SASS target unavailable"
			}
			if disasm.Target != "" {
				if reason := check(disasm.Target, false); reason != "" {
					return reason
				}
			} else {
				for _, line := range strings.Split(string(data), "\n") {
					line = strings.TrimSpace(line)
					if line == ".target" || strings.HasPrefix(line, ".target ") || strings.HasPrefix(line, ".target\t") || line == "code for" || strings.HasPrefix(line, "code for ") {
						return "retained SASS target unavailable"
					}
				}
			}
		}
		if roles[name]["cubin"] {
			image, err := cudakernel.ParseCUBIN(data)
			if err != nil || image == nil || image.Arch == "" {
				// Opaque artifact bytes or an unknown ELF ABI supply no target fact.
				// Their digest can still check the caller's existing image association;
				// do not invent a hardware target or a new binary-eligibility contract.
				continue
			}
			// The existing ELF reader exposes a base SM only. Compare that known base;
			// never manufacture an a/f feature suffix from ELF flags it does not decode.
			if reason := check(image.Arch, true); reason != "" {
				return reason
			}
		}
	}
	return ""
}

// Read critical status fields without float64 conversion or last-key-wins
// decoding. Other producer fields remain opaque and are not new schema inputs.
func profileShareStatus(cap *cudacapture.Capture) (map[string]json.RawMessage, error) {
	data := cap.Files["capture-status.json"]
	if len(data) == 0 || len(data) > 64<<20 {
		return nil, fmt.Errorf("capture status unavailable or oversized")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	if err := profileShareJSONKeys(d, 0); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, fmt.Errorf("capture status has trailing JSON")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil || fields == nil {
		return nil, fmt.Errorf("capture status is not an object")
	}
	if nested, ok := fields["status"]; ok {
		fields = nil
		if err := json.Unmarshal(nested, &fields); err != nil || fields == nil {
			return nil, fmt.Errorf("capture status is not an object")
		}
	}
	return fields, nil
}

func profileShareJSONKeys(d *json.Decoder, depth int) error {
	if depth > 64 {
		return fmt.Errorf("capture status nesting limit exceeded")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			token, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := token.(string)
			if !ok || seen[strings.ToLower(key)] {
				return fmt.Errorf("capture status has duplicate or aliased fields")
			}
			seen[strings.ToLower(key)] = true
			if err := profileShareJSONKeys(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := profileShareJSONKeys(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("capture status delimiter invalid")
	}
	_, err = d.Token()
	return err
}
