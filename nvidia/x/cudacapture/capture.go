package cudacapture

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/google/pprof/profile"
)

// Wire protocol prefix for v2 captures embedded in pprof comments.
const (
	captureV2Prefix      = "#cudaprof:capture:v2\n"
	captureV3Prefix      = "#cudaprof:capture:v3\n"
	captureV1Prefix      = "#cudaprof:capture:v1\n"
	MaxCaptureFiles      = 4096
	MaxCaptureFileBytes  = 64 << 20
	MaxCaptureTotalBytes = 256 << 20
	MaxCaptureInputBytes = 512 << 20
)

// Reserved filenames that cannot be attached as arbitrary data files.
var reservedNames = []string{
	"gpu.pprof",
	"capture.pprof",
	"manifest.json",
}

// CodeFiles specifies the filenames within the capture corresponding to
// the CUDA source, PTX, CUBIN, and SASS disassembly, and optional control flow graphs.
type CodeFiles struct {
	Source string `json:"source"`
	PTX    string `json:"ptx"`
	CUBIN  string `json:"cubin"`
	SASS   string `json:"sass"`
	TileIR string `json:"tileir,omitempty"`
	CFG    string `json:"cfg,omitempty"`
	BBCFG  string `json:"bbcfg,omitempty"`
}

// Module identifies one loaded code image generation. File roles may be empty
// when the corresponding artifact was unavailable; CodeUnavailable records why.
type Module struct {
	ID         string    `json:"id"`
	Generation uint64    `json:"generation,omitempty"`
	Name       string    `json:"name,omitempty"`
	Code       CodeFiles `json:"code,omitempty"`
}

// MetricDescriptor describes the native meaning and scope of an observation.
// NumericType is retained so readers do not reinterpret uint64 or distributions.
type MetricDescriptor struct {
	ID          string `json:"id"`
	Namespace   string `json:"namespace,omitempty"`
	Name        string `json:"name"`
	Definition  string `json:"definition,omitempty"`
	Unit        string `json:"unit,omitempty"`
	NumericType string `json:"numeric_type,omitempty"`
	Aggregation string `json:"aggregation,omitempty"`
	Scope       string `json:"scope,omitempty"`
	Denominator string `json:"denominator,omitempty"`
	Window      string `json:"window,omitempty"`
	Collector   string `json:"collector,omitempty"`
	Support     string `json:"support,omitempty"`
}

// Observation is one typed native value. Value is JSON to preserve integers,
// finite floats, and structured distributions without lossy conversion.
type Observation struct {
	DescriptorID string          `json:"descriptor_id"`
	EntityID     string          `json:"entity_id,omitempty"`
	RunID        string          `json:"run_id,omitempty"`
	PassID       string          `json:"pass_id,omitempty"`
	Start        uint64          `json:"start,omitempty"`
	End          uint64          `json:"end,omitempty"`
	Instant      uint64          `json:"instant,omitempty"`
	Value        json.RawMessage `json:"value,omitempty"`
	Availability string          `json:"availability"`
}

// Capture holds in-memory profiling data, attached files, code file role mappings,
// and opaque JSON metadata.
type Capture struct {
	Profile      *profile.Profile
	Files        map[string][]byte
	Code         CodeFiles
	Modules      []Module
	Descriptors  []MetricDescriptor
	Observations []Observation
	// CodeUnavailable maps source, ptx, cubin, and sass to an explicit
	// disposition when that artifact was not retained or could not be captured.
	CodeUnavailable map[string]string
	Metadata        json.RawMessage
}

// Manifest represents the serialization schema for manifest.json.
type Manifest struct {
	Version      int                `json:"version"`
	Code         CodeFiles          `json:"code,omitempty"`
	Modules      []Module           `json:"modules,omitempty"`
	Descriptors  []MetricDescriptor `json:"descriptors,omitempty"`
	Observations []Observation      `json:"observations,omitempty"`
	Unavailable  map[string]string  `json:"code_unavailable,omitempty"`
	SHA256       map[string]string  `json:"sha256"`
	Metadata     json.RawMessage    `json:"metadata,omitempty"`
}

// embeddedCaptureV2 is the JSON structure stored inside pprof comments in v2.
type embeddedCaptureV2 struct {
	Manifest Manifest          `json:"manifest"`
	Files    map[string][]byte `json:"files"`
}

// embeddedCaptureV1 represents the legacy v1 embedded capture schema.
type embeddedCaptureV1 struct {
	Manifest legacyV1Manifest  `json:"manifest"`
	Files    map[string][]byte `json:"files"`
}

type legacyV1Manifest struct {
	Version      int                `json:"version"`
	Created      string             `json:"created"`
	Device       string             `json:"device"`
	Arch         string             `json:"architecture"`
	GoVersion    string             `json:"go_version"`
	NVRTCVersion string             `json:"nvrtc_version"`
	Elements     int                `json:"elements"`
	Passes       int                `json:"passes"`
	Rounds       int                `json:"fma_rounds"`
	Intervals    map[string][]int64 `json:"cuda_event_nanoseconds"`
	Verified     int                `json:"verified_results"`
	MaxError     float64            `json:"max_absolute_error"`
	SHA256       map[string]string  `json:"sha256"`
}

// validateFilename checks that name is a portable, clean, non-empty, non-reserved basename.
func validateFilename(name string) error {
	if name == "" {
		return errors.New("cudacapture: empty filename")
	}
	if filepath.Base(name) != name || strings.ContainsAny(name, "/\\:") {
		return fmt.Errorf("cudacapture: filename %q is not a flat relative name", name)
	}
	if name == "." || name == ".." {
		return fmt.Errorf("cudacapture: invalid filename %q", name)
	}
	lower := strings.ToLower(name)
	for _, res := range reservedNames {
		if lower == strings.ToLower(res) {
			return fmt.Errorf("cudacapture: filename %q is reserved", name)
		}
	}
	return nil
}

func rejectDuplicateJSON(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	var walk func() error
	walk = func() error {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		delim, ok := tok.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]struct{})
			for dec.More() {
				key, err := dec.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok {
					return errors.New("cudacapture: invalid JSON object key")
				}
				if _, exists := seen[name]; exists {
					return fmt.Errorf("cudacapture: duplicate JSON member %q", name)
				}
				seen[name] = struct{}{}
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		case '[':
			for dec.More() {
				if err := walk(); err != nil {
					return err
				}
			}
			_, err = dec.Token()
			return err
		default:
			return errors.New("cudacapture: invalid JSON delimiter")
		}
	}
	if err := walk(); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		if err == nil {
			return errors.New("cudacapture: trailing JSON value")
		}
		return err
	}
	return nil
}

// validateCapture performs sanity checks on the Capture structure.
func validateCapture(c *Capture, l Limits) error {
	if c == nil {
		return errors.New("cudacapture: nil capture")
	}
	if c.Profile == nil {
		return errors.New("cudacapture: missing profile")
	}
	if err := c.Profile.CheckValid(); err != nil {
		return fmt.Errorf("cudacapture: invalid profile: %w", err)
	}
	if len(c.Files) == 0 {
		return errors.New("cudacapture: empty file set")
	}
	if len(c.Files) > l.Files {
		return fmt.Errorf("cudacapture: too many files: %d", len(c.Files))
	}
	var totalBytes int64
	for name, data := range c.Files {
		if int64(len(data)) > l.FileBytes {
			return fmt.Errorf("cudacapture: file %q exceeds size limit", name)
		}
		totalBytes += int64(len(data))
		if totalBytes > l.TotalBytes {
			return errors.New("cudacapture: files exceed total size limit")
		}
	}
	roles := map[string]string{"source": c.Code.Source, "ptx": c.Code.PTX, "cubin": c.Code.CUBIN, "sass": c.Code.SASS}
	for role, name := range roles {
		if name == "" {
			if c.CodeUnavailable == nil || c.CodeUnavailable[role] == "" {
				return errors.New("cudacapture: missing required code roles (source, ptx, cubin, sass)")
			}
			continue
		}
		if _, ok := c.Files[name]; !ok {
			return fmt.Errorf("cudacapture: %s file %q not found in files", role, name)
		}
	}
	for role, name := range map[string]string{"cfg": c.Code.CFG, "bbcfg": c.Code.BBCFG, "tileir": c.Code.TileIR} {
		if name != "" {
			if _, ok := c.Files[name]; !ok {
				return fmt.Errorf("cudacapture: %s file %q not found in files", role, name)
			}
		}
	}

	for _, module := range c.Modules {
		if module.ID == "" {
			return errors.New("cudacapture: module is missing identity")
		}
		for role, name := range map[string]string{"source": module.Code.Source, "ptx": module.Code.PTX, "cubin": module.Code.CUBIN, "sass": module.Code.SASS, "tileir": module.Code.TileIR, "cfg": module.Code.CFG, "bbcfg": module.Code.BBCFG} {
			if name == "" {
				continue
			}
			if _, ok := c.Files[name]; !ok {
				return fmt.Errorf("cudacapture: module %q %s file %q not found in files", module.ID, role, name)
			}
		}
	}
	descriptors := make(map[string]struct{}, len(c.Descriptors))
	for _, descriptor := range c.Descriptors {
		if descriptor.ID == "" || descriptor.Name == "" {
			return errors.New("cudacapture: metric descriptor requires id and name")
		}
		if _, exists := descriptors[descriptor.ID]; exists {
			return fmt.Errorf("cudacapture: duplicate metric descriptor %q", descriptor.ID)
		}
		descriptors[descriptor.ID] = struct{}{}
	}
	validAvailability := map[string]bool{"collected": true, "not_requested": true, "unsupported": true, "permission_denied": true, "conflict": true, "not_observed": true, "not_implemented": true, "failed": true, "budget_exhausted": true, "lost_or_incomplete": true}
	for _, observation := range c.Observations {
		if _, ok := descriptors[observation.DescriptorID]; !ok {
			return fmt.Errorf("cudacapture: observation references unknown descriptor %q", observation.DescriptorID)
		}
		if !validAvailability[observation.Availability] {
			return fmt.Errorf("cudacapture: invalid observation availability %q", observation.Availability)
		}
		if observation.End != 0 && observation.Start != 0 && observation.End < observation.Start {
			return errors.New("cudacapture: observation interval ends before it starts")
		}
		if len(observation.Value) > 0 {
			var value any
			if err := json.Unmarshal(observation.Value, &value); err != nil {
				return fmt.Errorf("cudacapture: invalid observation value: %w", err)
			}
		}
		if observation.Availability == "collected" && len(observation.Value) == 0 {
			return errors.New("cudacapture: collected observation is missing value")
		}
	}

	caseMap := make(map[string]string, len(c.Files))
	for name := range c.Files {
		if err := validateFilename(name); err != nil {
			return err
		}
		lower := strings.ToLower(name)
		if orig, exists := caseMap[lower]; exists && orig != name {
			return fmt.Errorf("cudacapture: case collision between %q and %q", orig, name)
		}
		caseMap[lower] = name
	}

	if len(c.Metadata) > 0 {
		var js json.RawMessage
		if err := json.Unmarshal(c.Metadata, &js); err != nil {
			return fmt.Errorf("cudacapture: invalid metadata JSON: %w", err)
		}
	}

	return nil
}

// Write serializes the capture as a single pprof profile with embedded files
// and metadata, using the default limits. It is WriteLimits with a zero Limits.
func Write(w io.Writer, c *Capture) error {
	return WriteLimits(w, c, Limits{})
}

// WriteLimits is Write bounded by l. See Limits.
func WriteLimits(w io.Writer, c *Capture, l Limits) error {
	l = l.withDefaults()
	if err := validateCapture(c, l); err != nil {
		return err
	}

	p := c.Profile.Copy()
	// Ensure profile does not already have capture comments
	for _, comment := range p.Comments {
		if strings.HasPrefix(comment, "#cudaprof:capture:") {
			return errors.New("cudacapture: profile already contains an embedded capture")
		}
	}

	version := 2
	if len(c.Modules) > 0 || len(c.CodeUnavailable) > 0 || len(c.Descriptors) > 0 || len(c.Observations) > 0 {
		version = 3
	}
	manifest := Manifest{
		Version:      version,
		Code:         c.Code,
		Modules:      c.Modules,
		Descriptors:  c.Descriptors,
		Observations: c.Observations,
		Unavailable:  c.CodeUnavailable,
		SHA256:       make(map[string]string, len(c.Files)),
		Metadata:     c.Metadata,
	}
	for name, data := range c.Files {
		manifest.SHA256[name] = fmt.Sprintf("%x", sha256.Sum256(data))
	}

	embedded := embeddedCaptureV2{
		Manifest: manifest,
		Files:    c.Files,
	}
	data, err := json.Marshal(embedded)
	if err != nil {
		return fmt.Errorf("cudacapture: marshal embedded capture: %w", err)
	}

	prefix := captureV2Prefix
	if version == 3 {
		prefix = captureV3Prefix
	}
	p.Comments = append(p.Comments,
		captureBanner(c),
		prefix+string(data),
	)

	return p.Write(w)
}

func captureBanner(c *Capture) string {
	roles := make([]string, 0, 4)
	for role, name := range map[string]string{"source": c.Code.Source, "ptx": c.Code.PTX, "cubin": c.Code.CUBIN, "sass": c.Code.SASS} {
		if name != "" {
			roles = append(roles, role)
		}
	}
	sort.Strings(roles)
	if len(roles) == 0 {
		return "No source, PTX, CUBIN, or SASS code artifacts were captured; availability is recorded in the manifest."
	}
	return fmt.Sprintf("Embedded code artifacts: %s; use cudaprof for available source and NVIDIA disassembly.", strings.Join(roles, ", "))
}

// Read reads a capture from a reader containing a pprof profile with embedded
// artifacts, using the default limits. It is ReadLimits with a zero Limits.
func Read(r io.Reader) (*Capture, error) {
	return ReadLimits(r, Limits{})
}

// ReadLimits is Read bounded by l. Decoding a capture from an untrusted source
// should keep the defaults; see Limits.
func ReadLimits(r io.Reader, l Limits) (*Capture, error) {
	l = l.withDefaults()
	data, err := io.ReadAll(io.LimitReader(r, l.InputBytes+1))
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read profile data: %w", err)
	}
	if int64(len(data)) > l.InputBytes {
		return nil, errors.New("cudacapture: profile exceeds input size limit")
	}
	p, err := profile.ParseData(data)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: parse profile: %w", err)
	}
	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("cudacapture: invalid profile: %w", err)
	}

	var v2Comment, v3Comment, v1Comment string
	for _, c := range p.Comments {
		if strings.HasPrefix(c, "#cudaprof:capture:") {
			if strings.HasPrefix(c, captureV2Prefix) {
				if v2Comment != "" || v3Comment != "" || v1Comment != "" {
					return nil, errors.New("cudacapture: multiple embedded captures found")
				}
				v2Comment = strings.TrimPrefix(c, captureV2Prefix)
			} else if strings.HasPrefix(c, captureV3Prefix) {
				if v2Comment != "" || v3Comment != "" || v1Comment != "" {
					return nil, errors.New("cudacapture: multiple embedded captures found")
				}
				v3Comment = strings.TrimPrefix(c, captureV3Prefix)
			} else if strings.HasPrefix(c, captureV1Prefix) {
				if v2Comment != "" || v3Comment != "" || v1Comment != "" {
					return nil, errors.New("cudacapture: multiple embedded captures found")
				}
				v1Comment = strings.TrimPrefix(c, captureV1Prefix)
			} else {
				return nil, errors.New("cudacapture: unsupported embedded capture version")
			}
		}
	}

	if v2Comment != "" {
		return decodeEmbeddedV2(p, []byte(v2Comment), l)
	}
	if v3Comment != "" {
		return decodeEmbeddedV2(p, []byte(v3Comment), l)
	}
	if v1Comment != "" {
		return decodeEmbeddedV1(p, []byte(v1Comment), l)
	}

	return nil, errors.New("cudacapture: no embedded capture found in profile")
}

func decodeEmbeddedV2(p *profile.Profile, data []byte, l Limits) (*Capture, error) {
	if err := rejectDuplicateJSON(data); err != nil {
		return nil, err
	}
	var emb embeddedCaptureV2
	if err := json.Unmarshal(data, &emb); err != nil {
		return nil, fmt.Errorf("cudacapture: unmarshal v2 embedded: %w", err)
	}
	if emb.Manifest.Version != 2 && emb.Manifest.Version != 3 {
		return nil, fmt.Errorf("cudacapture: unsupported manifest version %d in v2 payload", emb.Manifest.Version)
	}

	if len(emb.Files) > l.Files {
		return nil, errors.New("cudacapture: embedded file count exceeds limit")
	}
	// Verify all file checksums
	if len(emb.Files) != len(emb.Manifest.SHA256) {
		return nil, errors.New("cudacapture: file count mismatch with manifest")
	}
	for name, fileBytes := range emb.Files {
		if int64(len(fileBytes)) > l.FileBytes {
			return nil, fmt.Errorf("cudacapture: embedded file %q exceeds size limit", name)
		}
		wantSum, ok := emb.Manifest.SHA256[name]
		if !ok {
			return nil, fmt.Errorf("cudacapture: file %q not in manifest checksums", name)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(fileBytes))
		if gotSum != wantSum {
			return nil, fmt.Errorf("cudacapture: checksum mismatch for %s", name)
		}
	}
	var totalBytes int64
	for _, fileBytes := range emb.Files {
		totalBytes += int64(len(fileBytes))
		if totalBytes > l.TotalBytes {
			return nil, errors.New("cudacapture: embedded files exceed total size limit")
		}
	}

	cap := &Capture{
		Profile:         p,
		Files:           emb.Files,
		Code:            emb.Manifest.Code,
		Modules:         emb.Manifest.Modules,
		CodeUnavailable: emb.Manifest.Unavailable,
		Descriptors:     emb.Manifest.Descriptors,
		Observations:    emb.Manifest.Observations,
		Metadata:        emb.Manifest.Metadata,
	}
	if err := validateCapture(cap, l); err != nil {
		return nil, err
	}
	return cap, nil
}

func decodeEmbeddedV1(p *profile.Profile, data []byte, l Limits) (*Capture, error) {
	if err := rejectDuplicateJSON(data); err != nil {
		return nil, err
	}
	var emb embeddedCaptureV1
	if err := json.Unmarshal(data, &emb); err != nil {
		return nil, fmt.Errorf("cudacapture: unmarshal v1 embedded: %w", err)
	}
	if emb.Manifest.Version != 1 {
		return nil, fmt.Errorf("cudacapture: unsupported manifest version %d in v1 payload", emb.Manifest.Version)
	}
	if len(emb.Files) > l.Files {
		return nil, errors.New("cudacapture: embedded file count exceeds limit")
	}

	var totalBytes int64
	for name, fileBytes := range emb.Files {
		if err := validateFilename(name); err != nil {
			return nil, err
		}
		if int64(len(fileBytes)) > l.FileBytes {
			return nil, fmt.Errorf("cudacapture: embedded file %q exceeds size limit", name)
		}
		totalBytes += int64(len(fileBytes))
		if totalBytes > l.TotalBytes {
			return nil, errors.New("cudacapture: embedded files exceed total size limit")
		}
		wantSum, ok := emb.Manifest.SHA256[name]
		if !ok {
			return nil, fmt.Errorf("cudacapture: file %q not in v1 manifest checksums", name)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(fileBytes))
		if gotSum != wantSum {
			return nil, fmt.Errorf("cudacapture: checksum mismatch for %s", name)
		}
	}

	metaBytes, err := json.Marshal(emb.Manifest)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: marshal v1 metadata: %w", err)
	}

	cap := &Capture{
		Profile: p,
		Files:   emb.Files,
		Code: CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Metadata: metaBytes,
	}
	return cap, nil
}

// WriteDir writes the capture to dir as separate files (gpu.pprof,
// manifest.json, and data files), using the default limits. It is
// WriteDirLimits with a zero Limits.
func WriteDir(dir string, c *Capture) error {
	return WriteDirLimits(dir, c, Limits{})
}

// WriteDirLimits is WriteDir bounded by l. See Limits.
func WriteDirLimits(dir string, c *Capture, l Limits) error {
	l = l.withDefaults()
	if err := validateCapture(c, l); err != nil {
		return err
	}
	if info, err := os.Lstat(dir); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return fmt.Errorf("cudacapture: output path %q is not a directory", dir)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("cudacapture: check output directory: %w", err)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	version := 2
	if len(c.Modules) > 0 || len(c.CodeUnavailable) > 0 || len(c.Descriptors) > 0 || len(c.Observations) > 0 {
		version = 3
	}
	manifest := Manifest{
		Version:      version,
		Code:         c.Code,
		Modules:      c.Modules,
		Descriptors:  c.Descriptors,
		Observations: c.Observations,
		Unavailable:  c.CodeUnavailable,
		SHA256:       make(map[string]string, len(c.Files)),
		Metadata:     c.Metadata,
	}
	// Check every destination before writing. The later O_EXCL creates also
	// protect against a concurrent writer winning after this preflight.
	outputNames := make([]string, 0, len(c.Files)+2)
	for name := range c.Files {
		outputNames = append(outputNames, name)
	}
	outputNames = append(outputNames, "manifest.json", "gpu.pprof")
	for _, name := range outputNames {
		if _, err := os.Lstat(filepath.Join(dir, name)); err == nil {
			return fmt.Errorf("cudacapture: output file already exists: %s", name)
		} else if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("cudacapture: check output file %q: %w", name, err)
		}
	}

	for name, data := range c.Files {
		manifest.SHA256[name] = fmt.Sprintf("%x", sha256.Sum256(data))
		path := filepath.Join(dir, name)
		if err := writeNewFile(path, data, 0644); err != nil {
			return fmt.Errorf("cudacapture: write file %q: %w", name, err)
		}
	}

	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("cudacapture: marshal manifest: %w", err)
	}
	if err := writeNewFile(filepath.Join(dir, "manifest.json"), append(manifestData, '\n'), 0644); err != nil {
		return fmt.Errorf("cudacapture: write manifest.json: %w", err)
	}

	f, err := os.OpenFile(filepath.Join(dir, "gpu.pprof"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return fmt.Errorf("cudacapture: create gpu.pprof: %w", err)
	}
	if err := c.Profile.Write(f); err != nil {
		_ = f.Close()
		_ = os.Remove(filepath.Join(dir, "gpu.pprof"))
		return fmt.Errorf("cudacapture: write gpu.pprof: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("cudacapture: close gpu.pprof: %w", err)
	}
	return nil
}

func writeNewFile(path string, data []byte, mode os.FileMode) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return err
	}
	return nil
}

// readRegularFile reads a bounded regular file without following symlinks.
// Directory captures are untrusted input, so the size check happens before
// allocation and is repeated while reading in case the file changes.
func readRegularFile(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
		return nil, fmt.Errorf("file is not a regular file")
	}
	if info.Size() > limit {
		return nil, fmt.Errorf("file exceeds size limit")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("file exceeds size limit")
	}
	return data, nil
}

// ReadDir loads an unpacked capture from dir. Supports both v2 and legacy v1 directory layouts.
func ReadDir(dir string) (*Capture, error) {
	return ReadDirLimits(dir, Limits{})
}

// ReadDirLimits is ReadDir bounded by l. Reading a directory from an untrusted
// source should keep the defaults; see Limits.
func ReadDirLimits(dir string, l Limits) (*Capture, error) {
	l = l.withDefaults()
	manifestBytes, err := readRegularFile(filepath.Join(dir, "manifest.json"), l.FileBytes)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read manifest.json: %w", err)
	}
	if err := rejectDuplicateJSON(manifestBytes); err != nil {
		return nil, err
	}

	var probe struct {
		Version int `json:"version"`
	}
	if err := json.Unmarshal(manifestBytes, &probe); err != nil {
		return nil, fmt.Errorf("cudacapture: parse manifest.json: %w", err)
	}

	switch probe.Version {
	case 2, 3:
		return readDirV2(dir, manifestBytes, l)
	case 1:
		return readDirV1(dir, manifestBytes, l)
	default:
		return nil, fmt.Errorf("cudacapture: unsupported manifest version %d", probe.Version)
	}
}

func readDirV2(dir string, manifestBytes []byte, l Limits) (*Capture, error) {
	var m Manifest
	if err := json.Unmarshal(manifestBytes, &m); err != nil {
		return nil, fmt.Errorf("cudacapture: parse manifest v2: %w", err)
	}

	if len(m.SHA256) > l.Files {
		return nil, errors.New("cudacapture: directory file count exceeds limit")
	}
	files := make(map[string][]byte, len(m.SHA256))
	var totalBytes int64
	for name, wantSum := range m.SHA256 {
		if err := validateFilename(name); err != nil {
			return nil, err
		}
		info, err := os.Lstat(filepath.Join(dir, name))
		if err != nil {
			return nil, fmt.Errorf("cudacapture: stat file %q: %w", name, err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return nil, fmt.Errorf("cudacapture: file %q is not a regular file", name)
		}
		if info.Size() > l.FileBytes {
			return nil, fmt.Errorf("cudacapture: file %q exceeds size limit", name)
		}
		totalBytes += info.Size()
		if totalBytes > l.TotalBytes {
			return nil, errors.New("cudacapture: directory files exceed total size limit")
		}
		data, err := readRegularFile(filepath.Join(dir, name), l.FileBytes)
		if err != nil {
			return nil, fmt.Errorf("cudacapture: read file %q: %w", name, err)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(data))
		if gotSum != wantSum {
			return nil, fmt.Errorf("cudacapture: checksum mismatch for %s", name)
		}
		files[name] = data
	}

	pprofBytes, err := readRegularFile(filepath.Join(dir, "gpu.pprof"), l.InputBytes)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read gpu.pprof: %w", err)
	}
	p, err := profile.ParseData(pprofBytes)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: parse gpu.pprof: %w", err)
	}
	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("cudacapture: invalid gpu.pprof: %w", err)
	}

	cap := &Capture{
		Profile:         p,
		Files:           files,
		Code:            m.Code,
		Modules:         m.Modules,
		CodeUnavailable: m.Unavailable,
		Descriptors:     m.Descriptors,
		Observations:    m.Observations,
		Metadata:        m.Metadata,
	}
	if err := validateCapture(cap, l); err != nil {
		return nil, err
	}
	return cap, nil
}

func readDirV1(dir string, manifestBytes []byte, l Limits) (*Capture, error) {
	var m legacyV1Manifest
	if err := json.Unmarshal(manifestBytes, &m); err != nil {
		return nil, fmt.Errorf("cudacapture: parse manifest v1: %w", err)
	}

	v1Files := []string{"gpu.pprof", "go.trace", "main.go", "capture.go", "workload.go", "kernels.cu", "kernels.ptx", "kernels.cubin", "kernels.sass", "nvdisasm.txt", "README.md"}
	files := make(map[string][]byte)
	var totalBytes int64
	for _, name := range v1Files {
		if name == "gpu.pprof" {
			continue
		}
		if err := validateFilename(name); err != nil {
			return nil, err
		}
		info, err := os.Lstat(filepath.Join(dir, name))
		if err != nil {
			return nil, fmt.Errorf("cudacapture: stat file %q: %w", name, err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return nil, fmt.Errorf("cudacapture: file %q is not a regular file", name)
		}
		if info.Size() > l.FileBytes {
			return nil, fmt.Errorf("cudacapture: file %q exceeds size limit", name)
		}
		totalBytes += info.Size()
		if totalBytes > l.TotalBytes {
			return nil, errors.New("cudacapture: directory files exceed total size limit")
		}
		data, err := readRegularFile(filepath.Join(dir, name), l.FileBytes)
		if err != nil {
			return nil, fmt.Errorf("cudacapture: read file %q: %w", name, err)
		}
		if wantSum, ok := m.SHA256[name]; ok && wantSum != "" {
			gotSum := fmt.Sprintf("%x", sha256.Sum256(data))
			if gotSum != wantSum {
				return nil, fmt.Errorf("cudacapture: checksum mismatch for %s", name)
			}
		}
		files[name] = data
	}

	var cfgFile, bbcfgFile string
	for _, name := range []string{"kernels.cfg.dot", "kernels.bbcfg.dot"} {
		path := filepath.Join(dir, name)
		info, err := os.Lstat(path)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, fmt.Errorf("cudacapture: stat file %q: %w", name, err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return nil, fmt.Errorf("cudacapture: file %q is not a regular file", name)
		}
		if info.Size() > l.FileBytes {
			return nil, fmt.Errorf("cudacapture: file %q exceeds size limit", name)
		}
		totalBytes += info.Size()
		if totalBytes > l.TotalBytes {
			return nil, errors.New("cudacapture: directory files exceed total size limit")
		}
		data, err := readRegularFile(path, l.FileBytes)
		if err != nil {
			return nil, fmt.Errorf("cudacapture: read file %q: %w", name, err)
		}
		if wantSum, ok := m.SHA256[name]; ok && wantSum != "" {
			gotSum := fmt.Sprintf("%x", sha256.Sum256(data))
			if gotSum != wantSum {
				return nil, fmt.Errorf("cudacapture: checksum mismatch for %s", name)
			}
		}
		files[name] = data
		if name == "kernels.cfg.dot" {
			cfgFile = name
		} else if name == "kernels.bbcfg.dot" {
			bbcfgFile = name
		}
	}

	pprofBytes, err := readRegularFile(filepath.Join(dir, "gpu.pprof"), l.FileBytes)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read gpu.pprof: %w", err)
	}
	p, err := profile.ParseData(pprofBytes)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: parse gpu.pprof: %w", err)
	}
	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("cudacapture: invalid gpu.pprof: %w", err)
	}

	return &Capture{
		Profile: p,
		Files:   files,
		Code: CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
			CFG:    cfgFile,
			BBCFG:  bbcfgFile,
		},
		Metadata: manifestBytes,
	}, nil
}
