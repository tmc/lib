package cudacapture

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/pprof/profile"
)

// Wire protocol prefix for v2 captures embedded in pprof comments.
const (
	captureV2Prefix = "#cudaprof:capture:v2\n"
	captureV1Prefix = "#cudaprof:capture:v1\n"
)

// Reserved filenames that cannot be attached as arbitrary data files.
var reservedNames = []string{
	"gpu.pprof",
	"capture.pprof",
	"manifest.json",
}

// CodeFiles specifies the filenames within the capture corresponding to
// the CUDA source, PTX, CUBIN, and SASS disassembly.
type CodeFiles struct {
	Source string `json:"source"`
	PTX    string `json:"ptx"`
	CUBIN  string `json:"cubin"`
	SASS   string `json:"sass"`
}

// Capture holds in-memory profiling data, attached files, code file role mappings,
// and opaque JSON metadata.
type Capture struct {
	Profile  *profile.Profile
	Files    map[string][]byte
	Code     CodeFiles
	Metadata json.RawMessage
}

// Manifest represents the serialization schema for manifest.json.
type Manifest struct {
	Version  int               `json:"version"`
	Code     CodeFiles         `json:"code,omitempty"`
	SHA256   map[string]string `json:"sha256"`
	Metadata json.RawMessage   `json:"metadata,omitempty"`
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

// validateCapture performs sanity checks on the Capture structure.
func validateCapture(c *Capture) error {
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
	if c.Code.Source == "" || c.Code.PTX == "" || c.Code.CUBIN == "" || c.Code.SASS == "" {
		return errors.New("cudacapture: missing required code roles (source, ptx, cubin, sass)")
	}

	// Verify all code files exist in Files
	for role, name := range map[string]string{
		"source": c.Code.Source,
		"ptx":    c.Code.PTX,
		"cubin":  c.Code.CUBIN,
		"sass":   c.Code.SASS,
	} {
		if _, ok := c.Files[name]; !ok {
			return fmt.Errorf("cudacapture: %s file %q not found in files", role, name)
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

// Write serializes the capture as a single pprof profile with embedded files and metadata.
func Write(w io.Writer, c *Capture) error {
	if err := validateCapture(c); err != nil {
		return err
	}

	p := c.Profile.Copy()
	// Ensure profile does not already have capture comments
	for _, comment := range p.Comments {
		if strings.HasPrefix(comment, "#cudaprof:capture:") {
			return errors.New("cudacapture: profile already contains an embedded capture")
		}
	}

	manifest := Manifest{
		Version:  2,
		Code:     c.Code,
		SHA256:   make(map[string]string, len(c.Files)),
		Metadata: c.Metadata,
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

	p.Comments = append(p.Comments,
		"Source, PTX, SASS, CUBIN, and artifacts are embedded; use cudaprof for source and NVIDIA disassembly.",
		captureV2Prefix+string(data),
	)

	return p.Write(w)
}

// Read reads a capture from a reader containing a pprof profile with embedded artifacts.
func Read(r io.Reader) (*Capture, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read profile data: %w", err)
	}
	p, err := profile.ParseData(data)
	if err != nil {
		return nil, fmt.Errorf("cudacapture: parse profile: %w", err)
	}
	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("cudacapture: invalid profile: %w", err)
	}

	var v2Comment, v1Comment string
	for _, c := range p.Comments {
		if strings.HasPrefix(c, "#cudaprof:capture:") {
			if strings.HasPrefix(c, captureV2Prefix) {
				if v2Comment != "" || v1Comment != "" {
					return nil, errors.New("cudacapture: multiple embedded captures found")
				}
				v2Comment = strings.TrimPrefix(c, captureV2Prefix)
			} else if strings.HasPrefix(c, captureV1Prefix) {
				if v2Comment != "" || v1Comment != "" {
					return nil, errors.New("cudacapture: multiple embedded captures found")
				}
				v1Comment = strings.TrimPrefix(c, captureV1Prefix)
			} else {
				return nil, errors.New("cudacapture: unsupported embedded capture version")
			}
		}
	}

	if v2Comment != "" {
		return decodeEmbeddedV2(p, []byte(v2Comment))
	}
	if v1Comment != "" {
		return decodeEmbeddedV1(p, []byte(v1Comment))
	}

	return nil, errors.New("cudacapture: no embedded capture found in profile")
}

func decodeEmbeddedV2(p *profile.Profile, data []byte) (*Capture, error) {
	var emb embeddedCaptureV2
	if err := json.Unmarshal(data, &emb); err != nil {
		return nil, fmt.Errorf("cudacapture: unmarshal v2 embedded: %w", err)
	}
	if emb.Manifest.Version != 2 {
		return nil, fmt.Errorf("cudacapture: unsupported manifest version %d in v2 payload", emb.Manifest.Version)
	}

	// Verify all file checksums
	if len(emb.Files) != len(emb.Manifest.SHA256) {
		return nil, errors.New("cudacapture: file count mismatch with manifest")
	}
	for name, fileBytes := range emb.Files {
		wantSum, ok := emb.Manifest.SHA256[name]
		if !ok {
			return nil, fmt.Errorf("cudacapture: file %q not in manifest checksums", name)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(fileBytes))
		if gotSum != wantSum {
			return nil, fmt.Errorf("cudacapture: checksum mismatch for %s", name)
		}
	}

	cap := &Capture{
		Profile:  p,
		Files:    emb.Files,
		Code:     emb.Manifest.Code,
		Metadata: emb.Manifest.Metadata,
	}
	if err := validateCapture(cap); err != nil {
		return nil, err
	}
	return cap, nil
}

func decodeEmbeddedV1(p *profile.Profile, data []byte) (*Capture, error) {
	var emb embeddedCaptureV1
	if err := json.Unmarshal(data, &emb); err != nil {
		return nil, fmt.Errorf("cudacapture: unmarshal v1 embedded: %w", err)
	}
	if emb.Manifest.Version != 1 {
		return nil, fmt.Errorf("cudacapture: unsupported manifest version %d in v1 payload", emb.Manifest.Version)
	}

	for name, fileBytes := range emb.Files {
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

// WriteDir writes the capture to dir as separate files (gpu.pprof, manifest.json, and data files).
func WriteDir(dir string, c *Capture) error {
	if err := validateCapture(c); err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	manifest := Manifest{
		Version:  2,
		Code:     c.Code,
		SHA256:   make(map[string]string, len(c.Files)),
		Metadata: c.Metadata,
	}

	for name, data := range c.Files {
		manifest.SHA256[name] = fmt.Sprintf("%x", sha256.Sum256(data))
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, data, 0644); err != nil {
			return fmt.Errorf("cudacapture: write file %q: %w", name, err)
		}
	}

	manifestData, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("cudacapture: marshal manifest: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), append(manifestData, '\n'), 0644); err != nil {
		return fmt.Errorf("cudacapture: write manifest.json: %w", err)
	}

	f, err := os.Create(filepath.Join(dir, "gpu.pprof"))
	if err != nil {
		return fmt.Errorf("cudacapture: create gpu.pprof: %w", err)
	}
	defer f.Close()
	if err := c.Profile.Write(f); err != nil {
		return fmt.Errorf("cudacapture: write gpu.pprof: %w", err)
	}

	return nil
}

// ReadDir loads an unpacked capture from dir. Supports both v2 and legacy v1 directory layouts.
func ReadDir(dir string) (*Capture, error) {
	manifestBytes, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("cudacapture: read manifest.json: %w", err)
	}

	var probe struct {
		Version int `json:"version"`
	}
	if err := json.Unmarshal(manifestBytes, &probe); err != nil {
		return nil, fmt.Errorf("cudacapture: parse manifest.json: %w", err)
	}

	switch probe.Version {
	case 2:
		return readDirV2(dir, manifestBytes)
	case 1:
		return readDirV1(dir, manifestBytes)
	default:
		return nil, fmt.Errorf("cudacapture: unsupported manifest version %d", probe.Version)
	}
}

func readDirV2(dir string, manifestBytes []byte) (*Capture, error) {
	var m Manifest
	if err := json.Unmarshal(manifestBytes, &m); err != nil {
		return nil, fmt.Errorf("cudacapture: parse manifest v2: %w", err)
	}

	files := make(map[string][]byte, len(m.SHA256))
	for name, wantSum := range m.SHA256 {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil, fmt.Errorf("cudacapture: read file %q: %w", name, err)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(data))
		if gotSum != wantSum {
			return nil, fmt.Errorf("cudacapture: checksum mismatch for %s", name)
		}
		files[name] = data
	}

	pprofBytes, err := os.ReadFile(filepath.Join(dir, "gpu.pprof"))
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
		Profile:  p,
		Files:    files,
		Code:     m.Code,
		Metadata: m.Metadata,
	}
	if err := validateCapture(cap); err != nil {
		return nil, err
	}
	return cap, nil
}

func readDirV1(dir string, manifestBytes []byte) (*Capture, error) {
	var m legacyV1Manifest
	if err := json.Unmarshal(manifestBytes, &m); err != nil {
		return nil, fmt.Errorf("cudacapture: parse manifest v1: %w", err)
	}

	v1Files := []string{"gpu.pprof", "go.trace", "main.go", "capture.go", "workload.go", "kernels.cu", "kernels.ptx", "kernels.cubin", "kernels.sass", "nvdisasm.txt", "README.md"}
	files := make(map[string][]byte)
	for _, name := range v1Files {
		if name == "gpu.pprof" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, name))
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

	pprofBytes, err := os.ReadFile(filepath.Join(dir, "gpu.pprof"))
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
		},
		Metadata: manifestBytes,
	}, nil
}
