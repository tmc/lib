package metalcapture

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

// Wire protocol prefix for Metal captures embedded in pprof comments.
const (
	captureV1Prefix = "#metalprof:capture:v1\n"
	captureV2Prefix = "#metalprof:capture:v2\n"
)

// Reserved filenames that cannot be attached as arbitrary data files.
var reservedNames = []string{
	"gpu.pprof",
	"capture.pprof",
	"manifest.json",
}

// CodeFiles specifies the filenames within the capture corresponding to
// the Metal Shading Language source, compiler IR (AIR), metallib container,
// and optional GPU assembly disassembly.
type CodeFiles struct {
	Source   string `json:"source"`
	AIR      string `json:"air,omitempty"`
	MetalLib string `json:"metallib,omitempty"`
	Assembly string `json:"assembly,omitempty"`
}

// Capture holds in-memory profiling data, attached files, code file role mappings,
// typed hardware/software observations, and opaque JSON metadata.
type Capture struct {
	Profile      *profile.Profile  `json:"-"`
	Files        map[string][]byte `json:"files"`
	Code         CodeFiles         `json:"code"`
	Metadata     json.RawMessage   `json:"metadata,omitempty"`
	Observations []Observation     `json:"observations,omitempty"`
}

// Manifest represents the serialization schema for manifest.json.
type Manifest struct {
	Version      int               `json:"version"`
	Code         CodeFiles         `json:"code,omitempty"`
	SHA256       map[string]string `json:"sha256"`
	Metadata     json.RawMessage   `json:"metadata,omitempty"`
	Observations []Observation     `json:"observations,omitempty"`
}

type embeddedCaptureV1 struct {
	Manifest Manifest          `json:"manifest"`
	Files    map[string][]byte `json:"files"`
}

type embeddedCaptureV2 struct {
	Manifest Manifest          `json:"manifest"`
	Files    map[string][]byte `json:"files"`
}

// validateFilename checks that name is a portable, clean, non-empty, non-reserved basename.
func validateFilename(name string) error {
	if name == "" {
		return errors.New("metalcapture: empty filename")
	}
	if filepath.Base(name) != name || strings.ContainsAny(name, "/\\:") {
		return fmt.Errorf("metalcapture: filename %q is not a flat relative name", name)
	}
	if name == "." || name == ".." {
		return fmt.Errorf("metalcapture: invalid filename %q", name)
	}
	lower := strings.ToLower(name)
	for _, res := range reservedNames {
		if lower == strings.ToLower(res) {
			return fmt.Errorf("metalcapture: filename %q is reserved", name)
		}
	}
	return nil
}

func validateCapture(c *Capture) error {
	if c == nil {
		return errors.New("metalcapture: nil capture")
	}
	if c.Profile == nil {
		return errors.New("metalcapture: nil profile")
	}
	if err := c.Profile.CheckValid(); err != nil {
		return fmt.Errorf("metalcapture: invalid profile: %w", err)
	}
	if c.Files == nil {
		c.Files = make(map[string][]byte)
	}

	for role, name := range map[string]string{
		"source":   c.Code.Source,
		"air":      c.Code.AIR,
		"metallib": c.Code.MetalLib,
		"assembly": c.Code.Assembly,
	} {
		if name == "" {
			continue
		}
		if _, ok := c.Files[name]; !ok {
			return fmt.Errorf("metalcapture: %s file %q not found in files", role, name)
		}
	}

	caseMap := make(map[string]string, len(c.Files))
	for name := range c.Files {
		if err := validateFilename(name); err != nil {
			return err
		}
		lower := strings.ToLower(name)
		if orig, exists := caseMap[lower]; exists && orig != name {
			return fmt.Errorf("metalcapture: case collision between %q and %q", orig, name)
		}
		caseMap[lower] = name
	}

	if len(c.Metadata) > 0 {
		var js json.RawMessage
		if err := json.Unmarshal(c.Metadata, &js); err != nil {
			return fmt.Errorf("metalcapture: invalid metadata JSON: %w", err)
		}
	}

	return nil
}

// Write serializes the capture as a single pprof profile with embedded files, observations, and metadata.
func Write(w io.Writer, c *Capture) error {
	if err := validateCapture(c); err != nil {
		return err
	}

	p := c.Profile.Copy()
	for _, comment := range p.Comments {
		if strings.HasPrefix(comment, "#metalprof:capture:") {
			return errors.New("metalcapture: profile already contains an embedded capture")
		}
	}

	// Always write v2 if observations are present, otherwise v2 by default for modern format
	manifest := Manifest{
		Version:      2,
		Code:         c.Code,
		SHA256:       make(map[string]string, len(c.Files)),
		Metadata:     c.Metadata,
		Observations: c.Observations,
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
		return fmt.Errorf("metalcapture: marshal embedded capture: %w", err)
	}

	p.Comments = append(p.Comments,
		"Metal source, intermediate representation, and runtime artifacts are embedded; use metalprof for inspection.",
		captureV2Prefix+string(data),
	)

	return p.Write(w)
}

// Read reads a capture from a reader containing a pprof profile with embedded artifacts (v1 or v2).
func Read(r io.Reader) (*Capture, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("metalcapture: read profile data: %w", err)
	}
	p, err := profile.ParseData(data)
	if err != nil {
		return nil, fmt.Errorf("metalcapture: parse profile: %w", err)
	}
	if err := p.CheckValid(); err != nil {
		return nil, fmt.Errorf("metalcapture: invalid profile: %w", err)
	}

	var v1Comment, v2Comment string
	for _, c := range p.Comments {
		if strings.HasPrefix(c, "#metalprof:capture:") {
			if strings.HasPrefix(c, captureV2Prefix) {
				if v2Comment != "" || v1Comment != "" {
					return nil, errors.New("metalcapture: multiple embedded captures found")
				}
				v2Comment = strings.TrimPrefix(c, captureV2Prefix)
			} else if strings.HasPrefix(c, captureV1Prefix) {
				if v2Comment != "" || v1Comment != "" {
					return nil, errors.New("metalcapture: multiple embedded captures found")
				}
				v1Comment = strings.TrimPrefix(c, captureV1Prefix)
			} else {
				return nil, errors.New("metalcapture: unsupported embedded capture version")
			}
		}
	}

	if v2Comment != "" {
		return decodeEmbeddedV2(p, []byte(v2Comment))
	}
	if v1Comment != "" {
		return decodeEmbeddedV1(p, []byte(v1Comment))
	}

	return nil, errors.New("metalcapture: no embedded capture found in profile")
}

func decodeEmbeddedV1(p *profile.Profile, data []byte) (*Capture, error) {
	var emb embeddedCaptureV1
	if err := json.Unmarshal(data, &emb); err != nil {
		return nil, fmt.Errorf("metalcapture: unmarshal v1 embedded: %w", err)
	}
	if emb.Manifest.Version != 1 {
		return nil, fmt.Errorf("metalcapture: unsupported manifest version %d", emb.Manifest.Version)
	}

	if len(emb.Files) != len(emb.Manifest.SHA256) {
		return nil, errors.New("metalcapture: file count mismatch with manifest")
	}
	for name, fileBytes := range emb.Files {
		wantSum, ok := emb.Manifest.SHA256[name]
		if !ok {
			return nil, fmt.Errorf("metalcapture: file %q not in manifest checksums", name)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(fileBytes))
		if gotSum != wantSum {
			return nil, fmt.Errorf("metalcapture: checksum mismatch for %s", name)
		}
	}

	cap := &Capture{
		Profile:      p,
		Files:        emb.Files,
		Code:         emb.Manifest.Code,
		Metadata:     emb.Manifest.Metadata,
		Observations: emb.Manifest.Observations,
	}
	if err := validateCapture(cap); err != nil {
		return nil, err
	}
	return cap, nil
}

func decodeEmbeddedV2(p *profile.Profile, data []byte) (*Capture, error) {
	var emb embeddedCaptureV2
	if err := json.Unmarshal(data, &emb); err != nil {
		return nil, fmt.Errorf("metalcapture: unmarshal v2 embedded: %w", err)
	}
	if emb.Manifest.Version != 2 {
		return nil, fmt.Errorf("metalcapture: unsupported manifest version %d", emb.Manifest.Version)
	}

	if len(emb.Files) != len(emb.Manifest.SHA256) {
		return nil, errors.New("metalcapture: file count mismatch with manifest")
	}
	for name, fileBytes := range emb.Files {
		wantSum, ok := emb.Manifest.SHA256[name]
		if !ok {
			return nil, fmt.Errorf("metalcapture: file %q not in manifest checksums", name)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(fileBytes))
		if gotSum != wantSum {
			return nil, fmt.Errorf("metalcapture: checksum mismatch for %s", name)
		}
	}

	cap := &Capture{
		Profile:      p,
		Files:        emb.Files,
		Code:         emb.Manifest.Code,
		Metadata:     emb.Manifest.Metadata,
		Observations: emb.Manifest.Observations,
	}
	if err := validateCapture(cap); err != nil {
		return nil, err
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

	// Write gpu.pprof without embedded capture comment to prevent recursion
	p := c.Profile.Copy()
	var cleanComments []string
	for _, comm := range p.Comments {
		if !strings.HasPrefix(comm, "#metalprof:capture:") {
			cleanComments = append(cleanComments, comm)
		}
	}
	p.Comments = cleanComments

	pprofPath := filepath.Join(dir, "gpu.pprof")
	pf, err := os.OpenFile(pprofPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("metalcapture: create gpu.pprof: %w", err)
	}
	defer pf.Close()
	if err := p.Write(pf); err != nil {
		return fmt.Errorf("metalcapture: write gpu.pprof: %w", err)
	}

	// Write attached files
	manifest := Manifest{
		Version:      2,
		Code:         c.Code,
		SHA256:       make(map[string]string, len(c.Files)),
		Metadata:     c.Metadata,
		Observations: c.Observations,
	}
	for name, fileBytes := range c.Files {
		filePath := filepath.Join(dir, name)
		if err := os.WriteFile(filePath, fileBytes, 0644); err != nil {
			return fmt.Errorf("metalcapture: write file %s: %w", name, err)
		}
		manifest.SHA256[name] = fmt.Sprintf("%x", sha256.Sum256(fileBytes))
	}

	// Write manifest.json
	manifestBytes, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("metalcapture: marshal manifest: %w", err)
	}
	manifestPath := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(manifestPath, manifestBytes, 0644); err != nil {
		return fmt.Errorf("metalcapture: write manifest.json: %w", err)
	}

	return nil
}

// ReadDir reconstructs a Capture from dir containing manifest.json and gpu.pprof.
func ReadDir(dir string) (*Capture, error) {
	manifestPath := filepath.Join(dir, "manifest.json")
	manifestData, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("metalcapture: read manifest.json: %w", err)
	}

	var m Manifest
	if err := json.Unmarshal(manifestData, &m); err != nil {
		return nil, fmt.Errorf("metalcapture: unmarshal manifest: %w", err)
	}
	if m.Version != 1 && m.Version != 2 {
		return nil, fmt.Errorf("metalcapture: unsupported manifest version %d", m.Version)
	}

	pprofPath := filepath.Join(dir, "gpu.pprof")
	pprofData, err := os.ReadFile(pprofPath)
	if err != nil {
		return nil, fmt.Errorf("metalcapture: read gpu.pprof: %w", err)
	}
	p, err := profile.ParseData(pprofData)
	if err != nil {
		return nil, fmt.Errorf("metalcapture: parse gpu.pprof: %w", err)
	}

	files := make(map[string][]byte, len(m.SHA256))
	for name, wantSum := range m.SHA256 {
		filePath := filepath.Join(dir, name)
		fileData, err := os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("metalcapture: read attached file %s: %w", name, err)
		}
		gotSum := fmt.Sprintf("%x", sha256.Sum256(fileData))
		if gotSum != wantSum {
			return nil, fmt.Errorf("metalcapture: checksum mismatch for file %s", name)
		}
		files[name] = fileData
	}

	cap := &Capture{
		Profile:      p,
		Files:        files,
		Code:         m.Code,
		Metadata:     m.Metadata,
		Observations: m.Observations,
	}
	if err := validateCapture(cap); err != nil {
		return nil, err
	}
	return cap, nil
}
