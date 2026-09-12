package profilebundle

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/pprof/profile"
)

// A comment beginning with # is retained by pprof but omitted from report
// headings. Keep the payload in a standard field so pprof can round-trip it.
const capturePrefix = "#cudaprof:capture:v1\n"

type embeddedCapture struct {
	Manifest Manifest          `json:"manifest"`
	Files    map[string][]byte `json:"files"`
}

// Pack writes the profile and its recorded files as one gzip-compressed pprof.
// The outer profile contains the samples; its metadata contains the other files.
func Pack(dir string, w io.Writer) error {
	m, err := Read(dir)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(filepath.Join(dir, "gpu.pprof"))
	if err != nil {
		return err
	}
	p, err := profile.ParseData(data)
	if err != nil {
		return err
	}
	if err := p.CheckValid(); err != nil {
		return err
	}
	for _, c := range p.Comments {
		if strings.HasPrefix(c, "#cudaprof:capture:") {
			return fmt.Errorf("profile already contains a capture")
		}
	}
	b := embeddedCapture{Manifest: *m, Files: make(map[string][]byte)}
	// The enclosing profile cannot contain its own checksum.
	delete(b.Manifest.SHA256, "gpu.pprof")
	for _, name := range Files {
		if name == "gpu.pprof" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return err
		}
		b.Files[name] = data
	}
	data, err = json.Marshal(b)
	if err != nil {
		return err
	}
	p.Comments = append(p.Comments, "Source, PTX, SASS, CUBIN, and Go trace are embedded; use cudaprof for source and NVIDIA disassembly.", capturePrefix+string(data))
	return p.Write(w)
}

// Capture is an opened profile and its recorded source and assembly files.
// Close removes any temporary files created for an embedded capture.
type Capture struct {
	Profile  *profile.Profile
	Manifest *Manifest
	Dir      string
	temp     bool
}

func (c *Capture) Close() error {
	if c.temp {
		return os.RemoveAll(c.Dir)
	}
	return nil
}

// Open reads a self-contained profile, or a profile beside its recorded files.
// Upstream pprof's source reader requires files, so embedded files are expanded
// into a private directory under ~/tmp for the lifetime of the viewer.
func Open(path string) (*Capture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	p, err := profile.ParseData(data)
	if err != nil {
		return nil, err
	}
	if err := p.CheckValid(); err != nil {
		return nil, err
	}
	b, err := embedded(p)
	if err != nil {
		return nil, err
	}
	if b == nil {
		dir := filepath.Dir(path)
		m, err := Read(dir)
		if err != nil {
			return nil, err
		}
		return &Capture{Profile: p, Manifest: m, Dir: dir}, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	parent := filepath.Join(home, "tmp")
	if err := os.MkdirAll(parent, 0755); err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp(parent, "cudaprof-")
	if err != nil {
		return nil, err
	}
	for name, data := range b.Files {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			os.RemoveAll(dir)
			return nil, err
		}
	}
	return &Capture{Profile: p, Manifest: &b.Manifest, Dir: dir, temp: true}, nil
}

func embedded(p *profile.Profile) (*embeddedCapture, error) {
	var b *embeddedCapture
	for _, c := range p.Comments {
		if !strings.HasPrefix(c, "#cudaprof:capture:") {
			continue
		}
		if b != nil {
			return nil, fmt.Errorf("multiple embedded captures")
		}
		if !strings.HasPrefix(c, capturePrefix) {
			return nil, fmt.Errorf("unsupported embedded capture version")
		}
		b = new(embeddedCapture)
		if err := json.Unmarshal([]byte(strings.TrimPrefix(c, capturePrefix)), b); err != nil {
			return nil, fmt.Errorf("read embedded capture: %w", err)
		}
	}
	if b == nil {
		return nil, nil
	}
	if b.Manifest.Version != 1 {
		return nil, fmt.Errorf("unsupported bundle version %d", b.Manifest.Version)
	}
	// Only this fixed set of filenames may be materialized. No archive paths or
	// additional manifest entries are used to choose filesystem destinations.
	if len(b.Files) != len(Files)-1 || len(b.Manifest.SHA256) != len(Files)-1 {
		return nil, fmt.Errorf("unexpected embedded file set")
	}
	for _, name := range Files {
		if name == "gpu.pprof" {
			continue
		}
		data, ok := b.Files[name]
		if !ok {
			return nil, fmt.Errorf("missing embedded file %s", name)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != b.Manifest.SHA256[name] {
			return nil, fmt.Errorf("%s: sha256 mismatch", name)
		}
	}
	return b, nil
}
