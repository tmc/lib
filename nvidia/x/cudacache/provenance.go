package cudacache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
)

// RecipeVersion names the recipe NVRTCAdapter follows to compile, validate
// and publish an artifact. It is bumped whenever those semantics change, so
// that artifacts produced under an older recipe stop matching.
const RecipeVersion = "cudacache/nvrtc-recipe/v1"

const provenanceDomain = "cudacache-tool-provenance-v1"

// ToolProvenance is the exact toolchain behind an artifact: the recipe, the
// bytes of each tool library, the driver version and the headers.
//
// A zero DriverVersion means the driver was not probed. It is not a
// measurement of version 0.
type ToolProvenance struct {
	Recipe        string       `json:"recipe"`
	Tools         []ToolDigest `json:"tools,omitempty"`
	DriverVersion int          `json:"driver_version,omitempty"` // cuDriverGetVersion, e.g. 13000
	HeadersSHA256 string       `json:"headers_sha256,omitempty"` // caller-defined digest of header inputs
}

// Fingerprint returns a domain-separated SHA-256, in hex, over Recipe, the
// (Name, SHA256) of each tool in name order, DriverVersion and HeadersSHA256.
// File identities are excluded: the same bytes at another path, inode or
// modification time have the same fingerprint.
func (p ToolProvenance) Fingerprint() string {
	h := sha256.New()
	field := func(s string) {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(s)))
		h.Write(size[:])
		h.Write([]byte(s))
	}
	field(provenanceDomain)
	field("recipe")
	field(p.Recipe)
	tools := sortedTools(p.Tools)
	field("tools")
	field(strconv.Itoa(len(tools)))
	for _, t := range tools {
		field(t.Name)
		field(t.SHA256)
	}
	field("driver_version")
	field(strconv.Itoa(p.DriverVersion))
	field("headers_sha256")
	field(p.HeadersSHA256)
	return hex.EncodeToString(h.Sum(nil))
}

func sortedTools(tools []ToolDigest) []ToolDigest {
	s := slices.Clone(tools)
	slices.SortFunc(s, func(a, b ToolDigest) int {
		if c := strings.Compare(a.Name, b.Name); c != 0 {
			return c
		}
		return strings.Compare(a.SHA256, b.SHA256)
	})
	return s
}

// Diff describes, one entry per field, how p differs from q in the fields
// Fingerprint covers. It returns nil when their fingerprints are equal.
func (p ToolProvenance) Diff(q ToolProvenance) []string {
	var d []string
	if p.Recipe != q.Recipe {
		d = append(d, fmt.Sprintf("recipe %q != %q", p.Recipe, q.Recipe))
	}
	pt, qt := toolsByName(p.Tools), toolsByName(q.Tools)
	var names []string
	for name := range pt {
		names = append(names, name)
	}
	for name := range qt {
		if _, ok := pt[name]; !ok {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	for _, name := range names {
		a, b := strings.Join(pt[name], ","), strings.Join(qt[name], ",")
		switch {
		case a == b:
		case a == "":
			d = append(d, fmt.Sprintf("tool %s absent != sha256 %s", name, b))
		case b == "":
			d = append(d, fmt.Sprintf("tool %s sha256 %s != absent", name, a))
		default:
			d = append(d, fmt.Sprintf("tool %s sha256 %s != %s", name, a, b))
		}
	}
	if p.DriverVersion != q.DriverVersion {
		d = append(d, fmt.Sprintf("driver_version %d != %d", p.DriverVersion, q.DriverVersion))
	}
	if p.HeadersSHA256 != q.HeadersSHA256 {
		d = append(d, fmt.Sprintf("headers_sha256 %q != %q", p.HeadersSHA256, q.HeadersSHA256))
	}
	return d
}

func toolsByName(tools []ToolDigest) map[string][]string {
	m := make(map[string][]string)
	for _, t := range sortedTools(tools) {
		m[t.Name] = append(m[t.Name], t.SHA256)
	}
	return m
}

// ProbeToolchain reports the provenance of the NVRTC and driver libraries
// this process has already loaded, with Recipe set to RecipeVersion. It
// loads neither library: a library that is not loaded is absent from Tools,
// and DriverVersion stays 0 unless the driver is loaded.
//
// A loaded library that cannot be located (see LoadedLibraryPath) or hashed
// is an error. Each probe rehashes every loaded library.
func ProbeToolchain() (ToolProvenance, error) {
	p := ToolProvenance{Recipe: RecipeVersion}
	if nvrtc.Loaded() {
		d, err := probeLoadedTool("nvrtc", nvrtc.Path(), toolFileIdentity)
		if err != nil {
			return ToolProvenance{}, fmt.Errorf("probe toolchain: %w", err)
		}
		p.Tools = append(p.Tools, d)
	}
	if driver.Loaded() {
		d, err := probeLoadedTool("driver", driver.Path(), toolFileIdentity)
		if err != nil {
			return ToolProvenance{}, fmt.Errorf("probe toolchain: %w", err)
		}
		p.Tools = append(p.Tools, d)
		v, err := driver.DriverGetVersion()
		if err != nil {
			return ToolProvenance{}, fmt.Errorf("probe toolchain: driver version: %w", err)
		}
		p.DriverVersion = v
	}
	return p, nil
}

func probeLoadedTool(name, opened string, identity identityFunc) (ToolDigest, error) {
	path, err := LoadedLibraryPath(opened)
	if err != nil {
		return ToolDigest{}, err
	}
	return digestTool(name, path, identity)
}
