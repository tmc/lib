package cudaprecompiled

import (
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"slices"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/internal/precompile"
)

// A Module is CUDA C++ device code with an optional precompiled image.
// The fields must not change after the first call to Load.
type Module struct {
	// Image is the precompiled code, or nil.
	Image []byte
	// Sum identifies the sources and options Image was built from.
	Sum string

	// Files holds the sources and headers.
	Files fs.FS
	// Sources are the translation units, as paths in Files.
	Sources []string
	// Headers are the headers the sources may include, as paths in Files.
	Headers []string
	// Include are include directories, as paths in Files.
	Include []string
	// Options are other NVRTC options, such as -DN=4 or -std=c++17.
	Options []string
	// NoLTO links several sources without link-time optimization.
	NoLTO bool
	// Names maps name expressions, such as "scale<float>", to the lowered
	// names in Image.
	Names map[string]string

	mu       sync.Mutex
	compiled map[string]string // lowered names from the last compile
}

func (m *Module) config() *precompile.Config {
	names := slices.Sorted(maps.Keys(m.Names))
	return &precompile.Config{
		Files:   m.Files,
		Sources: m.Sources,
		Headers: m.Headers,
		Include: m.Include,
		Options: m.Options,
		Names:   names,
		NoLTO:   m.NoLTO,
	}
}

// Stale reports whether Image was built from different sources or options
// than the ones in m. A Module without an image is not stale.
func (m *Module) Stale() (bool, error) {
	if m.Image == nil {
		return false, nil
	}
	sum, err := m.config().Sum()
	if err != nil {
		return false, fmt.Errorf("cudaprecompiled: %w", err)
	}
	return sum != m.Sum, nil
}

// Load loads the module on dev. It uses Image if Image is current and has
// code for dev; otherwise it compiles the sources for dev.
func (m *Module) Load(dev *cuda.Device) (*cuda.Module, error) {
	stale, err := m.Stale()
	if err != nil {
		return nil, err
	}
	if m.Image != nil && !stale {
		mod, err := cuda.LoadModule(dev, m.Image)
		if err == nil || !unusable(err) {
			return mod, err
		}
	}
	return m.compile(dev)
}

// unusable reports whether err means the image has no code this device and
// driver can run, so compiling from source may still succeed.
func unusable(err error) bool {
	return errors.Is(err, driver.ErrNoBinaryForGpu) ||
		errors.Is(err, driver.ErrUnsupportedPtxVersion) ||
		errors.Is(err, driver.ErrInvalidPtx)
}

func (m *Module) compile(dev *cuda.Device) (*cuda.Module, error) {
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("cudaprecompiled: %w", err)
	}
	code, names, err := m.config().Native(fmt.Sprintf("sm_%d%d", major, minor))
	if err != nil {
		return nil, fmt.Errorf("cudaprecompiled: %w", err)
	}
	mod, err := cuda.LoadModule(dev, code)
	if err != nil {
		return nil, fmt.Errorf("cudaprecompiled: %w", err)
	}
	m.mu.Lock()
	m.compiled = names
	m.mu.Unlock()
	return mod, nil
}

// LoweredName returns the lowered name of a name expression, for use with
// [cuda.Module.Function]. After Load compiles the sources, it returns the
// name from that compile.
func (m *Module) LoweredName(expr string) (string, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.compiled != nil {
		name, ok := m.compiled[expr]
		return name, ok
	}
	name, ok := m.Names[expr]
	return name, ok
}
