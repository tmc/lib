package cudabpfdev

import (
	"bytes"
	"fmt"
	"runtime"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// Geometry describes an ordinary kernel launch. Cluster launches and other
// extended launch attributes are outside this source contract.
type Geometry struct {
	Grid, Block [3]uint32
	SharedBytes uint32
}

// Launch describes one kernel before instrumentation. Artifact contains its
// bytes, parameter ABI, target and digest; the digest and entry point together
// identify the kernel. Arguments and stream ownership remain with the caller.
type Launch struct {
	Artifact *cudakernel.Artifact
	Geometry Geometry
}

// LaunchSource supplies a caller-owned snapshot of a kernel and its geometry.
// Implementations must not infer artifact identity from a symbol name alone.
// A source does not authorize instrumentation: injection requires a separate
// opt-in policy and an exact kernel allowlist.
type LaunchSource interface {
	DescribeLaunch() (Launch, error)
}

// ArtifactSource supplies a directly loaded cudakernel artifact, without an
// interposition shim. Its zero value is invalid. Construct it with
// NewArtifactSource; descriptions do not share mutable storage with the source.
type ArtifactSource struct {
	launch Launch
}

var _ LaunchSource = (*ArtifactSource)(nil)

// NewArtifactSource copies and validates artifact and geometry. This is a host
// check, not proof that the driver accepts the image or launch configuration.
// Only PTX and CUBIN can be loaded through this source; LTOIR must first be linked.
func NewArtifactSource(artifact *cudakernel.Artifact, geometry Geometry) (*ArtifactSource, error) {
	if artifact == nil {
		return nil, fmt.Errorf("cudabpfdev: nil source artifact")
	}
	a := cloneArtifact(artifact)
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("cudabpfdev: source artifact: %w", err)
	}
	if a.Kind != cudakernel.KindPTX && a.Kind != cudakernel.KindCUBIN {
		return nil, fmt.Errorf("cudabpfdev: source artifact kind %s is not directly loadable", a.Kind)
	}
	if strings.ContainsRune(a.EntryPoint, 0) {
		return nil, fmt.Errorf("cudabpfdev: source entry point contains NUL")
	}
	if a.Kind == cudakernel.KindPTX && bytes.ContainsRune(bytes.TrimRight(a.Raw, "\x00"), 0) {
		return nil, fmt.Errorf("cudabpfdev: source PTX contains interior NUL")
	}
	for i := range geometry.Grid {
		if geometry.Grid[i] == 0 || geometry.Block[i] == 0 {
			return nil, fmt.Errorf("cudabpfdev: zero launch dimension")
		}
	}
	return &ArtifactSource{launch: Launch{Artifact: a, Geometry: geometry}}, nil
}

// DescribeLaunch returns a detached copy of the original launch description.
func (s *ArtifactSource) DescribeLaunch() (Launch, error) {
	if s == nil || s.launch.Artifact == nil {
		return Launch{}, fmt.Errorf("cudabpfdev: uninitialized artifact source")
	}
	l := s.launch
	l.Artifact = cloneArtifact(l.Artifact)
	return l, nil
}

func cloneArtifact(a *cudakernel.Artifact) *cudakernel.Artifact {
	c := *a
	c.Raw = slices.Clone(a.Raw)
	c.Schema.Params = slices.Clone(a.Schema.Params)
	c.CompilerOpts = slices.Clone(a.CompilerOpts)
	c.Dependencies = slices.Clone(a.Dependencies)
	return &c
}

// Load loads the original image unchanged, apart from the PTX NUL terminator
// required by the driver. It performs no instrumentation or launch. The driver
// checks device compatibility. Keep dev open throughout this call and until
// the module is unloaded or successfully transferred to cuda.OwnModule.
//
// A nonzero module belongs to the caller even when an error is returned, for
// example when restoring the previous context fails after a successful load.
// Unload it with dev's context current. Never discard a nonzero error result.
func (s *ArtifactSource) Load(dev *cuda.Device) (driver.Module, error) {
	if dev == nil {
		return 0, fmt.Errorf("cudabpfdev: nil device")
	}
	return s.load(dev.WithContext, loadImage)
}

func loadImage(image []byte, jit cudakernel.JITOptions) (driver.Module, error) {
	opts, vals := jit.DriverOptions()
	var mod driver.Module
	var err error
	if len(opts) != 0 {
		mod, err = driver.ModuleLoadDataEx(image, uint32(len(opts)), &opts[0], &vals[0])
	} else {
		mod, err = driver.ModuleLoadData(image)
	}
	runtime.KeepAlive(image)
	return mod, err

}

func (s *ArtifactSource) load(run func(func() error) error, load func([]byte, cudakernel.JITOptions) (driver.Module, error)) (driver.Module, error) {
	l, err := s.DescribeLaunch()
	if err != nil {
		return 0, err
	}
	a := l.Artifact
	image := a.Raw
	jit := cudakernel.JITOptions{}
	if a.Kind == cudakernel.KindPTX {
		image = append(bytes.TrimRight(image, "\x00"), 0)
		jit = a.JITOptions
	}
	var mod driver.Module
	err = run(func() error {
		var err error
		mod, err = load(image, jit)
		return err
	})
	if err != nil {
		return mod, fmt.Errorf("cudabpfdev: load original artifact: %w", err)
	}
	return mod, nil
}
