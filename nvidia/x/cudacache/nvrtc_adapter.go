package cudacache

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// NVRTCAdapter coordinates artifact lookup and compilation between cudacache and nvrtcx.
// It uses CompilerCacheKey to bind nvrtc.Version, driver SONAME, target architecture,
// compilation flags, and output kind magic bytes.
//
// Each entry is published as an artifact plus a [Manifest] recording the
// identity and toolchain behind it. Lookup trusts an artifact only through a
// verified manifest.
type NVRTCAdapter struct {
	cache *Cache
}

// NewNVRTCAdapter wraps c as an NVRTC compiler cache adapter.
func NewNVRTCAdapter(c *Cache) *NVRTCAdapter {
	return &NVRTCAdapter{cache: c}
}

// CurrentEnvironmentIdentity constructs a CompilerIdentity from currently loaded or default
// NVRTC and driver library information for the target architecture, flags, and output kind.
//
// Provenance is set from [ProbeToolchain], which loads nothing: load NVRTC
// and the driver first, or the identity has no tool digests and a different
// key from one taken after loading. The error is non-nil only when a loaded
// library cannot be located or hashed.
func CurrentEnvironmentIdentity(targetArch string, flags []string, outputKind nvrtcx.OutputKind) (CompilerIdentity, error) {
	var nvrtcVer string
	if nvrtc.Loaded() {
		major, minor, err := nvrtc.Version()
		if err == nil {
			nvrtcVer = fmt.Sprintf("%d.%d", major, minor)
		}
	}
	if nvrtcVer == "" {
		// Fallback to path or default version tag if not yet dynamically loaded
		p := nvrtc.Path()
		if p != "" {
			nvrtcVer = p
		} else {
			nvrtcVer = "nvrtc-default"
		}
	}

	var driverName string
	if driver.Loaded() {
		driverName = driver.Path()
	}
	if driverName == "" {
		driverName = "libcuda.so.1"
	}

	prov, err := ProbeToolchain()
	if err != nil {
		return CompilerIdentity{}, err
	}
	return CompilerIdentity{
		NVRTCVersion: nvrtcVer,
		DriverSONAME: driverName,
		TargetArch:   targetArch,
		Flags:        append([]string(nil), flags...),
		OutputKind:   outputKind,
		Provenance:   prov,
	}, nil
}

// Lookup attempts to find a previously compiled and cached artifact matching the input
// key and environment identity.
//
// Before reading the cache, Lookup checks that each tool in id.Provenance
// still holds the bytes it recorded, and reports [ErrToolIdentityChanged] if
// one does not. An artifact with no manifest is a miss. A manifest that
// fails its digest or its binding to the key, or whose artifact is missing
// or corrupt, is quarantined with its artifact and reported as
// [ErrCorruptArtifact].
//
// If id.Provenance names tools and the manifest recorded a different
// toolchain, the entry is evicted and Lookup reports a miss, leaving a
// marker naming the pair and the evicted manifest. If the marker names that
// pair for an earlier manifest, a rebuild has reproduced the mismatch, and
// Lookup keeps the entry and returns a [*ToolchainMismatchError]. A hit
// removes the marker.
func (a *NVRTCAdapter) Lookup(ctx context.Context, inputKey string, id CompilerIdentity) (*cudakernel.Artifact, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	for _, d := range id.Provenance.Tools {
		if err := VerifyTool(d); err != nil {
			return nil, false, fmt.Errorf("cudacache: lookup: %w", err)
		}
	}
	c := a.cache
	key := CompilerCacheKey(inputKey, id)
	data, err := readFileBounded(c.manifestPath(key), c.maxArtifactBytes)
	if os.IsNotExist(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("cudacache: read manifest: %w", err)
	}
	m, err := checkManifest(key, data)
	if err != nil {
		c.quarantineEntry(key, data, m.ArtifactKey)
		return nil, false, fmt.Errorf("%w: %v", ErrCorruptArtifact, err)
	}
	art, err := c.GetArtifact(ctx, m.ArtifactKey)
	switch {
	case errors.Is(err, ErrNotFound):
		c.quarantineEntry(key, data, m.ArtifactKey)
		return nil, false, fmt.Errorf("%w: manifest names missing artifact %s", ErrCorruptArtifact, m.ArtifactKey)
	case errors.Is(err, ErrCorruptArtifact):
		c.quarantineEntry(key, data, m.ArtifactKey)
		return nil, false, err
	case err != nil:
		return nil, false, err
	}
	if art.Digest != m.PayloadSHA256 {
		c.quarantineEntry(key, data, m.ArtifactKey)
		return nil, false, fmt.Errorf("%w: artifact digest %s, manifest records %s", ErrCorruptArtifact, art.Digest, m.PayloadSHA256)
	}
	if err := checkCompilerArtifact(id, art); err != nil {
		c.quarantineEntry(key, data, m.ArtifactKey)
		return nil, false, fmt.Errorf("%w: %v", ErrCorruptArtifact, err)
	}
	if len(id.Provenance.Tools) > 0 && m.Recorded.Fingerprint() != id.Provenance.Fingerprint() {
		heal, err := c.healToolchain(key, id.Provenance, m.Recorded, m.Nonce)
		if err != nil {
			return nil, false, err
		}
		if heal {
			if err := c.evictEntry(key, data, m.ArtifactKey); err != nil {
				return nil, false, fmt.Errorf("cudacache: %w", err)
			}
		}
		return nil, false, nil
	}
	if err := c.clearToolchainMarker(key); err != nil {
		return nil, false, fmt.Errorf("cudacache: %w", err)
	}
	return art, true, nil
}

// Store caches an artifact under the input key and environment identity.
//
// The manifest records id.Provenance as the toolchain that produced art.
// The artifact is written first and the manifest last, each atomically, so
// a concurrent Lookup sees either no entry or a complete one.
func (a *NVRTCAdapter) Store(ctx context.Context, inputKey string, id CompilerIdentity, art *cudakernel.Artifact) error {
	return a.store(ctx, inputKey, id, id.Provenance, art)
}

func (a *NVRTCAdapter) store(ctx context.Context, inputKey string, id CompilerIdentity, recorded ToolProvenance, art *cudakernel.Artifact) error {
	if art == nil {
		return errors.New("cudacache: nil artifact")
	}
	if err := checkCompilerArtifact(id, art); err != nil {
		return err
	}
	key := CompilerCacheKey(inputKey, id)
	artKey := key + "-" + art.Digest
	if err := a.cache.PutArtifact(ctx, artKey, art); err != nil {
		return err
	}
	m := Manifest{
		Version:       manifestVersion,
		Key:           key,
		InputKey:      inputKey,
		Identity:      id,
		Recorded:      recorded,
		ArtifactKey:   artKey,
		PayloadSHA256: art.Digest,
	}
	if err := a.cache.writeManifest(m); err != nil {
		return fmt.Errorf("cudacache: %w", err)
	}
	return nil
}

// checkCompilerArtifact checks the artifact metadata this API can bind. The
// input key is opaque; it does not supply an expected entry or source payload.
func checkCompilerArtifact(id CompilerIdentity, art *cudakernel.Artifact) error {
	if err := art.Validate(); err != nil {
		return fmt.Errorf("cudacache: artifact: %w", err)
	}
	if id.TargetArch == "" || id.TargetArch != art.Target {
		return errors.New("cudacache: artifact target differs from compiler identity")
	}
	valid := false
	switch id.OutputKind {
	case nvrtcx.OutputAuto:
		valid = art.Kind == cudakernel.KindPTX || art.Kind == cudakernel.KindCUBIN
	case nvrtcx.OutputPTX:
		valid = art.Kind == cudakernel.KindPTX
	case nvrtcx.OutputCUBIN:
		valid = art.Kind == cudakernel.KindCUBIN
	case nvrtcx.OutputLTOIR:
		valid = art.Kind == cudakernel.KindLTOIR
	case nvrtcx.OutputTileIR:
		valid = art.Kind == cudakernel.KindTileIR
	case nvrtcx.OutputPTXTileIR:
		valid = art.Kind == cudakernel.KindPTX || art.Kind == cudakernel.KindTileIR
	}
	if !valid {
		return errors.New("cudacache: artifact kind differs from compiler output or is unsupported")
	}
	return nil
}

// GetOrCompile returns the cached artifact for inputKey and id, or compiles
// and caches it. A corrupt entry counts as a miss.
//
// compile runs between two checks of id.Provenance.Tools (see
// [WithRevalidatedTools]) and reports the artifact with the toolchain that
// produced it, which the manifest records. If a tool changed while compile
// ran, GetOrCompile returns [ErrToolIdentityChanged] and caches nothing.
func (a *NVRTCAdapter) GetOrCompile(ctx context.Context, inputKey string, id CompilerIdentity, compile func(context.Context) (*cudakernel.Artifact, ToolProvenance, error)) (*cudakernel.Artifact, error) {
	art, ok, err := a.Lookup(ctx, inputKey, id)
	if err != nil && !errors.Is(err, ErrCorruptArtifact) {
		return nil, err
	}
	if ok {
		return art, nil
	}
	type result struct {
		art  *cudakernel.Artifact
		prov ToolProvenance
	}
	r, err := WithRevalidatedTools(id.Provenance.Tools, func() (result, error) {
		art, prov, err := compile(ctx)
		return result{art, prov}, err
	})
	if err != nil {
		return nil, fmt.Errorf("cudacache: compile: %w", err)
	}
	if err := a.store(ctx, inputKey, id, r.prov, r.art); err != nil {
		return nil, err
	}
	return r.art, nil
}
