package cudacache

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// NVRTCAdapter coordinates artifact lookup and compilation between cudacache and nvrtcx.
// It uses CompilerCacheKey to bind nvrtc.Version, driver SONAME, target architecture,
// compilation flags, and output kind magic bytes.
type NVRTCAdapter struct {
	cache *Cache
}

// NewNVRTCAdapter wraps c as an NVRTC compiler cache adapter.
func NewNVRTCAdapter(c *Cache) *NVRTCAdapter {
	return &NVRTCAdapter{cache: c}
}

// CurrentEnvironmentIdentity constructs a CompilerIdentity from currently loaded or default
// NVRTC and driver library information for the target architecture, flags, and output kind.
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

	return CompilerIdentity{
		NVRTCVersion: nvrtcVer,
		DriverSONAME: driverName,
		TargetArch:   targetArch,
		Flags:        append([]string(nil), flags...),
		OutputKind:   outputKind,
	}, nil
}

// Lookup attempts to find a previously compiled and cached artifact matching the input
// key and environment identity.
func (a *NVRTCAdapter) Lookup(ctx context.Context, inputKey string, id CompilerIdentity) (*cudakernel.Artifact, bool, error) {
	key := CompilerCacheKey(inputKey, id)
	art, err := a.cache.GetArtifact(ctx, key)
	if err == nil {
		return art, true, nil
	}
	if err == ErrNotFound {
		return nil, false, nil
	}
	return nil, false, err
}

// Store caches an artifact under the input key and environment identity.
func (a *NVRTCAdapter) Store(ctx context.Context, inputKey string, id CompilerIdentity, art *cudakernel.Artifact) error {
	key := CompilerCacheKey(inputKey, id)
	return a.cache.PutArtifact(ctx, key, art)
}
