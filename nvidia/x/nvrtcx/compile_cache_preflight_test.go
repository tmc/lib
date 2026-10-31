package nvrtcx

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

const cachePreflightSource = `extern "C" __global__ void cache_key_fixture() {}`

// These are asserted cache metadata, not a loaded DSO or compiled program.
// A miss must stop before creating a native program on every fixture path.
func cachePreflightFixture(t *testing.T) (*compileCachePreflight, *int) {
	t.Helper()
	calls := new(int)
	p := &compileCachePreflight{major: 13, path: "/not-loaded/libnvrtc-fixture.so", identity: "sha256=" + strings.Repeat("a", 64), miss: errors.New("cache preflight miss")}
	path, identity := p.path, p.identity
	p.verify = func(gotPath, gotIdentity string) error {
		*calls++
		if gotPath != path || gotIdentity != identity {
			t.Errorf("cache-hit verification = (%q, %q), want (%q, %q)", gotPath, gotIdentity, path, identity)
		}
		return nil
	}
	return p, calls
}

func cachePreflightOptions(p *compileCachePreflight) []Option {
	return []Option{WithName("cache_key_fixture.cu"), WithOutput(OutputPTX), WithNoLineInfo(), WithNoToolkitIncludes(), func(o *options) { o.cachePreflight = p }}
}

func cachePreflightInputKey(flags []string) string {
	key := computeCacheKey(cachePreflightSource, "cache_key_fixture.cu", flags, nil)
	key = jitCacheKey(key, cudakernel.JITOptions{})
	return computeCacheKey(key, "name-expressions-v1", nil, nil)
}

func seedCachePreflight(t *testing.T, p *compileCachePreflight, disk bool, flags ...string) {
	t.Helper()
	useTestCache(t)
	input := cachePreflightInputKey(flags)
	identity := fmt.Sprintf("nvrtc %d.%d %s %s output=%d cache=%q", p.major, p.minor, p.path, p.identity, OutputPTX, cacheMagic)
	key := compilerCacheKey(input, identity)
	// These constructed payloads pass the ordinary cache codec. They are not
	// claimed to have been produced by NVRTC or to be executable on a device.
	for k, payload := range map[string]string{key: "fingerprint-A-cache-entry", input: "poisoned-input-only-cache-entry"} {
		module := &CompiledModule{key: k, output: OutputPTX, ptx: payload, compilerOptions: append([]string(nil), flags...), compilerOptionsKnown: true}
		if err := storeCache(k, module); err != nil {
			t.Fatal(err)
		}
	}
	if disk {
		memCacheMu.Lock()
		memCache = make(map[string]*CompiledModule)
		memCacheMu.Unlock()
	}
}

func TestCompileCachePreflightHit(t *testing.T) {
	for _, disk := range []bool{false, true} {
		t.Run(fmt.Sprint("disk=", disk), func(t *testing.T) {
			p, calls := cachePreflightFixture(t)
			seedCachePreflight(t, p, disk)
			module, err := Compile(cachePreflightSource, cachePreflightOptions(p)...)
			if err != nil || module == nil || module.PTX() != "fingerprint-A-cache-entry" || *calls != 1 {
				t.Fatalf("same-identity Compile cache hit: payload=%q error=%v verifier=%d", module.PTX(), err, *calls)
			}
		})
	}
}

func TestCompileCachePreflightChangedIdentity(t *testing.T) {
	for _, field := range []string{"identity", "version", "path"} {
		t.Run(field, func(t *testing.T) {
			for _, disk := range []bool{false, true} {
				t.Run(fmt.Sprint("disk=", disk), func(t *testing.T) {
					p, calls := cachePreflightFixture(t)
					seedCachePreflight(t, p, disk)
					beforePath, beforeIdentity := p.path, p.identity
					switch field {
					case "identity":
						p.identity = "sha256=" + strings.Repeat("b", 64)
					case "version":
						p.minor++
					case "path":
						p.path = "/not-loaded/other-libnvrtc-fixture.so"
					}
					// If an unqualified key selects the poisoned cache entry,
					// verification succeeds. The origin assertion below, rather
					// than a verifier mismatch, must expose the wrong key.
					p.verify = func(path, identity string) error {
						*calls++
						if path != p.path || identity != p.identity {
							t.Errorf("changed metadata verifier received (%q,%q), want (%q,%q)", path, identity, p.path, p.identity)
						}
						return nil
					}
					module, err := Compile(cachePreflightSource, cachePreflightOptions(p)...)
					if module != nil || !errors.Is(err, p.miss) {
						t.Fatalf("changed compiler %s reused a cache entry: payload=%q error=%v; old=(%q,%q)", field, module.PTX(), err, beforePath, beforeIdentity)
					}
					if *calls != 0 {
						t.Fatal("cache miss unexpectedly verified a hit")
					}
				})
			}
		})
	}
}

func TestCompileCachePreflightMissFence(t *testing.T) {
	for _, mode := range []string{"empty_cache", "no_cache"} {
		t.Run(mode, func(t *testing.T) {
			p, calls := cachePreflightFixture(t)
			useTestCache(t)
			opts := cachePreflightOptions(p)
			if mode == "no_cache" {
				seedCachePreflight(t, p, false, "--no-cache")
				opts = append(opts, WithNoCache())
			}
			module, err := Compile(cachePreflightSource, opts...)
			if module != nil || !errors.Is(err, p.miss) || *calls != 0 {
				t.Fatalf("cache-only miss reached native path: payload=%q error=%v verifier=%d", module.PTX(), err, *calls)
			}
		})
	}
}

func TestCompileCachePreflightVerifierRefusal(t *testing.T) {
	p, calls := cachePreflightFixture(t)
	seedCachePreflight(t, p, false)
	want := errors.New("cache metadata verifier refused")
	p.verify = func(string, string) error { *calls++; return want }
	module, err := Compile(cachePreflightSource, cachePreflightOptions(p)...)
	if module != nil || !errors.Is(err, want) || *calls != 1 {
		t.Fatalf("cache verifier refusal: payload=%q error=%v calls=%d", module.PTX(), err, *calls)
	}
}

func TestCompileCachePreflightMalformed(t *testing.T) {
	for _, field := range []string{"major", "minor", "path", "identity", "verifier", "miss"} {
		t.Run(field, func(t *testing.T) {
			p, calls := cachePreflightFixture(t)
			switch field {
			case "major":
				p.major = 0
			case "minor":
				p.minor = -1
			case "path":
				p.path = ""
			case "identity":
				p.identity = ""
			case "verifier":
				p.verify = nil
			case "miss":
				p.miss = nil
			}
			module, err := Compile(cachePreflightSource, cachePreflightOptions(p)...)
			if module != nil || err == nil || err.Error() != "nvrtcx: incomplete cache preflight" || *calls != 0 {
				t.Fatalf("malformed fixture %s: payload=%q error=%v verifier=%d", field, module.PTX(), err, *calls)
			}
		})
	}
}

func TestCompileCachePreflightCanceled(t *testing.T) {
	for _, fixture := range []bool{false, true} {
		t.Run(fmt.Sprint("fixture=", fixture), func(t *testing.T) {
			p, calls := cachePreflightFixture(t)
			if !fixture {
				p = nil
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			opts := append(cachePreflightOptions(p), WithContext(ctx))
			module, err := Compile(cachePreflightSource, opts...)
			if module != nil || !errors.Is(err, context.Canceled) || *calls != 0 {
				t.Fatalf("canceled preflight: payload=%q error=%v verifier=%d", module.PTX(), err, *calls)
			}
		})
	}
}
