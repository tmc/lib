package cudacache_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// writeTool writes a stand-in tool library and returns its path.
func writeTool(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func digestTool(t *testing.T, name, path string) cudacache.ToolDigest {
	t.Helper()
	d, err := cudacache.DigestTool(name, path)
	if err != nil {
		t.Fatalf("DigestTool(%s): %v", name, err)
	}
	return d
}

// toolchainFixture is a cache with two stand-in tool files and an identity
// whose provenance names them.
type toolchainFixture struct {
	dir     string
	cache   *cudacache.Cache
	adapter *cudacache.NVRTCAdapter
	nvrtc   string // path of the stand-in NVRTC library
	driver  string // path of the stand-in driver library
	id      cudacache.CompilerIdentity
}

func newToolchainFixture(t *testing.T) *toolchainFixture {
	t.Helper()
	f := &toolchainFixture{dir: t.TempDir()}
	root := filepath.Join(f.dir, "cache")
	c, err := cudacache.New(cudacache.Config{RootDir: root})
	if err != nil {
		t.Fatal(err)
	}
	f.cache = c
	f.adapter = cudacache.NewNVRTCAdapter(c)
	f.nvrtc = writeTool(t, f.dir, "libnvrtc.so.13.0.88", "nvrtc bytes, build 1")
	f.driver = writeTool(t, f.dir, "libcuda.so.580.1", "driver bytes, build 1")
	f.id = cudacache.CompilerIdentity{
		NVRTCVersion: "13.0",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_121a",
		Flags:        []string{"--std=c++20"},
		OutputKind:   nvrtcx.OutputPTX,
		Provenance: cudacache.ToolProvenance{
			Recipe:        cudacache.RecipeVersion,
			Tools:         []cudacache.ToolDigest{digestTool(t, "nvrtc", f.nvrtc), digestTool(t, "driver", f.driver)},
			DriverVersion: 13000,
		},
	}
	return f
}

func (f *toolchainFixture) manifestPath() string {
	return filepath.Join(f.dir, "cache", "manifests", cudacache.CompilerCacheKey("input", f.id)+".manifest")
}

func (f *toolchainFixture) markerPath() string {
	return filepath.Join(f.dir, "cache", "artifacts", cudacache.CompilerCacheKey("input", f.id)+".toolchain-mismatch-marker")
}

func (f *toolchainFixture) quarantined(t *testing.T) int {
	t.Helper()
	ents, err := os.ReadDir(filepath.Join(f.dir, "cache", "quarantine"))
	if err != nil {
		t.Fatal(err)
	}
	return len(ents)
}

func readManifest(t *testing.T, path string) cudacache.Manifest {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var m cudacache.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func mustLookup(t *testing.T, a *cudacache.NVRTCAdapter, id cudacache.CompilerIdentity) (*cudakernel.Artifact, bool) {
	t.Helper()
	art, ok, err := a.Lookup(context.Background(), "input", id)
	if err != nil {
		t.Fatalf("Lookup: %v", err)
	}
	return art, ok
}

// rewriteInPlace overwrites path with content of the same length through
// the existing inode, as an in-place package upgrade can.
func rewriteInPlace(t *testing.T, path, content string) {
	t.Helper()
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if int64(len(content)) != fi.Size() {
		t.Fatalf("rewriteInPlace: %d bytes over %d", len(content), fi.Size())
	}
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte(content), 0); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

// replaceByRename renames a new file over path, as a package manager does.
func replaceByRename(t *testing.T, path, content string) {
	t.Helper()
	tmp := path + ".new"
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, path); err != nil {
		t.Fatal(err)
	}
}

func TestCompilerCacheKey_ToolDSOMismatchInvalidatesKey(t *testing.T) {
	sha := func(c byte) string { return strings.Repeat(string(c), 64) }
	base := cudacache.CompilerIdentity{
		NVRTCVersion: "13.0",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_121a",
		Flags:        []string{"--std=c++20"},
		OutputKind:   nvrtcx.OutputCUBIN,
		Provenance: cudacache.ToolProvenance{
			Recipe: cudacache.RecipeVersion,
			Tools: []cudacache.ToolDigest{
				{Name: "nvrtc", SHA256: sha('a'), File: cudacache.ToolFileIdentity{Path: "/usr/lib/libnvrtc.so.13", Size: 100, Inode: 7, Unix: true}},
				{Name: "driver", SHA256: sha('b'), File: cudacache.ToolFileIdentity{Path: "/usr/lib/libcuda.so.1", Size: 200, Inode: 8, Unix: true}},
			},
			DriverVersion: 13000,
			HeadersSHA256: sha('c'),
		},
	}
	with := func(edit func(p *cudacache.ToolProvenance)) cudacache.CompilerIdentity {
		id := base
		id.Provenance.Tools = append([]cudacache.ToolDigest(nil), base.Provenance.Tools...)
		edit(&id.Provenance)
		return id
	}
	baseKey := cudacache.CompilerCacheKey("input", base)

	changed := []struct {
		name string
		id   cudacache.CompilerIdentity
	}{
		{"nvrtc sha", with(func(p *cudacache.ToolProvenance) { p.Tools[0].SHA256 = sha('d') })},
		{"driver sha", with(func(p *cudacache.ToolProvenance) { p.Tools[1].SHA256 = sha('d') })},
		{"tool name", with(func(p *cudacache.ToolProvenance) { p.Tools[0].Name = "nvrtc-builtins" })},
		{"driver version", with(func(p *cudacache.ToolProvenance) { p.DriverVersion = 13010 })},
		{"headers", with(func(p *cudacache.ToolProvenance) { p.HeadersSHA256 = sha('e') })},
		{"recipe", with(func(p *cudacache.ToolProvenance) { p.Recipe = "cudacache/nvrtc-recipe/v0" })},
		{"tool added", with(func(p *cudacache.ToolProvenance) {
			p.Tools = append(p.Tools, cudacache.ToolDigest{Name: "nvjitlink", SHA256: sha('f')})
		})},
		{"tool removed", with(func(p *cudacache.ToolProvenance) { p.Tools = p.Tools[:1] })},
	}
	if len(changed) < 8 {
		t.Fatalf("%d provenance mutations, want at least 8", len(changed))
	}
	seen := map[string]string{baseKey: "base"}
	for _, tt := range changed {
		k := cudacache.CompilerCacheKey("input", tt.id)
		if prev, ok := seen[k]; ok {
			t.Errorf("%s: key %s equals the key for %s", tt.name, k, prev)
		}
		seen[k] = tt.name
	}

	// Adjacent: identity of the file, and the order tools are listed in, do
	// not name different bytes, so they must not change the key.
	same := []struct {
		name string
		id   cudacache.CompilerIdentity
	}{
		{"file path", with(func(p *cudacache.ToolProvenance) { p.Tools[0].File.Path = "/opt/cuda/lib64/libnvrtc.so.13" })},
		{"file mtime", with(func(p *cudacache.ToolProvenance) { p.Tools[0].File.ModTimeNanos = 12345 })},
		{"file inode", with(func(p *cudacache.ToolProvenance) { p.Tools[1].File.Inode = 99 })},
		{"tool order", with(func(p *cudacache.ToolProvenance) { p.Tools[0], p.Tools[1] = p.Tools[1], p.Tools[0] })},
	}
	for _, tt := range same {
		if k := cudacache.CompilerCacheKey("input", tt.id); k != baseKey {
			t.Errorf("%s: key changed to %s", tt.name, k)
		}
	}

	// A stored entry must not be returned to an identity whose tool digest
	// differs, even when the recorded file identity is unchanged.
	f := newToolchainFixture(t)
	art := newTestArtifact(t, "k", "sm_121a")
	if err := f.adapter.Store(context.Background(), "input", f.id, art); err != nil {
		t.Fatal(err)
	}
	if _, ok := mustLookup(t, f.adapter, f.id); !ok {
		t.Fatal("Lookup under the storing identity: miss, want hit")
	}
	stale := f.id
	stale.Provenance.Tools = append([]cudacache.ToolDigest(nil), f.id.Provenance.Tools...)
	stale.Provenance.Tools[0].SHA256 = sha('9')
	if got, ok, err := f.adapter.Lookup(context.Background(), "input", stale); ok || got != nil || !errors.Is(err, cudacache.ErrToolIdentityChanged) {
		t.Fatalf("Lookup with fabricated nvrtc sha256 = %v, %v, %v; want identity refusal", got, ok, err)
	}
	// A genuine update receives a newly measured identity and different key;
	// it is a clean miss, not a false-digest input error.
	rewriteInPlace(t, f.nvrtc, "nvrtc bytes, build 2")
	updated := f.id
	updated.Provenance.Tools = append([]cudacache.ToolDigest(nil), f.id.Provenance.Tools...)
	updated.Provenance.Tools[0] = digestTool(t, "nvrtc", f.nvrtc)
	if cudacache.CompilerCacheKey("input", updated) == cudacache.CompilerCacheKey("input", f.id) {
		t.Fatal("actual updated tool reused old key")
	}
	if got, ok := mustLookup(t, f.adapter, updated); ok || got != nil {
		t.Fatalf("updated tool lookup=%v,%t; want clean miss", got, ok)
	}
}

func TestCompilerCacheKeyV3Pinned(t *testing.T) {
	// The ExampleCompilerCacheKey identity, with a zero Provenance.
	id := cudacache.CompilerIdentity{
		NVRTCVersion: "12.8",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_121a",
		Flags:        []string{"--std=c++20", "--extra-device-vectorization"},
		OutputKind:   nvrtcx.OutputCUBIN,
	}
	// Computed outside Go from the recipe in the CompilerCacheKey source
	// (8-byte little-endian length before each field, domain
	// "cudacache-compiler-key-v3", provenance followed by JIT option count).
	const want = "1a171154b60a360be90264412debf44447b7d5fa566473a8fedc2dc7373659ba"
	if got := cudacache.CompilerCacheKey("kernel_source_hash_1234", id); got != want {
		t.Fatalf("CompilerCacheKey = %s, want %s; a key format change must bump the domain", got, want)
	}
	const wantFP = "3830eaf751b44c2a90fc9fa6aadf5c10340004934c97c94118e299a32d05c13e"
	if got := (cudacache.ToolProvenance{}).Fingerprint(); got != wantFP {
		t.Fatalf("zero Fingerprint = %s, want %s", got, wantFP)
	}
}

func TestDigestToolKnownVector(t *testing.T) {
	path := writeTool(t, t.TempDir(), "abc", "abc")
	d := digestTool(t, "abc", path)
	// FIPS 180-2, appendix B.1.
	const want = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if d.SHA256 != want {
		t.Fatalf("sha256(abc) = %s, want %s", d.SHA256, want)
	}
	if d.File.Path != path || d.File.Size != 3 {
		t.Fatalf("File = %+v, want path %s size 3", d.File, path)
	}
}

func TestToolFileIdentity_DetectsInPlaceDSOModification(t *testing.T) {
	dir := t.TempDir()
	path := writeTool(t, dir, "libnvrtc.so.13", "version one")
	d := digestTool(t, "nvrtc", path)
	if err := cudacache.VerifyTool(d); err != nil {
		t.Fatalf("VerifyTool on untouched file: %v", err)
	}

	// Adjacent: a touch changes the identity but not the bytes, so the
	// rehash must accept it.
	later := time.Unix(0, d.File.ModTimeNanos).Add(time.Hour)
	if err := os.Chtimes(path, later, later); err != nil {
		t.Fatal(err)
	}
	if err := cudacache.VerifyTool(d); err != nil {
		t.Fatalf("VerifyTool after touch: %v, want nil", err)
	}

	// Planted: same-length write through the same inode, with the
	// modification time put back, so only the change time and the bytes
	// record it.
	before, err := cudacache.StatTool(path)
	if err != nil {
		t.Fatal(err)
	}
	rewriteInPlace(t, path, "version two")
	mtime := time.Unix(0, before.ModTimeNanos)
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	after, err := cudacache.StatTool(path)
	if err != nil {
		t.Fatal(err)
	}
	if after == before {
		t.Fatalf("StatTool unchanged by in-place write: %+v", after)
	}
	if after.Size != before.Size || after.ModTimeNanos != before.ModTimeNanos {
		t.Fatalf("fixture: size or mtime differ (%+v vs %+v); the write must differ only in contents and ctime", after, before)
	}
	if after.Unix && after.Inode != before.Inode {
		t.Fatalf("fixture: inode changed %d -> %d; the write must be in place", before.Inode, after.Inode)
	}
	if err := cudacache.VerifyTool(d); !errors.Is(err, cudacache.ErrToolIdentityChanged) {
		t.Fatalf("VerifyTool after in-place write = %v, want ErrToolIdentityChanged", err)
	}

	// A file that is gone is changed, not an ordinary error.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := cudacache.VerifyTool(d); !errors.Is(err, cudacache.ErrToolIdentityChanged) {
		t.Fatalf("VerifyTool after remove = %v, want ErrToolIdentityChanged", err)
	}
	if err := cudacache.VerifyTool(cudacache.ToolDigest{Name: "empty"}); err == nil || errors.Is(err, cudacache.ErrToolIdentityChanged) {
		t.Fatalf("VerifyTool of empty digest = %v, want a plain error", err)
	}
}

// TestVerifyTool_DetectsSameSizeRewriteWithRestoredMtime rewrites a tool
// at the size and modification time its digest recorded, so only the
// device, inode and change time can tell VerifyTool to rehash.
func TestVerifyTool_DetectsSameSizeRewriteWithRestoredMtime(t *testing.T) {
	tests := []struct {
		name    string
		rewrite func(t *testing.T, path, content string)
	}{
		{"in place", rewriteInPlace},
		{"rename over path", replaceByRename},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTool(t, t.TempDir(), "libnvrtc.so.13", "nvrtc build 1")
			d := digestTool(t, "nvrtc", path)
			if !d.File.Unix {
				t.Fatalf("StatTool(%s) has no device, inode or change time on %s", path, runtime.GOOS)
			}
			tt.rewrite(t, path, "nvrtc build 2")
			mtime := time.Unix(0, d.File.ModTimeNanos)
			if err := os.Chtimes(path, mtime, mtime); err != nil {
				t.Fatal(err)
			}
			after, err := cudacache.StatTool(path)
			if err != nil {
				t.Fatal(err)
			}
			if after.Size != d.File.Size || after.ModTimeNanos != d.File.ModTimeNanos {
				t.Fatalf("fixture: size or mtime differ (%+v vs %+v); only dev, ino and ctime may", after, d.File)
			}
			if err := cudacache.VerifyTool(d); !errors.Is(err, cudacache.ErrToolIdentityChanged) {
				t.Fatalf("VerifyTool after %s rewrite at recorded size and mtime = %v, want ErrToolIdentityChanged", tt.name, err)
			}
		})
	}
}

func TestRevalidatedToolIdentity_AbortsOnConcurrentToolSwap(t *testing.T) {
	type result struct {
		value string
		err   error
		calls int
	}
	run := func(t *testing.T, before, during func(path string), opErr error) result {
		t.Helper()
		path := writeTool(t, t.TempDir(), "libnvrtc.so.13", "nvrtc build 1")
		tools := []cudacache.ToolDigest{digestTool(t, "nvrtc", path)}
		if before != nil {
			before(path)
		}
		var r result
		r.value, r.err = cudacache.WithRevalidatedTools(tools, func() (string, error) {
			r.calls++
			if during != nil {
				during(path)
			}
			return "artifact", opErr
		})
		return r
	}
	swap := func(path string) { replaceByRename(t, path, "nvrtc build 2") }
	sameBytes := func(path string) { replaceByRename(t, path, "nvrtc build 1") }
	opFailed := errors.New("compile failed")

	tests := []struct {
		name      string
		before    func(string)
		during    func(string)
		opErr     error
		wantValue string
		wantErr   error
		wantCalls int
	}{
		{name: "untouched", wantValue: "artifact", wantCalls: 1},
		{name: "swapped during op", during: swap, wantErr: cudacache.ErrToolIdentityChanged, wantCalls: 1},
		{name: "swapped before op", before: swap, wantErr: cudacache.ErrToolIdentityChanged, wantCalls: 0},
		{name: "same bytes renamed over", during: sameBytes, wantValue: "artifact", wantCalls: 1},
		{name: "op failed", opErr: opFailed, wantValue: "artifact", wantErr: opFailed, wantCalls: 1},
		{name: "op failed and swapped", during: swap, opErr: opFailed, wantErr: cudacache.ErrToolIdentityChanged, wantCalls: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := run(t, tt.before, tt.during, tt.opErr)
			if r.calls != tt.wantCalls {
				t.Errorf("op ran %d times, want %d", r.calls, tt.wantCalls)
			}
			if tt.wantErr == nil && r.err != nil || tt.wantErr != nil && !errors.Is(r.err, tt.wantErr) {
				t.Errorf("err = %v, want %v", r.err, tt.wantErr)
			}
			if r.value != tt.wantValue {
				t.Errorf("value = %q, want %q", r.value, tt.wantValue)
			}
		})
	}

	t.Run("GetOrCompile stores nothing", func(t *testing.T) {
		f := newToolchainFixture(t)
		art := newTestArtifact(t, "k", "sm_121a")
		_, err := f.adapter.GetOrCompile(context.Background(), "input", f.id, func(context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
			replaceByRename(t, f.nvrtc, "nvrtc bytes, build 2")
			return art, f.id.Provenance, nil
		})
		if !errors.Is(err, cudacache.ErrToolIdentityChanged) {
			t.Fatalf("GetOrCompile = %v, want ErrToolIdentityChanged", err)
		}
		if _, err := os.Stat(f.manifestPath()); !os.IsNotExist(err) {
			t.Fatalf("manifest after aborted compile: stat err %v, want not exist", err)
		}
	})
}

func TestToolchainFingerprint_DetectsDriverVersionShift(t *testing.T) {
	f := newToolchainFixture(t)
	old := f.id.Provenance
	shifted := old
	shifted.DriverVersion = 13010
	if old.Fingerprint() == shifted.Fingerprint() {
		t.Fatal("fingerprints equal across driver 13000 -> 13010")
	}
	diff := shifted.Diff(old)
	if len(diff) != 1 || diff[0] != "driver_version 13010 != 13000" {
		t.Fatalf("Diff = %q, want [driver_version 13010 != 13000]", diff)
	}
	if d := old.Diff(old); d != nil {
		t.Fatalf("Diff of equal provenance = %q, want nil", d)
	}

	// An entry the old driver produced, looked up under the new driver
	// with an unchanged key, is a toolchain mismatch: evicted, marked, miss.
	art := newTestArtifact(t, "k", "sm_121a")
	id := f.id
	id.Provenance = shifted
	_, err := f.adapter.GetOrCompile(context.Background(), "input", id, func(context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
		return art, old, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, err := f.adapter.Lookup(context.Background(), "input", id); ok || err != nil {
		t.Fatalf("Lookup of entry recorded under 13000 by 13010 = %v, %v; want miss", ok, err)
	}
	data, err := os.ReadFile(filepath.Join(f.dir, "cache", "artifacts", cudacache.CompilerCacheKey("input", id)+".toolchain-mismatch-marker"))
	if err != nil {
		t.Fatalf("marker: %v", err)
	}
	want := "active toolchain:\n" + shifted.Fingerprint() + "\n\nrecorded toolchain:\n" + old.Fingerprint() + "\n\nevicted manifest:\n"
	if !strings.HasPrefix(string(data), want) {
		t.Fatalf("marker = %q, want prefix %q", data, want)
	}
}

func TestUnhealableToolchainMismatch_WritesMarkerAndHaltsLoop(t *testing.T) {
	ctx := context.Background()
	art := newTestArtifact(t, "k", "sm_121a")

	// getOrCompile makes calls GetOrCompile calls whose compile reports the
	// provenance recorded(i) for its i'th compile, and returns the number
	// of compiles and each call's error.
	getOrCompile := func(t *testing.T, f *toolchainFixture, id cudacache.CompilerIdentity, calls int, recorded func(i int) cudacache.ToolProvenance) (int, []error) {
		t.Helper()
		compiles := 0
		errs := make([]error, calls)
		for i := range errs {
			_, errs[i] = f.adapter.GetOrCompile(ctx, "input", id, func(context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
				compiles++
				return art, recorded(compiles), nil
			})
		}
		return compiles, errs
	}

	t.Run("same mismatch halts", func(t *testing.T) {
		f := newToolchainFixture(t)
		active := f.id.Provenance
		recorded := active
		recorded.DriverVersion = 12090
		compiles, errs := getOrCompile(t, f, f.id, 6, func(int) cudacache.ToolProvenance { return recorded })
		// Call 1 misses and compiles; call 2 sees the mismatch for the first
		// time, marks it and recompiles; calls 3-6 see the same pair.
		if compiles != 2 {
			t.Errorf("compile ran %d times over 6 calls, want 2", compiles)
		}
		for i, err := range errs {
			if i < 2 {
				if err != nil {
					t.Errorf("call %d: %v, want nil", i+1, err)
				}
				continue
			}
			var me *cudacache.ToolchainMismatchError
			if !errors.Is(err, cudacache.ErrUnhealableToolchainMismatch) || !errors.As(err, &me) {
				t.Errorf("call %d: %v, want *ToolchainMismatchError", i+1, err)
				continue
			}
			if me.Marker != f.markerPath() || me.Active.Fingerprint() != active.Fingerprint() || me.Recorded.Fingerprint() != recorded.Fingerprint() {
				t.Errorf("call %d: error fields %+v", i+1, me)
			}
			if !strings.Contains(err.Error(), "driver_version 13000 != 12090") {
				t.Errorf("call %d: error %q does not name the difference", i+1, err)
			}
		}
		data, err := os.ReadFile(f.markerPath())
		if err != nil {
			t.Fatalf("marker: %v", err)
		}
		want := "active toolchain:\n" + active.Fingerprint() + "\n\nrecorded toolchain:\n" + recorded.Fingerprint() + "\n\nevicted manifest:\n"
		evicted, ok := strings.CutPrefix(string(data), want)
		if !ok {
			t.Fatalf("marker = %q, want prefix %q", data, want)
		}
		// The marker names the manifest call 2 evicted, not the rebuild.
		if current := readManifest(t, f.manifestPath()); strings.TrimSuffix(evicted, "\n") == current.Nonce {
			t.Errorf("marker names the current manifest %s, want the evicted one", current.Nonce)
		}
	})

	t.Run("converging rebuild clears marker", func(t *testing.T) {
		f := newToolchainFixture(t)
		stale := f.id.Provenance
		stale.DriverVersion = 12090
		compiles, errs := getOrCompile(t, f, f.id, 3, func(i int) cudacache.ToolProvenance {
			if i == 1 {
				return stale
			}
			return f.id.Provenance
		})
		if compiles != 2 {
			t.Errorf("compile ran %d times, want 2", compiles)
		}
		for i, err := range errs {
			if err != nil {
				t.Errorf("call %d: %v", i+1, err)
			}
		}
		if _, err := os.Stat(f.markerPath()); !os.IsNotExist(err) {
			t.Errorf("marker after a converged hit: stat err %v, want not exist", err)
		}
	})

	t.Run("different mismatch heals again", func(t *testing.T) {
		f := newToolchainFixture(t)
		r1, r2 := f.id.Provenance, f.id.Provenance
		r1.DriverVersion, r2.DriverVersion = 12080, 12090
		compiles, errs := getOrCompile(t, f, f.id, 3, func(i int) cudacache.ToolProvenance {
			if i == 1 {
				return r1
			}
			return r2
		})
		// Call 2 marks (A, r1) and rebuilds; call 3 finds (A, r2), a new pair.
		if compiles != 3 {
			t.Errorf("compile ran %d times, want 3", compiles)
		}
		for i, err := range errs {
			if err != nil {
				t.Errorf("call %d: %v", i+1, err)
			}
		}
	})

	t.Run("lookup racing the eviction heals", func(t *testing.T) {
		// A second lookup that read the manifest before the first evicted
		// it sees the marker the first wrote. It must heal, not give up
		// (review Q2 F2). Restoring the evicted files reproduces its view.
		f := newToolchainFixture(t)
		stale := f.id.Provenance
		stale.DriverVersion = 12090
		if _, err := f.adapter.GetOrCompile(ctx, "input", f.id, func(context.Context) (*cudakernel.Artifact, cudacache.ToolProvenance, error) {
			return art, stale, nil
		}); err != nil {
			t.Fatal(err)
		}
		m := readManifest(t, f.manifestPath())
		artPath := filepath.Join(f.dir, "cache", "artifacts", m.ArtifactKey+".artifact")
		saved := map[string][]byte{}
		for _, p := range []string{f.manifestPath(), artPath} {
			data, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			saved[p] = data
		}
		if _, ok := mustLookup(t, f.adapter, f.id); ok {
			t.Fatal("first Lookup of mismatched entry: hit, want miss")
		}
		for p, data := range saved {
			if err := os.WriteFile(p, data, 0o644); err != nil {
				t.Fatal(err)
			}
		}
		c, err := cudacache.New(cudacache.Config{RootDir: filepath.Join(f.dir, "cache")})
		if err != nil {
			t.Fatal(err)
		}
		if got, ok, err := cudacache.NewNVRTCAdapter(c).Lookup(ctx, "input", f.id); ok || err != nil {
			t.Fatalf("Lookup that raced the eviction = %v, %v, %v; want miss", got, ok, err)
		}
	})

	t.Run("no active tools never mismatches", func(t *testing.T) {
		f := newToolchainFixture(t)
		bare := f.id
		bare.Provenance = cudacache.ToolProvenance{Recipe: cudacache.RecipeVersion}
		compiles, errs := getOrCompile(t, f, bare, 3, func(int) cudacache.ToolProvenance { return f.id.Provenance })
		if compiles != 1 {
			t.Errorf("compile ran %d times, want 1", compiles)
		}
		for i, err := range errs {
			if err != nil {
				t.Errorf("call %d: %v", i+1, err)
			}
		}
		if _, err := os.Stat(filepath.Join(f.dir, "cache", "artifacts", cudacache.CompilerCacheKey("input", bare)+".toolchain-mismatch-marker")); !os.IsNotExist(err) {
			t.Errorf("marker with no active tools: stat err %v, want not exist", err)
		}
	})
}

func TestNVRTCAdapter_RevalidatesAtomicManifestOnLookup(t *testing.T) {
	ctx := context.Background()
	f := newToolchainFixture(t)
	art := newTestArtifact(t, "k", "sm_121a")
	if err := f.adapter.Store(ctx, "input", f.id, art); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(f.manifestPath())
	if err != nil {
		t.Fatalf("manifest: %v", err)
	}
	var m cudacache.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	key := cudacache.CompilerCacheKey("input", f.id)
	if m.Version != 1 || m.Key != key || m.InputKey != "input" || m.PayloadSHA256 != art.Digest || m.ArtifactKey != key+"-"+art.Digest {
		t.Errorf("manifest = %+v", m)
	}
	if m.Recorded.Fingerprint() != f.id.Provenance.Fingerprint() || len(m.ManifestSHA256) != 64 || m.Nonce == "" {
		t.Errorf("manifest recorded %+v sha %q", m.Recorded, m.ManifestSHA256)
	}
	for _, sub := range []string{"manifests", "artifacts"} {
		ents, err := os.ReadDir(filepath.Join(f.dir, "cache", sub))
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range ents {
			if strings.HasPrefix(e.Name(), ".tmp-") {
				t.Errorf("temporary file left in %s: %s", sub, e.Name())
			}
		}
	}

	got, ok := mustLookup(t, f.adapter, f.id)
	if !ok || got.Digest != art.Digest {
		t.Fatalf("Lookup = %v, %v; want hit with digest %s", got, ok, art.Digest)
	}

	// Without a manifest the artifact is not trusted, although it is on disk.
	if err := os.Rename(f.manifestPath(), f.manifestPath()+".aside"); err != nil {
		t.Fatal(err)
	}
	if got, ok := mustLookup(t, f.adapter, f.id); ok {
		t.Fatalf("Lookup without manifest = %v, want miss", got)
	}
	if _, err := os.Stat(filepath.Join(f.dir, "cache", "artifacts", m.ArtifactKey+".artifact")); err != nil {
		t.Fatalf("artifact file: %v", err)
	}
	if err := os.Rename(f.manifestPath()+".aside", f.manifestPath()); err != nil {
		t.Fatal(err)
	}

	// The tool changes on disk after the identity was taken.
	rewriteInPlace(t, f.nvrtc, "nvrtc bytes, build 2")
	got, ok, err = f.adapter.Lookup(ctx, "input", f.id)
	if !errors.Is(err, cudacache.ErrToolIdentityChanged) || ok || got != nil {
		t.Fatalf("Lookup after tool rewrite = %v, %v, %v; want ErrToolIdentityChanged", got, ok, err)
	}
}

func TestCache_QuarantinesTamperedManifestOrPayload(t *testing.T) {
	ctx := context.Background()
	editManifest := func(edit func(m map[string]any)) func(t *testing.T, f *toolchainFixture, m cudacache.Manifest) {
		return func(t *testing.T, f *toolchainFixture, _ cudacache.Manifest) {
			data, err := os.ReadFile(f.manifestPath())
			if err != nil {
				t.Fatal(err)
			}
			var raw map[string]any
			if err := json.Unmarshal(data, &raw); err != nil {
				t.Fatal(err)
			}
			edit(raw)
			if data, err = json.Marshal(raw); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(f.manifestPath(), data, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	editPayload := func(edit func(a *cudakernel.Artifact)) func(t *testing.T, f *toolchainFixture, m cudacache.Manifest) {
		return func(t *testing.T, f *toolchainFixture, m cudacache.Manifest) {
			path := filepath.Join(f.dir, "cache", "artifacts", m.ArtifactKey+".artifact")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var a cudakernel.Artifact
			if err := json.Unmarshal(data, &a); err != nil {
				t.Fatal(err)
			}
			edit(&a)
			if data, err = json.Marshal(&a); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, data, 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	other := newTestArtifact(t, "other", "sm_121a")

	tests := []struct {
		name   string
		tamper func(t *testing.T, f *toolchainFixture, m cudacache.Manifest)
		orphan bool // the manifest no longer names its payload, which stays on disk unreachable
	}{
		{"manifest sha flipped", editManifest(func(m map[string]any) {
			s := m["manifest_sha256"].(string)
			flip := "0"
			if s[0] == '0' {
				flip = "1"
			}
			m["manifest_sha256"] = flip + s[1:]
		}), false},
		{"recorded edited, sha kept", editManifest(func(m map[string]any) {
			m["recorded"].(map[string]any)["driver_version"] = 99999
		}), false},
		{"manifest truncated", func(t *testing.T, f *toolchainFixture, _ cudacache.Manifest) {
			data, err := os.ReadFile(f.manifestPath())
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(f.manifestPath(), data[:len(data)/2], 0o644); err != nil {
				t.Fatal(err)
			}
		}, true},
		{"payload replaced consistently", editPayload(func(a *cudakernel.Artifact) {
			a.Raw, a.Digest = other.Raw, other.Digest
			a.EntryPoint = other.EntryPoint
		}), false},
		{"payload raw only", editPayload(func(a *cudakernel.Artifact) {
			a.Raw = append([]byte(nil), other.Raw...)
		}), false},
		{"payload removed", func(t *testing.T, f *toolchainFixture, m cudacache.Manifest) {
			if err := os.Remove(filepath.Join(f.dir, "cache", "artifacts", m.ArtifactKey+".artifact")); err != nil {
				t.Fatal(err)
			}
		}, false},
		{"identity edited, sha recomputed", func(t *testing.T, f *toolchainFixture, m cudacache.Manifest) {
			m.Identity.TargetArch = "sm_100a"
			m.ManifestSHA256 = ""
			data, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(append([]byte("cudacache-manifest-v1\n"), data...))
			m.ManifestSHA256 = hex.EncodeToString(sum[:])
			if data, err = json.Marshal(m); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(f.manifestPath(), data, 0o644); err != nil {
				t.Fatal(err)
			}
		}, false},
	}
	for _, tt := range tests {
		for _, fresh := range []bool{false, true} {
			name := tt.name + "/same cache"
			if fresh {
				name = tt.name + "/fresh cache"
			}
			t.Run(name, func(t *testing.T) {
				f := newToolchainFixture(t)
				art := newTestArtifact(t, "k", "sm_121a")
				if err := f.adapter.Store(ctx, "input", f.id, art); err != nil {
					t.Fatal(err)
				}
				neighbour := f.id
				neighbour.TargetArch = "sm_100a"
				if err := f.adapter.Store(ctx, "input", neighbour, newTestArtifact(t, "n", "sm_100a")); err != nil {
					t.Fatal(err)
				}
				data, err := os.ReadFile(f.manifestPath())
				if err != nil {
					t.Fatal(err)
				}
				var m cudacache.Manifest
				if err := json.Unmarshal(data, &m); err != nil {
					t.Fatal(err)
				}
				tt.tamper(t, f, m)
				// The memory copy of the payload hides a disk tamper, so the
				// payload cases read through a fresh Cache; the manifest cases
				// run on both.
				c, a := f.cache, f.adapter
				if fresh || strings.HasPrefix(tt.name, "payload") {
					c, err = cudacache.New(cudacache.Config{RootDir: filepath.Join(f.dir, "cache")})
					if err != nil {
						t.Fatal(err)
					}
					a = cudacache.NewNVRTCAdapter(c)
				}
				q0 := f.quarantined(t)
				got, ok, err := a.Lookup(ctx, "input", f.id)
				if !errors.Is(err, cudacache.ErrCorruptArtifact) || ok || got != nil {
					t.Fatalf("Lookup of tampered entry = %v, %v, %v; want ErrCorruptArtifact", got, ok, err)
				}
				// The recomputed digest must pass, so that the key binding is
				// what rejects the entry.
				if tt.name == "identity edited, sha recomputed" && !strings.Contains(err.Error(), "identity has key") {
					t.Fatalf("Lookup error %q, want the identity-to-key check to reject it", err)
				}
				if q := f.quarantined(t); q <= q0 {
					t.Errorf("quarantine holds %d files, had %d; want more", q, q0)
				}
				if got, ok := mustLookup(t, a, f.id); ok {
					t.Errorf("second Lookup = %v, want miss", got)
				}
				if _, err := c.GetArtifact(ctx, m.ArtifactKey); !tt.orphan && !errors.Is(err, cudacache.ErrNotFound) {
					t.Errorf("GetArtifact of quarantined payload = %v, want ErrNotFound", err)
				}
				if _, ok := mustLookup(t, a, neighbour); !ok {
					t.Errorf("untouched neighbour entry: miss, want hit")
				}
			})
		}
	}
}
