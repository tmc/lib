package nvrtcx

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestCompilerOptionsCacheAndArtifactRoundTrip(t *testing.T) {
	useTestCache(t)
	want := []string{"--extra-device-vectorization", "--generate-line-info"}
	module := &CompiledModule{
		output:               OutputPTX,
		ptx:                  "ptx",
		compilerOptions:      append([]string(nil), want...),
		compilerOptionsKnown: true,
	}

	got, known := module.CompilerOptions()
	if !known || !bytes.Equal([]byte(joinOptions(got)), []byte(joinOptions(want))) {
		t.Fatalf("CompilerOptions() = %q, %t; want %q, true", got, known, want)
	}
	got[0] = "changed"
	if after, _ := module.CompilerOptions(); after[0] != want[0] {
		t.Fatalf("CompilerOptions aliases module metadata: %q", after)
	}

	encoded, err := encodeCacheEntry(module)
	if err != nil {
		t.Fatal(err)
	}
	cached, err := decodeCacheEntry("key", encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got, known := cached.CompilerOptions(); !known || joinOptions(got) != joinOptions(want) {
		t.Fatalf("cached CompilerOptions() = %q, %t; want %q, true", got, known, want)
	}
	if err := storeCache("known", module); err != nil {
		t.Fatal(err)
	}
	if got, ok := lookupCache("known"); !ok {
		t.Fatal("memory cache missed module with observed compiler options")
	} else if options, known := got.CompilerOptions(); !known || joinOptions(options) != joinOptions(want) {
		t.Fatalf("memory cache CompilerOptions() = %q, %t; want %q, true", options, known, want)
	}
	memCacheMu.Lock()
	memCache = make(map[string]*CompiledModule)
	memCacheMu.Unlock()
	if got, ok := lookupCache("known"); !ok {
		t.Fatal("disk cache missed module with observed compiler options")
	} else if options, known := got.CompilerOptions(); !known || joinOptions(options) != joinOptions(want) {
		t.Fatalf("disk cache CompilerOptions() = %q, %t; want %q, true", options, known, want)
	}

	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := cached.ToArtifact("kernel", schema, "compute_80")
	if err != nil {
		t.Fatal(err)
	}
	if joinOptions(artifact.CompilerOpts) != joinOptions(want) {
		t.Fatalf("artifact compiler options = %q, want %q", artifact.CompilerOpts, want)
	}
}

func TestCompilerCacheKeyIdentityVersionIsUnchanged(t *testing.T) {
	if cacheMagic != "nvrtcx\x00\x08" {
		t.Fatalf("compiler cache identity version = %q, want v8", cacheMagic)
	}
}

func TestLegacyCacheCompilerOptionsRemainUnknown(t *testing.T) {
	module := &CompiledModule{
		output:               OutputPTX,
		ptx:                  "ptx",
		compilerOptions:      []string{"--extra-device-vectorization"},
		compilerOptionsKnown: true,
	}
	encoded, err := encodeLegacyV8ForTest(module)
	if err != nil {
		t.Fatal(err)
	}
	cached, err := decodeCacheEntry("legacy", encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got, known := cached.CompilerOptions(); known || got != nil {
		t.Fatalf("legacy CompilerOptions() = %q, %t; want nil, false", got, known)
	}
	if _, err := cached.ToArtifact("kernel", cudakernel.Schema{}, "compute_80"); err == nil {
		t.Fatal("ToArtifact accepted a legacy cache entry with unknown compiler options")
	}

	useTestCache(t)
	dir, err := getCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "legacy.cache"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	if got, ok := lookupCache("legacy"); ok || got != nil {
		t.Fatalf("lookupCache returned legacy unknown metadata module: %v, %t", got, ok)
	}
}

func encodeLegacyV8ForTest(cm *CompiledModule) ([]byte, error) {
	data, err := encodeCacheEntry(cm)
	if err != nil {
		return nil, err
	}
	magicLen := len(cacheMagic)
	lengths := make([]uint64, 8)
	fields := make([][]byte, len(lengths))
	start := magicLen + 1 + len(lengths)*8
	for i := range fields {
		lengths[i] = binary.LittleEndian.Uint64(data[magicLen+1+i*8:])
		end := start + int(lengths[i])
		fields[i] = append([]byte(nil), data[start:end]...)
		start = end
	}
	var current cacheMeta
	if err := json.Unmarshal(fields[7], &current); err != nil {
		return nil, err
	}
	fields[7], err = json.Marshal(cacheMeta{JIT: current.JIT, Deps: current.Deps})
	if err != nil {
		return nil, err
	}
	var out bytes.Buffer
	out.WriteString(cacheMagic)
	out.WriteByte(data[magicLen])
	for _, field := range fields {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(field)))
		out.Write(size[:])
	}
	for _, field := range fields {
		out.Write(field)
	}
	checksum := sha256.Sum256(out.Bytes())
	out.Write(checksum[:])
	return out.Bytes(), nil
}

func joinOptions(options []string) string {
	var out bytes.Buffer
	for _, option := range options {
		out.WriteString(option)
		out.WriteByte(0)
	}
	return out.String()
}
