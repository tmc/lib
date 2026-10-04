package cudaattn

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

type indexedPacket struct {
	Version                                       int
	SourceDigest, ImageDigest                     string
	CompilerPath, CompilerVersion, CompilerDigest string
	BuiltinsPath, BuiltinsDigest                  string
	Artifact                                      cudakernel.Artifact
}

func indexedDigest(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }
func indexedReadBounded(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > limit {
		return nil, fmt.Errorf("indexed packet path size/type refused")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !s.Mode().IsRegular() || s.Size() < 1 || s.Size() > limit {
		return nil, fmt.Errorf("indexed packet file size/type refused")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("indexed packet grew beyond bound")
	}
	return b, nil
}

func indexedReadPacket(dir, name, source, entry string, schema cudakernel.Schema, opts []string) (*indexedPacket, error) {
	if name == "" || filepath.Base(name) != name || strings.ContainsAny(name, "/\\") {
		return nil, fmt.Errorf("invalid indexed packet name")
	}
	b, err := indexedReadBounded(filepath.Join(dir, name+".json"), 16<<20)
	if err != nil {
		return nil, err
	}
	var p indexedPacket
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&p); err != nil {
		return nil, err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return nil, fmt.Errorf("indexed packet trailing json")
	}
	validDigest := func(s string) bool {
		b, err := hex.DecodeString(s)
		return err == nil && len(b) == 32 && s == strings.ToLower(s)
	}
	if p.Version != 1 || p.SourceDigest != indexedDigest([]byte(source)) || p.ImageDigest != p.Artifact.Digest || p.CompilerVersion != "nvrtc 13.4" || p.Artifact.Toolchain != p.CompilerVersion || !filepath.IsAbs(p.CompilerPath) || filepath.Base(p.CompilerPath) != "libnvrtc.so.13.4.59" || !filepath.IsAbs(p.BuiltinsPath) || filepath.Base(p.BuiltinsPath) != "libnvrtc-builtins.so.13.4.59" || !validDigest(p.CompilerDigest) || !validDigest(p.BuiltinsDigest) {
		return nil, fmt.Errorf("indexed compiler/source metadata disagreement")
	}
	a := &p.Artifact
	if err := a.Validate(); err != nil {
		return nil, err
	}
	if a.Kind != cudakernel.KindCUBIN || a.Target != "sm_121" || a.EntryPoint != entry || !reflect.DeepEqual(a.Schema, schema) || !reflect.DeepEqual(a.CompilerOpts, opts) {
		return nil, fmt.Errorf("indexed artifact contract mismatch")
	}
	path := "cudaattn_indexed.cu"
	if entry == "topk_fp32" {
		path = "cudablock_topk.cu"
	}
	found := false
	for _, dep := range a.Dependencies {
		if dep.Path != path {
			continue
		}
		if found || !dep.Tracked || dep.ErrorMsg != "" || dep.Digest != p.SourceDigest {
			return nil, fmt.Errorf("indexed packet source dependency disagreement")
		}
		found = true
	}
	if !found {
		return nil, fmt.Errorf("indexed packet source dependency missing")
	}
	cu, err := indexedReadBounded(filepath.Join(dir, name+".cu"), 1<<20)
	if err != nil {
		return nil, err
	}
	if string(cu) != source {
		return nil, fmt.Errorf("indexed retained source mismatch")
	}
	image, err := indexedReadBounded(filepath.Join(dir, name+".cubin"), 16<<20)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(image, a.Raw) || indexedDigest(image) != p.ImageDigest {
		return nil, fmt.Errorf("indexed retained image mismatch")
	}
	return &p, nil
}

func TestIndexedPacketBounds(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "file")
	for _, n := range []int{0, 1, 8, 9} {
		if err := os.WriteFile(path, make([]byte, n), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := indexedReadBounded(path, 8)
		if (err == nil) != (n >= 1 && n <= 8) {
			t.Fatalf("size %d: %v", n, err)
		}
	}
	if _, err := indexedReadBounded(dir, 8); err == nil {
		t.Fatal("directory accepted")
	}
	for _, name := range []string{"", "../outside"} {
		if _, err := indexedReadPacket(dir, name, "", "", cudakernel.Schema{}, nil); err == nil {
			t.Fatal("invalid name accepted")
		}
	}
}

func TestIndexedPacketMetadataControls(t *testing.T) {
	dir := t.TempDir()
	schema, _ := IndexedSchema()
	image := make([]byte, 64)
	copy(image, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1, 0x33, 7})
	binary.LittleEndian.PutUint16(image[16:], 2)
	binary.LittleEndian.PutUint16(image[18:], 190)
	binary.LittleEndian.PutUint32(image[20:], 1)
	binary.LittleEndian.PutUint32(image[48:], 121)
	binary.LittleEndian.PutUint16(image[52:], 64)
	opts := []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "indexed_attention", schema, "sm_121", cudakernel.WithToolchain("nvrtc 13.4"), cudakernel.WithCompilerOptions(opts), cudakernel.WithDependencies([]cudakernel.Dependency{{Path: "cudaattn_indexed.cu", Digest: indexedDigest([]byte(IndexedSource)), Tracked: true}}))
	if err != nil {
		t.Fatal(err)
	}
	if err := indexedAdmitArtifact(a); err != nil {
		t.Fatal(err)
	}
	for _, edit := range []func(*cudakernel.Artifact){
		func(a *cudakernel.Artifact) { a.CompilerOpts = []string{"--ftz=true"} },
		func(a *cudakernel.Artifact) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) },
		func(a *cudakernel.Artifact) { a.Dependencies[0].ErrorMsg = "untracked" },
		func(a *cudakernel.Artifact) { a.Schema.Params[0].Offset++ },
	} {
		q := indexedClone(a)
		edit(q)
		if indexedAdmitArtifact(q) == nil {
			t.Fatal("damaged loaded artifact contract admitted")
		}
	}
	p := indexedPacket{Version: 1, SourceDigest: indexedDigest([]byte(IndexedSource)), ImageDigest: a.Digest, CompilerPath: "/tool/libnvrtc.so.13.4.59", CompilerVersion: "nvrtc 13.4", CompilerDigest: strings.Repeat("a", 64), BuiltinsPath: "/tool/libnvrtc-builtins.so.13.4.59", BuiltinsDigest: strings.Repeat("b", 64), Artifact: *a}
	write := func(p indexedPacket) {
		t.Helper()
		b, err := json.Marshal(p)
		if err != nil {
			t.Fatal(err)
		}
		for suffix, data := range map[string][]byte{".json": b, ".cu": []byte(IndexedSource), ".cubin": image} {
			if err := os.WriteFile(filepath.Join(dir, "baseline"+suffix), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	read := func() error {
		_, err := indexedReadPacket(dir, "baseline", IndexedSource, "indexed_attention", schema, opts)
		return err
	}
	write(p)
	if err := read(); err != nil {
		t.Fatal(err)
	} // Parser-only synthetic ELF, never loaded.
	for _, tt := range []struct {
		name string
		edit func(*indexedPacket)
	}{
		{"version", func(p *indexedPacket) { p.Version = 2 }},
		{"empty-compiler", func(p *indexedPacket) { p.CompilerPath = "" }},
		{"relative-compiler", func(p *indexedPacket) { p.CompilerPath = "libnvrtc.so.13.4.59" }},
		{"compiler-version", func(p *indexedPacket) { p.CompilerVersion = "nvrtc 13.0" }},
		{"builtins-version", func(p *indexedPacket) { p.BuiltinsPath = "/tool/libnvrtc-builtins.so.13.0.88" }},
		{"digest", func(p *indexedPacket) { p.CompilerDigest = "unknown" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			q := p
			tt.edit(&q)
			write(q)
			if read() == nil {
				t.Fatal("metadata damage admitted")
			}
		})
	}
	for _, tt := range []struct {
		suffix string
		limit  int64
	}{{".cu", 1 << 20}, {".cubin", 16 << 20}} {
		write(p)
		f, err := os.OpenFile(filepath.Join(dir, "baseline"+tt.suffix), os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Truncate(tt.limit + 1); err != nil {
			t.Fatal(err)
		}
		f.Close()
		if read() == nil {
			t.Fatal("oversized retained data admitted")
		}
	}
}
