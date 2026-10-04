//go:build native

package cudablock

import (
	"encoding/binary"
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

const topkExtensionFileLimit = 16 << 20

func topkReadExtensionFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("extension packet file is not regular: %s", path)
	}
	if info.Size() > topkExtensionFileLimit {
		return nil, fmt.Errorf("oversized extension packet: %s", path)
	}
	// The limit also applies if the file grows after Stat.
	return topkReadExtensionData(f)
}

func topkReadExtensionData(r io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(r, topkExtensionFileLimit+1))
	if err != nil {
		return nil, err
	}
	if len(data) > topkExtensionFileLimit {
		return nil, fmt.Errorf("oversized extension packet")
	}
	return data, nil
}

type topkPacketGrowingReader struct{ read int }

func (r *topkPacketGrowingReader) Read(p []byte) (int, error) {
	clear(p)
	r.read += len(p)
	return len(p), nil
}

func TestTopKExtensionPacketGrowingRead(t *testing.T) {
	// An endless stream models growth beyond the earlier Stat result.
	r := new(topkPacketGrowingReader)
	data, err := topkReadExtensionData(r)
	if data != nil || err == nil || !strings.Contains(err.Error(), "oversized extension packet") {
		t.Fatalf("growing stream admitted: length=%d, error=%v", len(data), err)
	}
	if r.read != topkExtensionFileLimit+1 {
		t.Fatalf("read %d bytes, want %d", r.read, topkExtensionFileLimit+1)
	}
}

func topkReadExtensionPacket(dir, name, source, entry, target string) (*cudakernel.Artifact, topkPacketEntry, error) {
	var a cudakernel.Artifact
	var r topkPacketEntry
	fail := func(err error) (*cudakernel.Artifact, topkPacketEntry, error) { return nil, r, err }
	if !filepath.IsAbs(dir) {
		return fail(fmt.Errorf("absolute extension packet directory required"))
	}
	read := func(suffix string, v any) error {
		data, err := topkReadExtensionFile(filepath.Join(dir, name+suffix))
		if err != nil {
			return err
		}
		return json.Unmarshal(data, v)
	}
	if err := read(".artifact.json", &a); err != nil {
		return fail(err)
	}
	if err := read(".receipt.json", &r); err != nil {
		return fail(err)
	}
	s, err := topkReadExtensionFile(filepath.Join(dir, name+".cu"))
	if err != nil {
		return fail(err)
	}
	schema, err := TopKSchema()
	if err != nil {
		return fail(err)
	}
	if string(s) != source || r.Name != name || r.SourceSHA256 != topkDigest([]byte(source)) || r.ArtifactSHA256 != a.Digest || r.Target != target || a.Target != target || a.EntryPoint != entry || a.Kind != cudakernel.KindCUBIN || !reflect.DeepEqual(a.Schema, schema) || !reflect.DeepEqual(a.CompilerOpts, r.CompilerOptions) {
		return fail(fmt.Errorf("extension packet identity mismatch"))
	}
	// These are producer-reported facts. Agreement does not attest that the
	// recorded compiler produced this image from the retained source.
	opts := []string{"--gpu-architecture=" + target, "--std=c++17", "--generate-line-info"}
	if !filepath.IsAbs(r.CompilerPath) || strings.ContainsRune(r.CompilerPath, 0) || r.CompilerMajor <= 0 || r.CompilerMinor < 0 || a.Toolchain != fmt.Sprintf("nvrtc %d.%d", r.CompilerMajor, r.CompilerMinor) || !reflect.DeepEqual(a.CompilerOpts, opts) {
		return fail(fmt.Errorf("invalid retained compiler metadata"))
	}
	if err := a.Validate(); err != nil {
		return fail(err)
	}
	return &a, r, nil
}

func TestTopKExtensionPacket(t *testing.T) {
	for _, name := range []string{"valid", "empty-path", "relative-path", "nul-path", "zero-major", "negative-major", "negative-minor", "version-mismatch", "empty-toolchain", "wrong-options", "malformed-json", "oversized-cubin", "oversized-source"} {
		t.Run(name, func(t *testing.T) {
			// This synthetic ELF tests the host parser only; it is never loaded.
			raw := make([]byte, 64)
			copy(raw, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1, 0x33, 7})
			binary.LittleEndian.PutUint16(raw[16:], 2)
			binary.LittleEndian.PutUint16(raw[18:], 190)
			binary.LittleEndian.PutUint32(raw[20:], 1)
			binary.LittleEndian.PutUint32(raw[48:], 121)
			binary.LittleEndian.PutUint16(raw[52:], 64)
			schema, err := TopKSchema()
			if err != nil {
				t.Fatal(err)
			}
			opts := []string{"--gpu-architecture=sm_121", "--std=c++17", "--generate-line-info"}
			a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, "topk_bf16", schema, "sm_121", cudakernel.WithToolchain("nvrtc 13.4"), cudakernel.WithCompilerOptions(opts))
			if err != nil {
				t.Fatal(err)
			}
			r := topkPacketEntry{Name: "bf16", SourceSHA256: topkDigest([]byte(TopKBF16Source)), ArtifactSHA256: a.Digest, Target: a.Target, CompilerPath: "/recorded/libnvrtc.so", CompilerMajor: 13, CompilerMinor: 4, CompilerOptions: opts}
			switch name {
			case "empty-path":
				r.CompilerPath = ""
			case "relative-path":
				r.CompilerPath = "libnvrtc.so"
			case "nul-path":
				r.CompilerPath += "\x00"
			case "zero-major":
				r.CompilerMajor = 0
			case "negative-major":
				r.CompilerMajor = -1
			case "negative-minor":
				r.CompilerMinor = -1
			case "version-mismatch":
				r.CompilerMinor = 3
			case "empty-toolchain":
				a.Toolchain = ""
			case "wrong-options":
				a.CompilerOpts = nil
				r.CompilerOptions = nil
			}
			dir := t.TempDir()
			topkWriteJSON(t, filepath.Join(dir, "bf16.artifact.json"), a)
			topkWriteJSON(t, filepath.Join(dir, "bf16.receipt.json"), r)
			if err := os.WriteFile(filepath.Join(dir, "bf16.cu"), []byte(TopKBF16Source), 0600); err != nil {
				t.Fatal(err)
			}
			if name == "malformed-json" {
				if err := os.WriteFile(filepath.Join(dir, "bf16.receipt.json"), []byte("{"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if name == "oversized-cubin" || name == "oversized-source" {
				suffix := ".artifact.json"
				if name == "oversized-source" {
					suffix = ".cu"
				}
				if err := os.Truncate(filepath.Join(dir, "bf16"+suffix), topkExtensionFileLimit+1); err != nil {
					t.Fatal(err)
				}
			}
			_, _, err = topkReadExtensionPacket(dir, "bf16", TopKBF16Source, "topk_bf16", "sm_121")
			if name == "valid" && err != nil {
				t.Fatal(err)
			}
			if name != "valid" && err == nil {
				t.Fatal("invalid packet admitted")
			}
			if strings.HasPrefix(name, "oversized-") && !strings.Contains(err.Error(), "oversized extension packet") {
				t.Fatalf("wrong rejection: %v", err)
			}
		})
	}
}
