package model

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatma"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

var aotArchs = []string{"sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a", "sm_121a"}

func aotPTX(arch, entry string) []byte {
	return fmt.Appendf(nil, `.version 9.0
.target %s
.address_size 64

.visible .entry %s(
	.param .u64 %s_param_0
)
{
	ret;
}
`, arch, entry, entry)
}

func aotFixture(t *testing.T) ([]byte, []ArchLaunchSpec, map[string]cudatma.Descriptor) {
	t.Helper()
	var entries []FatbinEntry
	var specs []ArchLaunchSpec
	for _, arch := range aotArchs {
		sm, err := smNumber(arch)
		if err != nil {
			t.Fatal(err)
		}
		entries = append(entries, FatbinEntry{Kind: FatbinKindPTX, SM: sm, Payload: aotPTX(arch, "layer_megakernel")})
		s := ArchLaunchSpec{
			Arch:              arch,
			KernelSymbol:      "layer_megakernel",
			GridDim:           [3]uint32{132, 1, 1},
			BlockDim:          [3]uint32{384, 1, 1},
			ClusterDim:        [3]uint32{2, 1, 1},
			DynamicSharedSize: 200 << 10,
			ProducerRegs:      40,
			ConsumerRegs:      232,
			UsePDL:            true,
		}
		switch sm {
		case 100, 103, 107:
			s.GridDim[0] = 148
			s.TMEMColsUsed = 256
		case 120, 121:
			s.GridDim[0] = 48
			s.DynamicSharedSize = 96 << 10
			s.ProducerRegs, s.ConsumerRegs = 0, 0
		}
		specs = append(specs, s)
	}
	fatbin, err := BuildFatbin(entries)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]uint16, 64*256)
	l := StreamLoader{Target: "sm_100a"}
	b, err := l.PackTensor("w", 64, 256, raw)
	if err != nil {
		t.Fatal(err)
	}
	tma := map[string]cudatma.Descriptor{"w_tiles": b.WeightTMADesc, "w_source": b.SourceTMADesc}
	return fatbin, specs, tma
}

func TestFatbinRoundTrip(t *testing.T) {
	in := []FatbinEntry{
		{Kind: FatbinKindPTX, SM: 121, Payload: aotPTX("sm_121a", "k")},
		{Kind: FatbinKindCubin, SM: 100, Payload: []byte("\x7fELF not really")},
	}
	data, err := BuildFatbin(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(data)%8 != 0 {
		t.Errorf("container is %d bytes, not 8-byte aligned", len(data))
	}
	out, err := InspectFatbin(append(bytes.Clone(data), data...))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("two containers gave %d entries", len(out))
	}
	for i, e := range out {
		w := in[i%2]
		if e.Kind != w.Kind || e.SM != w.SM || e.Compressed || !bytes.Equal(bytes.TrimRight(e.Payload, "\x00"), w.Payload) {
			t.Errorf("entry %d = %+v, want %+v", i, e, w)
		}
	}

	bad := []struct {
		name string
		data []byte
		want string
	}{
		{"magic", append([]byte{1, 2, 3, 4}, data[4:]...), "bad magic"},
		{"truncated", data[:len(data)-9], "overruns"},
		{"short", data[:10], "truncated container header at offset 0"},
		{"kind", func() []byte { d := bytes.Clone(data); d[16] = 7; return d }(), "entry at offset 16 has kind 7"},
	}
	for _, tt := range bad {
		if _, err := InspectFatbin(tt.data); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: err = %v, want %q", tt.name, err, tt.want)
		}
	}
}

// nvccFatbin reads a fatbin that nvcc 13.4 built from
// testdata/nvcc-fill/fill.cu with
//
//	nvcc -fatbin -arch=sm_121a fill.cu                         # default
//	nvcc -fatbin -arch=sm_121a --no-compress fill.cu           # nocompress
//	nvcc -fatbin -arch=sm_121a -Xfatbin -compress-all fill.cu  # compressall
//
// stored as xxd -p hex. testdata/nvcc-fill/cuobjdump.txt lists each.
func nvccFatbin(t *testing.T, name string) []byte {
	t.Helper()
	h, err := os.ReadFile(filepath.Join("testdata", "nvcc-fill", name+".fatbin.hex"))
	if err != nil {
		t.Fatal(err)
	}
	data, err := hex.DecodeString(strings.Join(strings.Fields(string(h)), ""))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// TestInspectFatbinNVCC checks the entry header fields against fatbins
// from nvcc 13.4. cuobjdump lists each as PTX sm_121, ELF sm_121a and
// PTX sm_121a, with the PTX compressed by default and the cubin only
// under -compress-all.
func TestInspectFatbinNVCC(t *testing.T) {
	type entry struct {
		kind, sm, major, minor   int
		archSpecific, compressed bool
		uncompressed, payload    int
	}
	tests := []struct {
		name string
		want []entry
	}{
		{"nocompress", []entry{
			{FatbinKindPTX, 121, 9, 4, false, false, 0, 591},
			{FatbinKindCubin, 121, 1, 8, true, false, 0, 5568},
			{FatbinKindPTX, 121, 9, 4, true, false, 0, 592},
		}},
		{"default", []entry{
			{FatbinKindPTX, 121, 9, 4, false, true, 592, 0},
			{FatbinKindCubin, 121, 1, 8, true, false, 0, 5568},
			{FatbinKindPTX, 121, 9, 4, true, true, 593, 0},
		}},
		{"compressall", []entry{
			{FatbinKindPTX, 121, 9, 4, false, true, 592, 0},
			{FatbinKindCubin, 121, 1, 8, true, true, 5568, 0},
			{FatbinKindPTX, 121, 9, 4, true, true, 593, 0},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			entries, err := InspectFatbin(nvccFatbin(t, tt.name))
			if err != nil {
				t.Fatal(err)
			}
			var got []entry
			for _, e := range entries {
				got = append(got, entry{e.Kind, e.SM, e.Major, e.Minor, e.ArchSpecific, e.Compressed, e.UncompressedSize, len(e.Payload)})
				switch {
				case e.Compressed:
				case e.Kind == FatbinKindCubin:
					if sm := cubinSM(e.Payload); sm != 121 {
						t.Errorf("cubin e_flags SM = %d, want 121", sm)
					}
				case e.ArchSpecific != strings.Contains(string(e.Payload), ".target sm_121a\n"):
					t.Errorf("PTX entry arch-specific=%v disagrees with its .target", e.ArchSpecific)
				}
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("entries:\n got %v\nwant %v", got, tt.want)
			}

			spec := ArchLaunchSpec{Arch: "sm_121a", KernelSymbol: "aot_fill", GridDim: [3]uint32{8, 1, 1}, BlockDim: [3]uint32{128, 1, 1}}
			if _, err := PackAOTArtifact("fill", nvccFatbin(t, tt.name), []ArchLaunchSpec{spec}, nil); err != nil {
				t.Fatalf("pack: %v", err)
			}
			spec.KernelSymbol = "other"
			_, err = PackAOTArtifact("fill", nvccFatbin(t, tt.name), []ArchLaunchSpec{spec}, nil)
			if tt.name == "compressall" {
				// Nothing is checkable; compressed images match by SM.
				if err != nil {
					t.Fatalf("pack compressed with unknown symbol: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "other") {
				t.Fatalf("pack with unknown symbol: err = %v", err)
			}
		})
	}
}

// TestBuildFatbinMatchesNVCC rebuilds the uncompressed nvcc fatbin's
// entries and compares the header fields BuildFatbin writes with nvcc's.
func TestBuildFatbinMatchesNVCC(t *testing.T) {
	ref := nvccFatbin(t, "nocompress")
	in, err := InspectFatbin(ref)
	if err != nil {
		t.Fatal(err)
	}
	data, err := BuildFatbin(in)
	if err != nil {
		t.Fatal(err)
	}
	out, err := InspectFatbin(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(in) {
		t.Fatalf("%d entries, want %d", len(out), len(in))
	}
	if !bytes.Equal(data[:8], ref[:8]) {
		t.Errorf("container header % x, nvcc % x", data[:8], ref[:8])
	}
	const known = fatbinFlag64Bit | fatbinFlagHostLinux | fatbinFlagArchSpecific | fatbinFlagCompressLZ4 | fatbinFlagCompressZstd
	for i, p, q := 0, fatbinHeaderSize, fatbinHeaderSize; i < len(in); i++ {
		if !reflect.DeepEqual(out[i], in[i]) {
			t.Errorf("entry %d = %+v, want %+v", i, out[i], in[i])
		}
		h, r := data[p:p+fatbinEntrySize], ref[q:q+fatbinEntrySize]
		for _, f := range []struct {
			name string
			off  int
			size int
		}{{"kind", 0, 2}, {"version", 24, 4}, {"sm", 28, 4}} {
			if !bytes.Equal(h[f.off:f.off+f.size], r[f.off:f.off+f.size]) {
				t.Errorf("entry %d %s % x, nvcc % x", i, f.name, h[f.off:f.off+f.size], r[f.off:f.off+f.size])
			}
		}
		if hf, rf := binary.LittleEndian.Uint64(h[40:]), binary.LittleEndian.Uint64(r[40:])&known; hf != rf {
			t.Errorf("entry %d flags %#x, nvcc %#x (masked)", i, hf, rf)
		}
		p += int(binary.LittleEndian.Uint32(h[4:])) + int(binary.LittleEndian.Uint64(h[8:]))
		q += int(binary.LittleEndian.Uint32(r[4:])) + int(binary.LittleEndian.Uint64(r[8:]))
	}
}

func TestPackAOTArtifactDeterministic(t *testing.T) {
	fatbin, specs, tma := aotFixture(t)
	var digests, goSrcs, hdrs []string
	for range 2 {
		a, err := PackAOTArtifact("llama_block", fatbin, specs, tma)
		if err != nil {
			t.Fatal(err)
		}
		g, err := a.EmitGoEmbedPackage("llamaaot", "llama.fatbin")
		if err != nil {
			t.Fatal(err)
		}
		h, err := a.EmitCHeader("LLAMA_AOT")
		if err != nil {
			t.Fatal(err)
		}
		digests, goSrcs, hdrs = append(digests, a.SHA256Digest), append(goSrcs, g), append(hdrs, h)
	}
	if digests[0] != digests[1] || goSrcs[0] != goSrcs[1] || hdrs[0] != hdrs[1] {
		t.Fatal("packing the same inputs twice gave different outputs")
	}
	if len(digests[0]) != 64 {
		t.Fatalf("digest %q", digests[0])
	}
	// The digest covers the fatbin and the manifest.
	mutFatbin := bytes.Clone(fatbin)
	mutFatbin[len(mutFatbin)-20] ^= 1
	a, err := PackAOTArtifact("llama_block", mutFatbin, specs, tma)
	if err != nil {
		t.Fatal(err)
	}
	if a.SHA256Digest == digests[0] {
		t.Error("digest ignores the fatbin bytes")
	}
	specs[0].GridDim[0] += 2
	if a, err = PackAOTArtifact("llama_block", fatbin, specs, tma); err != nil {
		t.Fatal(err)
	}
	if a.SHA256Digest == digests[0] {
		t.Error("digest ignores the launch specs")
	}
}

func TestPackAOTArtifactRejects(t *testing.T) {
	fatbin, specs, tma := aotFixture(t)
	clone := func() []ArchLaunchSpec { return append([]ArchLaunchSpec(nil), specs...) }
	tests := []struct {
		name   string
		fatbin []byte
		specs  func() []ArchLaunchSpec
		tma    func() map[string]cudatma.Descriptor
		want   string
	}{
		{name: "duplicate", specs: func() []ArchLaunchSpec { return append(clone(), specs[0]) }, want: "duplicate spec"},
		{name: "no image", fatbin: func() []byte {
			f, _ := BuildFatbin([]FatbinEntry{{Kind: FatbinKindPTX, SM: 90, Payload: aotPTX("sm_90a", "layer_megakernel")}})
			return f
		}(), want: "fatbin has no sm_100 image"},
		{name: "wrong entry", specs: func() []ArchLaunchSpec { s := clone(); s[2].KernelSymbol = "other"; return s }, want: "no .entry other"},
		{name: "cluster", specs: func() []ArchLaunchSpec {
			s := clone()
			s[1].ClusterDim = [3]uint32{4, 4, 2}
			s[1].GridDim = [3]uint32{148, 4, 2}
			return s
		}, want: "exceeds 16"},
		{name: "cluster divides", specs: func() []ArchLaunchSpec { s := clone(); s[1].ClusterDim = [3]uint32{3, 1, 1}; return s }, want: "does not divide"},
		{name: "tmem on sm_121a", specs: func() []ArchLaunchSpec { s := clone(); s[5].TMEMColsUsed = 128; return s }, want: "has no tensor memory"},
		{name: "smem", specs: func() []ArchLaunchSpec { s := clone(); s[4].DynamicSharedSize = 200 << 10; return s }, want: "exceed 101376"},
		{name: "regs", specs: func() []ArchLaunchSpec { s := clone(); s[0].ConsumerRegs = 230; return s }, want: "setmaxnreg"},
		{name: "unknown arch", specs: func() []ArchLaunchSpec { s := clone(); s[0].Arch = "sm_99a"; return s }, want: "unsupported target"},
		{name: "box", tma: func() map[string]cudatma.Descriptor {
			d := tma["w_tiles"]
			d.Box[1] = 300
			return map[string]cudatma.Descriptor{"w_tiles": d}
		}, want: "tensor map w_tiles: dimension 1"},
		{name: "base", tma: func() map[string]cudatma.Descriptor {
			d := tma["w_tiles"]
			d.Base = 0x1000
			return map[string]cudatma.Descriptor{"w_tiles": d}
		}, want: "want 0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, s, m := fatbin, specs, tma
			if tt.fatbin != nil {
				f = tt.fatbin
			}
			if tt.specs != nil {
				s = tt.specs()
			}
			if tt.tma != nil {
				m = tt.tma()
			}
			_, err := PackAOTArtifact("llama_block", f, s, m)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

// TestPackAOTArtifactCubin checks a real sm_121 cubin (the cudaoccupancy
// fixture) by symbol, stack frame and e_flags SM.
func TestPackAOTArtifactCubin(t *testing.T) {
	hexData, err := os.ReadFile(filepath.Join("..", "..", "cudaoccupancy", "testdata", "kernels.cubin.hex"))
	if err != nil {
		t.Skipf("cubin fixture: %v", err)
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(hexData)))
	if err != nil {
		t.Fatal(err)
	}
	if sm := cubinSM(cubin); sm != 121 {
		t.Fatalf("fixture e_flags SM = %d, want 121", sm)
	}
	spec := ArchLaunchSpec{Arch: "sm_121a", KernelSymbol: "vector_add", GridDim: [3]uint32{4, 1, 1}, BlockDim: [3]uint32{256, 1, 1}}
	for _, tt := range []struct {
		sm     int
		symbol string
		arch   string
		want   string
	}{
		{121, "vector_add", "sm_121a", ""},
		{121, "missing_kernel", "sm_121a", "no kernel missing_kernel"},
		{120, "vector_add", "sm_120a", "cubin image is for sm_121"},
	} {
		f, err := BuildFatbin([]FatbinEntry{{Kind: FatbinKindCubin, SM: tt.sm, Payload: cubin}})
		if err != nil {
			t.Fatal(err)
		}
		s := spec
		s.KernelSymbol, s.Arch = tt.symbol, tt.arch
		_, err = PackAOTArtifact("fixture", f, []ArchLaunchSpec{s}, nil)
		if tt.want == "" && err != nil || tt.want != "" && (err == nil || !strings.Contains(err.Error(), tt.want)) {
			t.Errorf("%s on sm_%d: err = %v, want %q", tt.symbol, tt.sm, err, tt.want)
		}
	}
}

// TestEmitGoEmbedPackageTypeChecks type-checks the generated package
// against the module's driver package and builds it.
func TestEmitGoEmbedPackageTypeChecks(t *testing.T) {
	fatbin, specs, tma := aotFixture(t)
	a, err := PackAOTArtifact("llama_block", fatbin, specs, tma)
	if err != nil {
		t.Fatal(err)
	}
	src, err := a.EmitGoEmbedPackage("llamaaot", "llama.fatbin")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "llamaaot.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse generated source: %v\n%s", err, src)
	}
	for _, imp := range f.Imports {
		if p := strings.Trim(imp.Path.Value, `"`); strings.Contains(p, "/nvidia/x/") {
			t.Errorf("generated package imports %s", p)
		}
	}
	conf := types.Config{Importer: importer.ForCompiler(fset, "source", nil)}
	if _, err := conf.Check("llamaaot", fset, []*ast.File{f}, nil); err != nil {
		t.Fatalf("type-check generated source: %v", err)
	}
	for _, want := range []string{"//go:embed llama.fatbin", `Arch: "sm_121a"`, `SHA256 = "` + a.SHA256Digest, "LaunchAttributeClusterDimension"} {
		if !strings.Contains(src, want) {
			t.Errorf("generated source missing %q", want)
		}
	}
	if testing.Short() {
		return
	}
	// Build inside the module so the driver import resolves; testdata
	// keeps the package out of ./... patterns.
	if _, err := os.Stat("testdata"); os.IsNotExist(err) {
		if err := os.Mkdir("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.Remove("testdata") })
	}
	dir, err := os.MkdirTemp("testdata", "aotgen")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	if err := os.WriteFile(filepath.Join(dir, "llamaaot.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "llama.fatbin"), a.FatbinBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "build", "./"+filepath.ToSlash(dir))
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build generated package: %v\n%s", err, out)
	}
}

func TestEmitCHeaderCompiles(t *testing.T) {
	fatbin, specs, tma := aotFixture(t)
	a, err := PackAOTArtifact("llama_block", fatbin, specs, tma)
	if err != nil {
		t.Fatal(err)
	}
	h, err := a.EmitCHeader("LLAMA_AOT")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"#ifndef LLAMA_AOT_H", "llama_aot_fatbin[", "LLAMA_AOT_ALIGN(128)", `{"sm_107a", "layer_megakernel", {148, 1, 1}`, "#define LLAMA_AOT_NUM_SPECS 6"} {
		if !strings.Contains(h, want) {
			t.Errorf("header missing %q", want)
		}
	}
	if _, err := a.EmitCHeader("lower"); err == nil {
		t.Error("accepted a lower-case guard prefix")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("no C compiler")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "llama_aot.h"), []byte(h), 0o644); err != nil {
		t.Fatal(err)
	}
	main := `#include "llama_aot.h"
int main(void) {
	return llama_aot_fatbin[0] == 0x50 && llama_aot_specs[0].grid_dim[0] > 0 && llama_aot_tensor_maps[0].rank > 0 ? 0 : 1;
}
`
	if err := os.WriteFile(filepath.Join(dir, "main.c"), []byte(main), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "main")
	if out, err := exec.Command(cc, "-std=c99", "-Wall", "-Werror", "-o", bin, filepath.Join(dir, "main.c")).CombinedOutput(); err != nil {
		t.Fatalf("cc: %v\n%s", err, out)
	}
	if out, err := exec.Command(bin).CombinedOutput(); err != nil {
		t.Fatalf("header values: %v\n%s", err, out)
	}
}

func ExamplePackAOTArtifact() {
	ptx := `.version 9.0
.target sm_121a
.address_size 64
.visible .entry decode_step(.param .u64 p)
{
	ret;
}
`
	fatbin, err := BuildFatbin([]FatbinEntry{{Kind: FatbinKindPTX, SM: 121, Payload: []byte(ptx)}})
	if err != nil {
		fmt.Println(err)
		return
	}
	spec := ArchLaunchSpec{
		Arch:         "sm_121a",
		KernelSymbol: "decode_step",
		GridDim:      [3]uint32{48, 1, 1},
		BlockDim:     [3]uint32{256, 1, 1},
	}
	a, err := PackAOTArtifact("decoder", fatbin, []ArchLaunchSpec{spec}, nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(a.FatbinBytes), len(a.SHA256Digest), a.ArchSpecs["sm_121a"].KernelSymbol)
	// Output: 184 64 decode_step
}

const aotNativeKernel = `extern "C" __global__ void aot_fill(int *out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = i * 3 + 1;
}
`

// aotNativeMain launches the generated runner's kernel and checks
// out[i] == 3i+1.
const aotNativeMain = `package main

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"

	aot "MODPATH/aot"
)

func main() {
	if err := run(); err != nil {
		fmt.Println("aot native:", err)
		os.Exit(1)
	}
	fmt.Println("AOT_NATIVE_OK")
}

func run() error {
	if err := driver.Init(0); err != nil {
		return err
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		return err
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return err
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		return err
	}
	r, err := aot.Load(dev)
	if err != nil {
		return err
	}
	s := r.Spec
	n := int32(s.GridDim[0] * s.BlockDim[0])
	out, err := driver.MemAllocBytes(uint(n) * 4)
	if err != nil {
		return err
	}
	defer out.Free()
	if err := r.Launch(0, []unsafe.Pointer{unsafe.Pointer(&out), unsafe.Pointer(&n)}); err != nil {
		return err
	}
	if err := driver.CtxSynchronize(); err != nil {
		return err
	}
	got := make([]int32, n)
	if err := out.CopyToHostPtr(unsafe.Pointer(&got[0]), uint(n)*4); err != nil {
		return err
	}
	for i, v := range got {
		if v != int32(3*i+1) {
			return fmt.Errorf("out[%d] = %d, want %d", i, v, 3*i+1)
		}
	}
	fmt.Printf("%s %s grid=%v cluster=%v pdl=%v elements=%d\n", s.Arch, s.KernelSymbol, s.GridDim, s.ClusterDim, s.UsePDL, n)
	return nil
}
`

// writeAOTProgram writes the generated package for a under dir/aot and
// the launching program under dir, inside the module's testdata.
func writeAOTProgram(t *testing.T, a AOTArtifact) string {
	t.Helper()
	if _, err := os.Stat("testdata"); os.IsNotExist(err) {
		if err := os.Mkdir("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.Remove("testdata") })
	}
	dir, err := os.MkdirTemp("testdata", "aotnative")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	src, err := a.EmitGoEmbedPackage("aot", "model.fatbin")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "aot"), 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{
		filepath.Join(dir, "aot", "aot.go"):       []byte(src),
		filepath.Join(dir, "aot", "model.fatbin"): a.FatbinBytes,
		filepath.Join(dir, "main.go"):             []byte(strings.ReplaceAll(aotNativeMain, "MODPATH", "github.com/tmc/lib/nvidia/x/egraph/model/"+filepath.ToSlash(dir))),
	}
	for name, data := range files {
		if err := os.WriteFile(name, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return "./" + filepath.ToSlash(dir)
}

func TestAOTNativeProgramBuilds(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a program")
	}
	f, err := BuildFatbin([]FatbinEntry{{Kind: FatbinKindPTX, SM: 121, Payload: aotPTX("sm_121a", "aot_fill")}})
	if err != nil {
		t.Fatal(err)
	}
	a, err := PackAOTArtifact("fill", f, []ArchLaunchSpec{{Arch: "sm_121a", KernelSymbol: "aot_fill", GridDim: [3]uint32{8, 1, 1}, BlockDim: [3]uint32{128, 1, 1}, ClusterDim: [3]uint32{2, 1, 1}, UsePDL: true}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "build", "-o", os.DevNull, writeAOTProgram(t, a))
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}
}

// TestAOTNative compiles a kernel to a fatbin with nvcc, packs it, and runs
// the generated runner on the device. Run it with
//
//	CUDA_EGRAPH_MODEL_NATIVE=1 gpu-lock go test ./x/egraph/model -run TestAOTNative -v
func TestAOTNative(t *testing.T) {
	if os.Getenv(modelNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_MODEL_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	target := os.Getenv("CUDA_EGRAPH_MODEL_NATIVE_ARCH")
	if target == "" {
		target = "sm_121a"
	}
	tmp := t.TempDir()
	cu, fb := filepath.Join(tmp, "fill.cu"), filepath.Join(tmp, "fill.fatbin")
	if err := os.WriteFile(cu, []byte(aotNativeKernel), 0o600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nvcc, "-fatbin", "-arch="+target, cu, "-o", fb).CombinedOutput(); err != nil {
		t.Fatalf("nvcc -fatbin (%s, CUDA %s): %v\n%s", target, cudaVer, err, out)
	}
	fatbin, err := os.ReadFile(fb)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := InspectFatbin(fatbin)
	if err != nil {
		t.Fatalf("inspect nvcc fatbin: %v", err)
	}
	for _, e := range entries {
		t.Logf("nvcc fatbin entry: kind=%d sm=%d compressed=%v bytes=%d", e.Kind, e.SM, e.Compressed, len(e.Payload))
	}
	spec := ArchLaunchSpec{Arch: target, KernelSymbol: "aot_fill", GridDim: [3]uint32{8, 1, 1}, BlockDim: [3]uint32{128, 1, 1}, ClusterDim: [3]uint32{2, 1, 1}, UsePDL: true}
	a, err := PackAOTArtifact("fill", fatbin, []ArchLaunchSpec{spec}, nil)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("go", "run", writeAOTProgram(t, a))
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "AOT_NATIVE_OK") {
		t.Fatalf("aot native program: %v\n%s", err, out)
	}
	t.Logf("aot native passed (arch=%s, CUDA=%s):\n%s", target, cudaVer, strings.TrimSpace(string(out)))
}
