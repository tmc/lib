package model

import (
	"bytes"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/format"
	"maps"
	"slices"
	"strconv"
	"strings"
	"text/template"

	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudatma"
)

// FatbinMagic opens every CUDA fatbin container.
const FatbinMagic = 0xBA55ED50

// Fatbin entry kinds.
const (
	FatbinKindPTX   = 1
	FatbinKindCubin = 2
)

// Fatbin entry flags.
const (
	fatbinFlag64Bit        = 0x1      // compile_size = 64bit
	fatbinFlagHostLinux    = 0x10     // host = linux
	fatbinFlagCompressLZ4  = 0x2000   // older toolkits; not seen from nvcc 13.4
	fatbinFlagCompressZstd = 0x8000   // nvcc 13.4 default for PTX
	fatbinFlagArchSpecific = 0x100000 // sm_NNNa image
)

// A FatbinEntry is one image in a fatbin container.
type FatbinEntry struct {
	Kind int // FatbinKindPTX or FatbinKindCubin
	SM   int // SM number, 121 for sm_121 and sm_121a
	// ArchSpecific marks an sm_NNNa image, which runs only on that SM.
	ArchSpecific bool
	// Major and Minor are the entry's code version: the PTX ISA version
	// for PTX (9.4 from CUDA 13.4), 1.8 for cubins from nvcc 13.4.
	Major, Minor int
	Compressed   bool
	// UncompressedSize is the payload size before compression,
	// or 0 for an uncompressed entry.
	UncompressedSize int
	// Payload is the image: PTX text without trailing NULs, or a cubin.
	// It is empty for a compressed entry; this package does not
	// decompress.
	Payload []byte
}

// The container layout is a 16-byte header {magic u32, version u16,
// header size u16, entries size u64} followed by entries, each a header
// {kind u16, u16, header size u32, payload size u64, compressed size u32,
// u32, minor u16, major u16, sm u32, u32, u32, flags u64, u64,
// uncompressed size u64} of at least 64 bytes and its payload. nvcc 13.4
// writes 80-byte PTX and 112-byte cubin headers whose tail holds options
// this package skips; the field offsets above are checked against its
// output in testdata/nvcc-fill. Several containers may be concatenated.
const (
	fatbinHeaderSize = 16
	fatbinEntrySize  = 64
)

// BuildFatbin writes entries into one fatbin container in order, with
// 64-byte entry headers and no options. Payloads are padded with NULs to
// 8 bytes. A PTX entry's version is read from its .version directive;
// a cubin's is 1.8, as nvcc 13.4 writes. It is deterministic; it does
// not compress. Driver acceptance of its output is not tested here.
func BuildFatbin(entries []FatbinEntry) ([]byte, error) {
	var body bytes.Buffer
	for i, e := range entries {
		if e.Kind != FatbinKindPTX && e.Kind != FatbinKindCubin {
			return nil, fmt.Errorf("build fatbin: entry %d has kind %d", i, e.Kind)
		}
		if e.Compressed {
			return nil, fmt.Errorf("build fatbin: entry %d: compression is not supported", i)
		}
		payload := e.Payload
		major, minor := 1, 8
		if e.Kind == FatbinKindPTX {
			var ok bool
			if major, minor, ok = ptxVersion(payload); !ok {
				return nil, fmt.Errorf("build fatbin: entry %d: PTX has no .version", i)
			}
			payload = append(slices.Clone(payload), 0)
		}
		flags := uint64(fatbinFlag64Bit | fatbinFlagHostLinux)
		if e.ArchSpecific {
			flags |= fatbinFlagArchSpecific
		}
		padded := (len(payload) + 7) &^ 7
		var h [fatbinEntrySize]byte
		binary.LittleEndian.PutUint16(h[0:], uint16(e.Kind))
		binary.LittleEndian.PutUint16(h[2:], 0x0101)
		binary.LittleEndian.PutUint32(h[4:], fatbinEntrySize)
		binary.LittleEndian.PutUint64(h[8:], uint64(padded))
		binary.LittleEndian.PutUint16(h[24:], uint16(minor))
		binary.LittleEndian.PutUint16(h[26:], uint16(major))
		binary.LittleEndian.PutUint32(h[28:], uint32(e.SM))
		binary.LittleEndian.PutUint64(h[40:], flags)
		body.Write(h[:])
		body.Write(payload)
		body.Write(make([]byte, padded-len(payload)))
	}
	out := make([]byte, fatbinHeaderSize, fatbinHeaderSize+body.Len())
	binary.LittleEndian.PutUint32(out[0:], FatbinMagic)
	binary.LittleEndian.PutUint16(out[4:], 1)
	binary.LittleEndian.PutUint16(out[6:], fatbinHeaderSize)
	binary.LittleEndian.PutUint64(out[8:], uint64(body.Len()))
	return append(out, body.Bytes()...), nil
}

// InspectFatbin lists the entries of one or more concatenated fatbin
// containers. Errors carry the byte offset of the bad header.
func InspectFatbin(data []byte) ([]FatbinEntry, error) {
	var entries []FatbinEntry
	for pos := 0; pos < len(data); {
		if len(data)-pos < fatbinHeaderSize {
			return nil, fmt.Errorf("inspect fatbin: truncated container header at offset %d", pos)
		}
		if m := binary.LittleEndian.Uint32(data[pos:]); m != FatbinMagic {
			return nil, fmt.Errorf("inspect fatbin: bad magic %#x at offset %d", m, pos)
		}
		hsize := int(binary.LittleEndian.Uint16(data[pos+6:]))
		size := binary.LittleEndian.Uint64(data[pos+8:])
		if hsize < fatbinHeaderSize || uint64(len(data)-pos-hsize) < size {
			return nil, fmt.Errorf("inspect fatbin: container at offset %d overruns the data", pos)
		}
		start := pos + hsize
		end := start + int(size)
		for p := start; p < end; {
			if end-p < fatbinEntrySize {
				return nil, fmt.Errorf("inspect fatbin: truncated entry header at offset %d", p)
			}
			kind := int(binary.LittleEndian.Uint16(data[p:]))
			ehsize := int(binary.LittleEndian.Uint32(data[p+4:]))
			psize := binary.LittleEndian.Uint64(data[p+8:])
			if ehsize < fatbinEntrySize || uint64(end-p-ehsize) < psize {
				return nil, fmt.Errorf("inspect fatbin: entry at offset %d overruns its container", p)
			}
			if kind != FatbinKindPTX && kind != FatbinKindCubin {
				return nil, fmt.Errorf("inspect fatbin: entry at offset %d has kind %d", p, kind)
			}
			flags := binary.LittleEndian.Uint64(data[p+40:])
			e := FatbinEntry{
				Kind:         kind,
				SM:           int(binary.LittleEndian.Uint32(data[p+28:])),
				ArchSpecific: flags&fatbinFlagArchSpecific != 0,
				Minor:        int(binary.LittleEndian.Uint16(data[p+24:])),
				Major:        int(binary.LittleEndian.Uint16(data[p+26:])),
				Compressed:   flags&(fatbinFlagCompressLZ4|fatbinFlagCompressZstd) != 0,
			}
			if e.Compressed {
				if csize := binary.LittleEndian.Uint32(data[p+16:]); uint64(csize) > psize {
					return nil, fmt.Errorf("inspect fatbin: entry at offset %d has compressed size %d beyond its payload %d", p, csize, psize)
				}
				e.UncompressedSize = int(binary.LittleEndian.Uint64(data[p+56:]))
			} else {
				payload := data[p+ehsize : p+ehsize+int(psize)]
				if kind == FatbinKindPTX {
					payload = bytes.TrimRight(payload, "\x00")
				}
				e.Payload = payload
			}
			entries = append(entries, e)
			p += ehsize + int(psize)
		}
		pos = end
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("inspect fatbin: no entries")
	}
	return entries, nil
}

// An ArchLaunchSpec is the launch geometry and resource budget of the
// megakernel on one SM target.
type ArchLaunchSpec struct {
	Arch              string    `json:"arch"` // "sm_90a", "sm_100a", "sm_121a", ...
	KernelSymbol      string    `json:"kernel_symbol"`
	GridDim           [3]uint32 `json:"grid_dim"`
	BlockDim          [3]uint32 `json:"block_dim"`
	ClusterDim        [3]uint32 `json:"cluster_dim"` // zero means no cluster
	DynamicSharedSize uint32    `json:"dynamic_shared_bytes"`
	ProducerRegs      uint32    `json:"producer_regs"` // setmaxnreg budget, 0 if unused
	ConsumerRegs      uint32    `json:"consumer_regs"`
	TMEMColsUsed      uint16    `json:"tmem_cols_used"`
	UsePDL            bool      `json:"use_pdl"`
}

// smNumber parses "sm_121a" to 121.
func smNumber(arch string) (int, error) {
	s, ok := strings.CutPrefix(arch, "sm_")
	if !ok {
		return 0, fmt.Errorf("arch %q has no sm_ prefix", arch)
	}
	s = strings.TrimRight(s, "af")
	n, err := strconv.Atoi(s)
	if err != nil || n < 10 {
		return 0, fmt.Errorf("arch %q has no SM number", arch)
	}
	return n, nil
}

// maxDynamicShared is the opt-in dynamic shared memory per block.
func maxDynamicShared(sm int) uint32 {
	if sm >= 120 {
		return 101376
	}
	return 232448
}

// Validate checks s against the limits of its target.
func (s ArchLaunchSpec) Validate() error {
	target, err := LookupArchTarget(s.Arch)
	if err != nil {
		return err
	}
	sm, err := smNumber(s.Arch)
	if err != nil {
		return err
	}
	if !isCIdent(s.KernelSymbol) {
		return fmt.Errorf("%s: bad kernel symbol %q", s.Arch, s.KernelSymbol)
	}
	threads := uint64(1)
	for i := range 3 {
		if s.GridDim[i] == 0 || s.BlockDim[i] == 0 {
			return fmt.Errorf("%s: zero grid or block dimension %d", s.Arch, i)
		}
		threads *= uint64(s.BlockDim[i])
	}
	if threads > 1024 {
		return fmt.Errorf("%s: %d threads per block exceed 1024", s.Arch, threads)
	}
	if s.ClusterDim != [3]uint32{} {
		ctas := uint32(1)
		for i := range 3 {
			c := s.ClusterDim[i]
			if c == 0 || s.GridDim[i]%c != 0 {
				return fmt.Errorf("%s: cluster dimension %d (%d) does not divide grid dimension %d", s.Arch, i, c, s.GridDim[i])
			}
			ctas *= c
		}
		if ctas > 16 {
			return fmt.Errorf("%s: cluster of %d CTAs exceeds 16", s.Arch, ctas)
		}
	}
	if m := maxDynamicShared(sm); s.DynamicSharedSize > m {
		return fmt.Errorf("%s: %d bytes of dynamic shared memory exceed %d", s.Arch, s.DynamicSharedSize, m)
	}
	for _, r := range []uint32{s.ProducerRegs, s.ConsumerRegs} {
		if r != 0 && (r < 24 || r > 256 || r%8 != 0) {
			return fmt.Errorf("%s: setmaxnreg count %d is not a multiple of 8 in 24..256", s.Arch, r)
		}
	}
	if c := s.TMEMColsUsed; c != 0 {
		if !target.HasTMEM {
			return fmt.Errorf("%s: has no tensor memory but uses %d columns", s.Arch, c)
		}
		if c < 32 || c > 512 || c&(c-1) != 0 {
			return fmt.Errorf("%s: %d tensor memory columns is not a power of two in 32..512", s.Arch, c)
		}
	}
	return nil
}

// An AOTArtifact is a self-contained ahead-of-time megakernel bundle: a
// fatbin, the launch geometry per target and the TMA descriptor templates
// whose Base the runtime patches.
type AOTArtifact struct {
	ModelName    string                        `json:"model_name"`
	SHA256Digest string                        `json:"sha256_digest"`
	FatbinBytes  []byte                        `json:"-"`
	ArchSpecs    map[string]ArchLaunchSpec     `json:"arch_specs"`
	TMATemplates map[string]cudatma.Descriptor `json:"tma_templates"`
}

// PackAOTArtifact validates and bundles a fatbin with its launch specs and
// TMA templates. Every spec must validate and have an image for its SM in
// the fatbin. An uncompressed PTX image must declare .entry KernelSymbol
// and a .target naming the SM. An uncompressed cubin must be an ELF
// defining KernelSymbol with a zero stack frame (no local-memory spills or
// stack arrays); for ELF ABI version 8 and later the cubin's e_flags SM
// must match. An sm_NNNa image only serves an sm_NNNa spec. One passing
// uncompressed image suffices; compressed images, which are not checked,
// satisfy a spec only when no uncompressed image for its SM exists. Every
// template must validate with Base 0.
func PackAOTArtifact(modelName string, fatbin []byte, specs []ArchLaunchSpec, tma map[string]cudatma.Descriptor) (AOTArtifact, error) {
	if !isCIdent(modelName) {
		return AOTArtifact{}, fmt.Errorf("pack aot artifact: bad model name %q", modelName)
	}
	if len(specs) == 0 {
		return AOTArtifact{}, fmt.Errorf("pack aot artifact: no launch specs")
	}
	entries, err := InspectFatbin(fatbin)
	if err != nil {
		return AOTArtifact{}, fmt.Errorf("pack aot artifact: %w", err)
	}
	a := AOTArtifact{
		ModelName:    modelName,
		FatbinBytes:  slices.Clone(fatbin),
		ArchSpecs:    make(map[string]ArchLaunchSpec),
		TMATemplates: make(map[string]cudatma.Descriptor),
	}
	for _, s := range specs {
		if err := s.Validate(); err != nil {
			return AOTArtifact{}, fmt.Errorf("pack aot artifact: %w", err)
		}
		if _, dup := a.ArchSpecs[s.Arch]; dup {
			return AOTArtifact{}, fmt.Errorf("pack aot artifact: duplicate spec for %s", s.Arch)
		}
		if err := checkImage(entries, s); err != nil {
			return AOTArtifact{}, fmt.Errorf("pack aot artifact: %w", err)
		}
		a.ArchSpecs[s.Arch] = s
	}
	for name, d := range tma {
		if !isCIdent(name) {
			return AOTArtifact{}, fmt.Errorf("pack aot artifact: bad tensor map name %q", name)
		}
		if d.Base != 0 {
			return AOTArtifact{}, fmt.Errorf("pack aot artifact: tensor map %s template has base %#x, want 0", name, d.Base)
		}
		if err := d.Validate(); err != nil {
			return AOTArtifact{}, fmt.Errorf("pack aot artifact: tensor map %s: %w", name, err)
		}
		a.TMATemplates[name] = d
	}
	sum, err := a.digest()
	if err != nil {
		return AOTArtifact{}, fmt.Errorf("pack aot artifact: %w", err)
	}
	a.SHA256Digest = sum
	return a, nil
}

func checkImage(entries []FatbinEntry, s ArchLaunchSpec) error {
	sm, _ := smNumber(s.Arch)
	var errs []string
	compressed := false
	for _, e := range entries {
		if e.SM != sm || e.ArchSpecific && !strings.HasSuffix(s.Arch, "a") {
			continue
		}
		if e.Compressed {
			compressed = true
			continue
		}
		var err error
		if e.Kind == FatbinKindPTX {
			err = checkPTX(e.Payload, s, sm)
		} else {
			err = checkCubin(e.Payload, s, sm)
		}
		if err == nil {
			return nil
		}
		errs = append(errs, err.Error())
	}
	if len(errs) == 0 && compressed {
		return nil
	}
	if len(errs) == 0 {
		return fmt.Errorf("%s: fatbin has no sm_%d image", s.Arch, sm)
	}
	return fmt.Errorf("%s: %s", s.Arch, strings.Join(errs, "; "))
}

// ptxVersion returns the PTX ISA version from ptx's .version directive.
func ptxVersion(ptx []byte) (major, minor int, ok bool) {
	for line := range strings.Lines(string(ptx)) {
		f := strings.Fields(line)
		if len(f) < 2 || f[0] != ".version" {
			continue
		}
		maj, min, found := strings.Cut(f[1], ".")
		a, err1 := strconv.Atoi(maj)
		b, err2 := strconv.Atoi(min)
		return a, b, found && err1 == nil && err2 == nil
	}
	return 0, 0, false
}

func checkPTX(ptx []byte, s ArchLaunchSpec, sm int) error {
	var target, entry bool
	for line := range strings.Lines(string(ptx)) {
		f := strings.Fields(strings.ReplaceAll(line, "(", " ("))
		if len(f) >= 2 && f[0] == ".target" {
			arch := strings.TrimSuffix(f[1], ",")
			if n, err := smNumber(arch); err == nil && n == sm {
				target = true
			}
		}
		for i := 0; i+1 < len(f); i++ {
			if f[i] == ".entry" && f[i+1] == s.KernelSymbol {
				entry = true
			}
		}
	}
	switch {
	case !target:
		return fmt.Errorf("ptx image has no .target sm_%d", sm)
	case !entry:
		return fmt.Errorf("ptx image has no .entry %s", s.KernelSymbol)
	}
	return nil
}

func checkCubin(cubin []byte, s ArchLaunchSpec, sm int) error {
	f, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		return fmt.Errorf("cubin image: %w", err)
	}
	if f.ABIVersion >= 8 {
		if got := cubinSM(cubin); got != sm {
			return fmt.Errorf("cubin image is for sm_%d", got)
		}
	}
	kernels, err := cudaoccupancy.InspectCubin(cubin)
	if err != nil {
		return fmt.Errorf("cubin image: %w", err)
	}
	for _, k := range kernels {
		if k.Name != s.KernelSymbol {
			continue
		}
		if k.StackFrameBytes != 0 {
			return fmt.Errorf("cubin kernel %s has a %d-byte stack frame", k.Name, k.StackFrameBytes)
		}
		return nil
	}
	return fmt.Errorf("cubin image has no kernel %s", s.KernelSymbol)
}

// cubinSM reads the SM from bits 8..15 of e_flags, where CUDA ELF ABI
// version 8 cubins record it.
func cubinSM(cubin []byte) int {
	if len(cubin) < 52 {
		return 0
	}
	return int(binary.LittleEndian.Uint32(cubin[48:]) >> 8 & 0xff)
}

// ManifestJSON returns the artifact's manifest: every field except the
// fatbin, with map keys sorted.
func (a AOTArtifact) ManifestJSON() ([]byte, error) {
	return json.MarshalIndent(a, "", "\t")
}

// digest hashes the fatbin and the manifest with an empty digest field.
func (a AOTArtifact) digest() (string, error) {
	a.SHA256Digest = ""
	m, err := a.ManifestJSON()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	h.Write(a.FatbinBytes)
	h.Write(m)
	return hex.EncodeToString(h.Sum(nil)), nil
}

func (a AOTArtifact) sortedSpecs() []ArchLaunchSpec {
	var out []ArchLaunchSpec
	for _, k := range slices.Sorted(maps.Keys(a.ArchSpecs)) {
		out = append(out, a.ArchSpecs[k])
	}
	return out
}

type namedTemplate struct {
	Name     string
	D        cudatma.Descriptor
	DataType int
}

func (a AOTArtifact) sortedTemplates() ([]namedTemplate, error) {
	var out []namedTemplate
	for _, k := range slices.Sorted(maps.Keys(a.TMATemplates)) {
		d := a.TMATemplates[k]
		dt, err := tensorMapDataType(d)
		if err != nil {
			return nil, fmt.Errorf("tensor map %s: %w", k, err)
		}
		out = append(out, namedTemplate{k, d, dt})
	}
	return out, nil
}

// tensorMapDataType mirrors cudatma's choice of CUtensorMapDataType.
func tensorMapDataType(d cudatma.Descriptor) (int, error) {
	const f16, f32, bf16 = 6, 7, 9
	switch d.ElementType {
	case cudatma.ElementTypeDefault:
		switch d.ElementBytes {
		case 2:
			return f16, nil
		case 4:
			return f32, nil
		}
	case cudatma.ElementTypeFloat16:
		return f16, nil
	case cudatma.ElementTypeBfloat16:
		return bf16, nil
	case cudatma.ElementTypeFloat32:
		return f32, nil
	}
	return 0, fmt.Errorf("no tensor map data type for %v with %d-byte elements", d.ElementType, d.ElementBytes)
}

// EmitGoEmbedPackage returns gofmt-formatted Go source for a package
// pkgName that embeds fatbinFileName (which the caller writes next to it
// with a.FatbinBytes) and exposes Load, Runner.TensorMap and Runner.Launch.
// The generated package imports only the standard library and
// github.com/tmc/lib/nvidia/driver, which loads libcuda without cgo; it
// does not import nvidia/x.
func (a AOTArtifact) EmitGoEmbedPackage(pkgName, fatbinFileName string) (string, error) {
	if !isCIdent(pkgName) || strings.ToLower(pkgName) != pkgName {
		return "", fmt.Errorf("emit go embed package: bad package name %q", pkgName)
	}
	if fatbinFileName == "" || strings.ContainsAny(fatbinFileName, " \t\n\"`*?[") || strings.HasPrefix(fatbinFileName, ".") {
		return "", fmt.Errorf("emit go embed package: bad fatbin file name %q", fatbinFileName)
	}
	tmpls, err := a.sortedTemplates()
	if err != nil {
		return "", fmt.Errorf("emit go embed package: %w", err)
	}
	var buf bytes.Buffer
	err = goEmbedTemplate.Execute(&buf, map[string]any{
		"Pkg":       pkgName,
		"File":      fatbinFileName,
		"A":         a,
		"Specs":     a.sortedSpecs(),
		"Templates": tmpls,
	})
	if err != nil {
		return "", fmt.Errorf("emit go embed package: %w", err)
	}
	src, err := format.Source(buf.Bytes())
	if err != nil {
		return "", fmt.Errorf("emit go embed package: format: %w", err)
	}
	return string(src), nil
}

var goEmbedTemplate = template.Must(template.New("go").Parse(`// Code generated by model.EmitGoEmbedPackage. DO NOT EDIT.

// Package {{.Pkg}} embeds the {{.A.ModelName}} ahead-of-time megakernel
// bundle and launches it through the CUDA Driver API.
package {{.Pkg}}

import (
	_ "embed"
	"fmt"
	"strconv"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Fatbin is the embedded multi-architecture fatbin.
//
//go:embed {{.File}}
var Fatbin []byte

// SHA256 is the bundle digest recorded at packaging time.
const SHA256 = "{{.A.SHA256Digest}}"

// A LaunchSpec is the launch geometry for one SM target.
type LaunchSpec struct {
	Arch               string
	KernelSymbol       string
	GridDim            [3]uint32
	BlockDim           [3]uint32
	ClusterDim         [3]uint32
	DynamicSharedBytes uint32
	UsePDL             bool
}

// Specs lists the launch specs by arch.
var Specs = []LaunchSpec{
{{- range .Specs}}
	{Arch: "{{.Arch}}", KernelSymbol: "{{.KernelSymbol}}", GridDim: [3]uint32{ {{- index .GridDim 0}}, {{index .GridDim 1}}, {{index .GridDim 2 -}} }, BlockDim: [3]uint32{ {{- index .BlockDim 0}}, {{index .BlockDim 1}}, {{index .BlockDim 2 -}} }, ClusterDim: [3]uint32{ {{- index .ClusterDim 0}}, {{index .ClusterDim 1}}, {{index .ClusterDim 2 -}} }, DynamicSharedBytes: {{.DynamicSharedSize}}, UsePDL: {{.UsePDL}}},
{{- end}}
}

// A TensorMapTemplate is a tiled tensor map with its global address left
// for [Runner.TensorMap] to fill in.
type TensorMapTemplate struct {
	Name        string
	DataType    driver.TensorMapDataType
	Rank        uint32
	Dims        [5]uint64
	Strides     [5]uint64
	Box         [5]uint32
	Interleave  driver.TensorMapInterleave
	Swizzle     driver.TensorMapSwizzle
	L2Promotion driver.TensorMapL2promotion
	OOBFill     driver.TensorMapFloatOOBfill
}

// TensorMaps lists the tensor map templates by name.
var TensorMaps = []TensorMapTemplate{
{{- range .Templates}}
	{Name: "{{.Name}}", DataType: {{.DataType}}, Rank: {{.D.Rank}}, Dims: [5]uint64{ {{- range $i, $v := .D.Dims}}{{if $i}}, {{end}}{{$v}}{{end -}} }, Strides: [5]uint64{ {{- range $i, $v := .D.Strides}}{{if $i}}, {{end}}{{$v}}{{end -}} }, Box: [5]uint32{ {{- range $i, $v := .D.Box}}{{if $i}}, {{end}}{{$v}}{{end -}} }, Interleave: {{printf "%d" .D.Interleave}}, Swizzle: {{if eq .D.Swizzle 32}}1{{else if eq .D.Swizzle 64}}2{{else if eq .D.Swizzle 128}}3{{else}}0{{end}}, L2Promotion: {{printf "%d" .D.L2Promotion}}, OOBFill: {{printf "%d" .D.OOBFill}}},
{{- end}}
}

// Select returns the spec for a device of compute capability
// major.minor: an exact sm_XY or sm_XYa image first, then an sm_XZf
// family image with the same major and Z <= minor.
func Select(major, minor int) (LaunchSpec, error) {
	want := "sm_" + strconv.Itoa(major) + strconv.Itoa(minor)
	for _, s := range Specs {
		if s.Arch == want || s.Arch == want+"a" {
			return s, nil
		}
	}
	for _, s := range Specs {
		n, ok := strings.CutSuffix(strings.TrimPrefix(s.Arch, "sm_"), "f")
		if !ok || len(n) < 2 {
			continue
		}
		maj, err1 := strconv.Atoi(n[:len(n)-1])
		min, err2 := strconv.Atoi(n[len(n)-1:])
		if err1 == nil && err2 == nil && maj == major && min <= minor {
			return s, nil
		}
	}
	return LaunchSpec{}, fmt.Errorf("{{.Pkg}}: no image for compute capability %d.%d", major, minor)
}

// A Runner holds the loaded module and kernel for one device.
type Runner struct {
	Spec     LaunchSpec
	Module   driver.Module
	Function driver.Function
}

// Load selects the spec for dev, loads [Fatbin] into the current context
// and resolves the kernel.
func Load(dev driver.Device) (*Runner, error) {
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("{{.Pkg}}: compute capability: %w", err)
	}
	spec, err := Select(major, minor)
	if err != nil {
		return nil, err
	}
	mod, err := driver.ModuleLoadData(Fatbin)
	if err != nil {
		return nil, fmt.Errorf("{{.Pkg}}: load module: %w", err)
	}
	fn, err := mod.Function(spec.KernelSymbol)
	if err != nil {
		return nil, fmt.Errorf("{{.Pkg}}: function %s: %w", spec.KernelSymbol, err)
	}
	if spec.DynamicSharedBytes > 48<<10 {
		if err := fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, int32(spec.DynamicSharedBytes)); err != nil {
			return nil, fmt.Errorf("{{.Pkg}}: dynamic shared memory: %w", err)
		}
	}
	if spec.ClusterDim[0]*spec.ClusterDim[1]*spec.ClusterDim[2] > 8 {
		if err := fn.SetAttribute(driver.FuncAttributeNonPortableClusterSizeAllowed, 1); err != nil {
			return nil, fmt.Errorf("{{.Pkg}}: non-portable cluster size: %w", err)
		}
	}
	return &Runner{Spec: spec, Module: mod, Function: fn}, nil
}

// TensorMap encodes the named template with global address base.
func (r *Runner) TensorMap(name string, base driver.DevicePtr) (*driver.TensorMap, error) {
	for _, t := range TensorMaps {
		if t.Name != name {
			continue
		}
		m := driver.NewTensorMap()
		elems := [5]uint32{1, 1, 1, 1, 1}
		if err := driver.TensorMapEncodeTiled(m, t.DataType, t.Rank, base, &t.Dims[0], &t.Strides[0], &t.Box[0], &elems[0], t.Interleave, t.Swizzle, t.L2Promotion, t.OOBFill); err != nil {
			return nil, fmt.Errorf("{{.Pkg}}: encode tensor map %s: %w", name, err)
		}
		return m, nil
	}
	return nil, fmt.Errorf("{{.Pkg}}: no tensor map %q", name)
}

// Launch launches the kernel on stream with the given parameters.
func (r *Runner) Launch(stream driver.Stream, params []unsafe.Pointer) error {
	s := r.Spec
	cfg := driver.LaunchConfig{
		GridDim:        driver.Dims(s.GridDim[0], s.GridDim[1], s.GridDim[2]),
		BlockDim:       driver.Dims(s.BlockDim[0], s.BlockDim[1], s.BlockDim[2]),
		SharedMemBytes: s.DynamicSharedBytes,
		HStream:        stream,
	}
	if s.ClusterDim != [3]uint32{} {
		var at driver.LaunchAttribute
		at.Id = driver.LaunchAttributeClusterDimension
		*at.Value.ClusterDim() = driver.Dims(s.ClusterDim[0], s.ClusterDim[1], s.ClusterDim[2])
		cfg.Attrs = append(cfg.Attrs, at)
	}
	if s.UsePDL {
		var at driver.LaunchAttribute
		at.Id = driver.LaunchAttributeProgrammaticStreamSerialization
		*at.Value.ProgrammaticStreamSerializationAllowed() = 1
		cfg.Attrs = append(cfg.Attrs, at)
	}
	if err := r.Function.LaunchKernelEx(&cfg, params); err != nil {
		return fmt.Errorf("{{.Pkg}}: launch %s: %w", s.KernelSymbol, err)
	}
	return nil
}
`))

// EmitCHeader returns a self-contained C header (C99 plus a compiler
// alignment attribute) holding the fatbin bytes, the launch specs and the
// tensor map templates. guardPrefix, an upper-case C identifier, prefixes
// the include guard and every declared name. The templates are fields for
// cuTensorMapEncodeTiled; the header does not encode them.
func (a AOTArtifact) EmitCHeader(guardPrefix string) (string, error) {
	if !isCIdent(guardPrefix) || strings.ToUpper(guardPrefix) != guardPrefix {
		return "", fmt.Errorf("emit c header: bad guard prefix %q", guardPrefix)
	}
	tmpls, err := a.sortedTemplates()
	if err != nil {
		return "", fmt.Errorf("emit c header: %w", err)
	}
	p, lp := guardPrefix, strings.ToLower(guardPrefix)
	var sb strings.Builder
	fmt.Fprintf(&sb, `/* Code generated by model.EmitCHeader. DO NOT EDIT. */
/* %s ahead-of-time megakernel bundle, sha256 %s */
#ifndef %s_H
#define %s_H

#include <stdint.h>

#if defined(_MSC_VER)
#define %s_ALIGN(n) __declspec(align(n))
#else
#define %s_ALIGN(n) __attribute__((aligned(n)))
#endif

#define %s_SHA256 "%s"
#define %s_FATBIN_SIZE %d
#define %s_NUM_SPECS %d
#define %s_NUM_TENSOR_MAPS %d

typedef struct {
	const char *arch;
	const char *kernel_symbol;
	uint32_t grid_dim[3];
	uint32_t block_dim[3];
	uint32_t cluster_dim[3];
	uint32_t dynamic_shared_bytes;
	uint32_t producer_regs;
	uint32_t consumer_regs;
	uint16_t tmem_cols_used;
	uint8_t use_pdl;
} %s_launch_spec;

/* Fields for cuTensorMapEncodeTiled; the global address is supplied at load time. */
typedef struct {
	const char *name;
	uint32_t data_type;
	uint32_t rank;
	uint64_t dims[5];
	uint64_t strides[5];
	uint32_t box[5];
	uint32_t interleave;
	uint32_t swizzle;
	uint32_t l2_promotion;
	uint32_t oob_fill;
} %s_tensor_map_template;

`, a.ModelName, a.SHA256Digest, p, p, p, p, p, a.SHA256Digest, p, len(a.FatbinBytes), p, len(a.ArchSpecs), p, len(tmpls), lp, lp)
	fmt.Fprintf(&sb, "static const %s_ALIGN(128) unsigned char %s_fatbin[%d] = {", p, lp, max(len(a.FatbinBytes), 1))
	for i, b := range a.FatbinBytes {
		if i%16 == 0 {
			sb.WriteString("\n\t")
		}
		fmt.Fprintf(&sb, "0x%02x,", b)
	}
	sb.WriteString("\n};\n\n")
	u32s := func(v [3]uint32) string { return fmt.Sprintf("{%d, %d, %d}", v[0], v[1], v[2]) }
	fmt.Fprintf(&sb, "static const %s_launch_spec %s_specs[%d] = {\n", lp, lp, max(len(a.ArchSpecs), 1))
	for _, s := range a.sortedSpecs() {
		pdl := 0
		if s.UsePDL {
			pdl = 1
		}
		fmt.Fprintf(&sb, "\t{%q, %q, %s, %s, %s, %d, %d, %d, %d, %d},\n",
			s.Arch, s.KernelSymbol, u32s(s.GridDim), u32s(s.BlockDim), u32s(s.ClusterDim),
			s.DynamicSharedSize, s.ProducerRegs, s.ConsumerRegs, s.TMEMColsUsed, pdl)
	}
	sb.WriteString("};\n\n")
	fmt.Fprintf(&sb, "static const %s_ALIGN(64) %s_tensor_map_template %s_tensor_maps[%d] = {\n", p, lp, lp, max(len(tmpls), 1))
	for _, t := range tmpls {
		d := t.D
		sw := map[uint32]int{0: 0, 32: 1, 64: 2, 128: 3}[d.Swizzle]
		fmt.Fprintf(&sb, "\t{%q, %d, %d, {%d, %d, %d, %d, %d}, {%d, %d, %d, %d, %d}, {%d, %d, %d, %d, %d}, %d, %d, %d, %d},\n",
			t.Name, t.DataType, d.Rank,
			d.Dims[0], d.Dims[1], d.Dims[2], d.Dims[3], d.Dims[4],
			d.Strides[0], d.Strides[1], d.Strides[2], d.Strides[3], d.Strides[4],
			d.Box[0], d.Box[1], d.Box[2], d.Box[3], d.Box[4],
			d.Interleave, sw, d.L2Promotion, d.OOBFill)
	}
	fmt.Fprintf(&sb, "};\n\n#endif /* %s_H */\n", p)
	return sb.String(), nil
}
