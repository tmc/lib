package cudahostemu

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// HazardHeader is the filename of the host hazard-sanitizer C++ header.
const HazardHeader = "hazard.h"

//go:embed hazard.h
var hazardHeaderBytes []byte

// HazardKind classifies a GPU synchronization or uniformity hazard caught on the host CPU.
type HazardKind string

const (
	HazardSharedRace        HazardKind = "shared_memory_race"
	HazardDSMEMBeforeWait   HazardKind = "dsmem_before_cluster_wait"
	HazardWGMMADivergence   HazardKind = "wgmma_warpgroup_divergence"
	HazardNamedBarrierCount HazardKind = "named_barrier_count_mismatch"
)

// Hazard describes a single detected concurrency or uniformity violation.
type Hazard struct {
	Kind    HazardKind
	CTA     int
	Thread  int
	PeerTID int
	Offset  int
	Detail  string
}

// HazardReport summarizes hazards observed during an emulated kernel launch.
type HazardReport struct {
	Hazards []Hazard
}

// Has reports whether r contains at least one hazard of the given kind.
func (r *HazardReport) Has(kind HazardKind) bool {
	if r == nil {
		return false
	}
	for _, h := range r.Hazards {
		if h.Kind == kind {
			return true
		}
	}
	return false
}

// Count returns the number of hazards in r matching kind.
func (r *HazardReport) Count(kind HazardKind) int {
	if r == nil {
		return 0
	}
	n := 0
	for _, h := range r.Hazards {
		if h.Kind == kind {
			n++
		}
	}
	return n
}

// Dim3 is a three-dimensional grid or block extent. Zero components default to 1.
type Dim3 struct {
	X, Y, Z int
}

// Launch configures a kernel launch for CheckHazards.
// Zero Grid, Block, or Cluster dimensions default to 1 (or Cluster for Grid.X
// when Cluster > 1). If Kernel is empty, CheckHazards launches the first
// __global__ kernel defined in src.
type Launch struct {
	Kernel      string   // __global__ function name, or raw C++ launch statement
	Grid        Dim3     // grid dimensions in blocks
	Block       Dim3     // block dimensions in threads
	Cluster     int      // cluster size in blocks along X
	SharedBytes int      // dynamic shared memory bytes per block
	Flags       []string // extra C++ compiler flags
	Mock        bool     // also include sgl_mock.h
}

// Arg describes a kernel argument passed to CheckHazards.
// A buffer argument sets CType (and optionally Name, Data, or Size); a scalar
// argument sets Expr to a C++ literal expression.
type Arg struct {
	Name  string // optional variable name in main
	CType string // element C type for pointer buffers (e.g. "float", "uint32_t")
	Data  []byte // initial buffer contents; if nil, Size zero bytes are allocated
	Size  int    // buffer size in bytes when Data is nil (default 4096)
	Expr  string // C++ expression for scalar arguments (e.g. "64", "1.0f")
}

// BufferArg returns a pointer buffer Arg initialized with data.
func BufferArg(name, ctype string, data []byte) Arg {
	return Arg{Name: name, CType: ctype, Data: data}
}

// ZeroBufferArg returns a zero-initialized pointer buffer Arg of size bytes.
func ZeroBufferArg(name, ctype string, bytes int) Arg {
	return Arg{Name: name, CType: ctype, Size: bytes}
}

// ScalarArg returns a scalar C++ expression Arg.
func ScalarArg(expr string) Arg {
	return Arg{Expr: expr}
}

// IntArg returns a scalar integer Arg.
func IntArg(v int) Arg {
	return Arg{Expr: strconv.Itoa(v)}
}

// CheckHazards compiles and runs src on the host CPU with EG_EMU_HAZARD_CHECK
// enabled and returns all concurrency, barrier, DSMEM, and WGMMA uniformity
// hazards observed during the launch.
func CheckHazards(ctx context.Context, src string, launch Launch, args ...Arg) (*HazardReport, error) {
	dir, err := os.MkdirTemp("", "cudahostemu-hazard-*")
	if err != nil {
		return nil, fmt.Errorf("check hazards temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, HazardHeader), hazardHeaderBytes, 0o644); err != nil {
		return nil, fmt.Errorf("write %s: %w", HazardHeader, err)
	}

	hazardFile := filepath.Join(dir, "hazards.tsv")
	launchStmt, bufs, err := buildHazardLaunch(src, launch, args)
	if err != nil {
		return nil, err
	}

	instrumented := instrumentSharedMemory(src)
	var fullSrc strings.Builder
	fullSrc.WriteString("#define EG_EMU_HAZARD_CHECK 1\n")
	fmt.Fprintf(&fullSrc, "#define EG_EMU_HAZARD_FILE %q\n", hazardFile)
	fullSrc.WriteString("#include \"" + HazardHeader + "\"\n")
	fullSrc.WriteString("#line 1 \"source\"\n")
	fullSrc.WriteString(instrumented)

	flags := append([]string{"-DEG_EMU_HAZARD_CHECK"}, launch.Flags...)
	if _, err := Run(ctx, dir, Program{
		Source:  fullSrc.String(),
		Launch:  launchStmt,
		Buffers: bufs,
		Flags:   flags,
		Mock:    launch.Mock,
	}); err != nil {
		return nil, err
	}

	hazards, err := readHazardFile(hazardFile)
	if err != nil {
		return nil, err
	}
	return &HazardReport{Hazards: hazards}, nil
}

var (
	reGlobalKernel = regexp.MustCompile(`__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)`)
	reExternShared = regexp.MustCompile(`extern\s+__shared__\s+(?:__align__\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_:<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\]\s*;`)
	reStaticShared = regexp.MustCompile(`__shared__\s+(?:__align__\s*\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_:<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\[[^\];]+\])\s*;`)
	reDynSmemCast  = regexp.MustCompile(`([A-Za-z_][A-Za-z0-9_:<>]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:\(\s*[A-Za-z_][A-Za-z0-9_:<>]*\s*\*\s*\)|reinterpret_cast<\s*[A-Za-z_][A-Za-z0-9_:<>]*\s*\*\s*>)\s*\(\s*eg_emu_dynamic_smem\s*\(\s*\)\s*\)\s*;`)
)

func instrumentSharedMemory(src string) string {
	s := reExternShared.ReplaceAllString(src, `eg_emu_smem_ptr<$1> $2(reinterpret_cast<$1*>(eg_emu_dynamic_smem()));`)
	s = reStaticShared.ReplaceAllString(s, `EG_EMU_SHARED($1, $2, $3);`)
	s = reDynSmemCast.ReplaceAllString(s, `eg_emu_smem_ptr<$1> $2(reinterpret_cast<$1*>(eg_emu_dynamic_smem()));`)
	return s
}

func buildHazardLaunch(src string, launch Launch, args []Arg) (string, []Buffer, error) {
	var bufs []Buffer
	var callArgs []string
	for i, a := range args {
		if a.Expr != "" && a.CType == "" && a.Data == nil && a.Size == 0 {
			callArgs = append(callArgs, a.Expr)
			continue
		}
		name := a.Name
		if name == "" {
			name = fmt.Sprintf("cudahostemu_arg_%d", i)
		}
		ctype := a.CType
		if ctype == "" {
			ctype = "uint8_t"
		}
		data := a.Data
		if len(data) == 0 {
			sz := a.Size
			if sz <= 0 {
				sz = 4096
			}
			data = make([]byte, sz)
		}
		bufs = append(bufs, Buffer{Name: name, CType: ctype, Data: data})
		callArgs = append(callArgs, name)
	}

	// If Kernel is already a raw C++ launch statement, use it directly.
	if strings.Contains(launch.Kernel, "(") {
		return launch.Kernel, bufs, nil
	}

	kernelName, params := findKernelSignature(src, launch.Kernel)
	if kernelName == "" {
		return "", nil, fmt.Errorf("cudahostemu: no __global__ kernel found in source")
	}

	// If the caller passed no explicit Args, synthesize default buffers/zeros
	// matching the kernel parameter list so kernels with parameters can be
	// checked without boilerplate buffer setup.
	if len(args) == 0 && len(params) > 0 {
		for i, p := range params {
			if strings.Contains(p, "*") {
				name := fmt.Sprintf("cudahostemu_auto_%d", i)
				ctype := extractBaseCType(p)
				bufs = append(bufs, Buffer{Name: name, CType: ctype, Data: make([]byte, 65536)})
				callArgs = append(callArgs, name)
			} else {
				callArgs = append(callArgs, "{}")
			}
		}
	}

	cluster := launch.Cluster
	if cluster <= 0 {
		cluster = 1
	}
	gx := launch.Grid.X
	if gx <= 0 {
		gx = cluster
	}
	gy := launch.Grid.Y
	if gy <= 0 {
		gy = 1
	}
	gz := launch.Grid.Z
	if gz <= 0 {
		gz = 1
	}
	bx := launch.Block.X
	if bx <= 0 {
		bx = 1
	}
	by := launch.Block.Y
	if by <= 0 {
		by = 1
	}
	bz := launch.Block.Z
	if bz <= 0 {
		bz = 1
	}
	smem := launch.SharedBytes
	if smem < 0 {
		smem = 0
	}

	argSuffix := ""
	if len(callArgs) > 0 {
		argSuffix = ", " + strings.Join(callArgs, ", ")
	}
	if cluster > 1 {
		return fmt.Sprintf("eg_emu_launch_cluster(dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %d, %s%s)",
			gx, gy, gz, bx, by, bz, smem, cluster, kernelName, argSuffix), bufs, nil
	}
	return fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), %d, %s%s)",
		gx, gy, gz, bx, by, bz, smem, kernelName, argSuffix), bufs, nil
}

func findKernelSignature(src, wantName string) (string, []string) {
	matches := reGlobalKernel.FindAllStringSubmatch(src, -1)
	for _, m := range matches {
		name := m[1]
		if wantName != "" && name != wantName {
			continue
		}
		rawParams := strings.TrimSpace(m[2])
		if rawParams == "" || rawParams == "void" {
			return name, nil
		}
		parts := strings.Split(rawParams, ",")
		params := make([]string, 0, len(parts))
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p != "" {
				params = append(params, p)
			}
		}
		return name, params
	}
	return wantName, nil
}

func extractBaseCType(param string) string {
	// Strip default value if any.
	if idx := strings.IndexByte(param, '='); idx >= 0 {
		param = param[:idx]
	}
	// Take everything before the last '*'.
	if idx := strings.LastIndexByte(param, '*'); idx >= 0 {
		param = param[:idx]
	}
	fields := strings.Fields(param)
	var kept []string
	for _, f := range fields {
		switch f {
		case "const", "volatile", "__restrict__", "__restrict", "restrict", "__grid_constant__":
			continue
		default:
			kept = append(kept, f)
		}
	}
	if len(kept) == 0 {
		return "uint8_t"
	}
	return strings.Join(kept, " ")
}

func readHazardFile(path string) ([]Hazard, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read hazard report: %w", err)
	}
	var hazards []Hazard
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 6)
		if len(parts) < 6 {
			continue
		}
		cta, _ := strconv.Atoi(parts[1])
		tid, _ := strconv.Atoi(parts[2])
		peer, _ := strconv.Atoi(parts[3])
		off, _ := strconv.Atoi(parts[4])
		hazards = append(hazards, Hazard{
			Kind:    HazardKind(parts[0]),
			CTA:     cta,
			Thread:  tid,
			PeerTID: peer,
			Offset:  off,
			Detail:  parts[5],
		})
	}
	return hazards, nil
}
