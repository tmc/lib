package numopt

import (
	"fmt"
	"strings"
)

// A KGroup is the K-tiles one sub-loop issues, all in one format.
type KGroup struct {
	Format TilePrecisionFormat
	Tiles  []int // K-tile indices, increasing
}

// PlanKLoop groups the tiles of assign by format, one group per format used,
// in format declaration order. Tile indices are positions in tiles, which is
// taken as the K order.
//
// Grouping reorders the K reduction. The products are the same, but FP32
// accumulation is not associative, so a grouped loop need not match an
// ungrouped one bit for bit.
func PlanKLoop(tiles []TileSensitivity, assign map[string]TilePrecisionFormat, arch string) ([]KGroup, error) {
	if err := Validate(assign, tiles, arch); err != nil {
		return nil, fmt.Errorf("plan k loop: %w", err)
	}
	byFormat := make([][]int, numFormats)
	for i, t := range tiles {
		f := assign[t.TileID]
		byFormat[f] = append(byFormat[f], i)
	}
	var groups []KGroup
	for f, ks := range byFormat {
		if len(ks) > 0 {
			groups = append(groups, KGroup{TilePrecisionFormat(f), ks})
		}
	}
	return groups, nil
}

// CheckGroups reports whether groups issue each of n K-tiles exactly once,
// with no empty group and no format in two groups.
func CheckGroups(groups []KGroup, n int) error {
	seen := make([]bool, n)
	formatSeen := make([]bool, numFormats)
	for _, g := range groups {
		if int(g.Format) >= numFormats {
			return fmt.Errorf("check groups: unknown format %d", g.Format)
		}
		if formatSeen[g.Format] {
			return fmt.Errorf("check groups: %v in two groups", g.Format)
		}
		formatSeen[g.Format] = true
		if len(g.Tiles) == 0 {
			return fmt.Errorf("check groups: empty %v group", g.Format)
		}
		for _, k := range g.Tiles {
			if k < 0 || k >= n {
				return fmt.Errorf("check groups: tile %d outside [0,%d)", k, n)
			}
			if seen[k] {
				return fmt.Errorf("check groups: tile %d issued twice", k)
			}
			seen[k] = true
		}
	}
	for k, ok := range seen {
		if !ok {
			return fmt.Errorf("check groups: tile %d never issued", k)
		}
	}
	return nil
}

// mmaAsm returns the inline-asm template and operand list that issue one
// tcgen05.mma of f for K-tile k, with the accumulate predicate taken from
// %[acc]. The spellings are those github.com/tmc/lib/nvidia/x/tcgen05
// emits for the same kinds.
func mmaAsm(f TilePrecisionFormat) (instr, operands string, err error) {
	const ab = `"r"(d_taddr), "l"(desc_a[k]), "l"(desc_b[k])`
	switch f {
	case PrecBF16:
		return "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;",
			ab + `, "r"(idesc[k])`, nil
	case PrecMXFP8E4M3:
		return "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;",
			ab + `, "r"(idesc[k]), "r"(sfa_taddr[k]), "r"(sfb_taddr[k])`, nil
	case PrecNVFP4ScaleVec4X:
		return "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;",
			ab + `, "r"(idesc[k]), "r"(sfa_taddr[k]), "r"(sfb_taddr[k])`, nil
	case PrecSparse2To4NVFP4:
		return "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, [%3], %4, [%5], [%6], p;",
			ab + `, "r"(meta_taddr[k]), "r"(idesc[k]), "r"(sfa_taddr[k]), "r"(sfb_taddr[k])`, nil
	case PrecRubin3BitLUT:
		return "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32 [%0], %1, %2, [%3], %4, [%5], [%6], p;",
			ab + `, "r"(lut_taddr[k]), "r"(idesc[k]), "r"(sfa_taddr[k]), "r"(sfb_taddr[k])`, nil
	}
	return "", "", fmt.Errorf("no tcgen05.mma spelling for %v in github.com/tmc/lib/nvidia/x/tcgen05", f)
}

// EmitKLoop returns CUDA C++ for a device function, name, that issues the
// tcgen05.mma of every K-tile in groups into one FP32 TMEM accumulator at
// d_taddr, one contiguous loop per group. Only the first MMA issued clears
// the accumulator (enable_input_d = 0); every later one accumulates.
//
// The function takes per-K-tile operand arrays indexed by K-tile: shared
// memory descriptors, instruction descriptors, and the TMEM addresses of
// block scales, 2:4 metadata and LUT centroids, as the format needs. It
// must be called by one elected thread; the caller commits with
// tcgen05.commit and waits before reading the accumulator. Off device it
// compiles to an empty body.
//
// [PrecRubinUE5M3] is refused: no PTX spelling for it is emitted anywhere in
// this repository to check against.
func EmitKLoop(name, arch string, groups []KGroup, numTiles int) (string, error) {
	rubin, err := rubinArch(arch)
	if err != nil {
		return "", fmt.Errorf("emit k loop: %w", err)
	}
	if err := CheckGroups(groups, numTiles); err != nil {
		return "", fmt.Errorf("emit k loop: %w", err)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: mixed-format tcgen05.mma K loop for %s, emitted by numopt.\n", name, arch)
	for gi, g := range groups {
		if formats[g.Format].rubin && !rubin {
			return "", fmt.Errorf("emit k loop: %v requires sm_107a or sm_107f, not %s", g.Format, arch)
		}
		if _, _, err := mmaAsm(g.Format); err != nil {
			return "", fmt.Errorf("emit k loop: %w", err)
		}
		fmt.Fprintf(&b, "__device__ const int %s_g%d[%d] = {", name, gi, len(g.Tiles))
		for i, k := range g.Tiles {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprint(&b, k)
		}
		fmt.Fprintf(&b, "}; // %v\n", g.Format)
	}
	fmt.Fprintf(&b, `__device__ __forceinline__ void %s(
    uint32_t d_taddr, const uint64_t* desc_a, const uint64_t* desc_b, const uint32_t* idesc,
    const uint32_t* sfa_taddr, const uint32_t* sfb_taddr, const uint32_t* meta_taddr, const uint32_t* lut_taddr) {
    (void)sfa_taddr; (void)sfb_taddr; (void)meta_taddr; (void)lut_taddr;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t issued = 0;
`, name)
	for gi, g := range groups {
		instr, ops, _ := mmaAsm(g.Format)
		nops := strings.Count(ops, `"(`) // operand count before the predicate
		fmt.Fprintf(&b, `    for (int i = 0; i < %d; ++i) { // %v
        const int k = %s_g%d[i];
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %%%d, 0;\n"
            "%s\n}\n"
            :: %s, "r"(issued) : "memory");
        ++issued;
    }
`, len(g.Tiles), g.Format, name, gi, nops, instr, ops)
	}
	b.WriteString(`#else
    (void)d_taddr; (void)desc_a; (void)desc_b; (void)idesc;
#endif
}
`)
	return b.String(), nil
}
