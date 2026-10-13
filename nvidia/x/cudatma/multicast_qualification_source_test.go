package cudatma

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func qualificationReplace(source, old, new string) (string, error) {
	if strings.Count(source, old) != 1 {
		return "", fmt.Errorf("fixture source seam must occur once: %q", old)
	}
	return strings.Replace(source, old, new, 1), nil
}

func qualificationSource(c qualificationCase) (string, error) {
	p, e := NewMulticastPlan(c.config())
	if e != nil {
		return "", e
	}
	r, _ := p.Requirements()
	source, e := p.Source()
	if e != nil {
		return "", e
	}
	// An extra diagnostic uses the status record's otherwise trailing padding.
	// Protocol and byte-copy instructions are unchanged. Observe actual shared
	// alignment rather than treating a host reservation as physical proof.
	replacements := [][2]string{{"rank, selected, phase, bytes, retired;", "rank, selected, phase, bytes, retired, tile_address;"}, {fmt.Sprintf("selected ? %du : 0u, 1};", r.TileBytes), fmt.Sprintf("selected ? %du : 0u, 1, tile};", r.TileBytes)}}
	for _, row := range replacements {
		source, e = qualificationReplace(source, row[0], row[1])
		if e != nil {
			return "", e
		}
	}
	start := strings.Index(source, "extern \"C\"")
	if start < 0 {
		return "", fmt.Errorf("kernel declaration absent")
	}
	ordinary := source[start:]
	for _, row := range [][2]string{{"tma_multicast(", "tma_ordinary("}, {fmt.Sprintf("if (rank == %d && threadIdx.x == 0) {\n    asm volatile(\"cp.async.bulk.tensor", p.config.SourceRank), "if (selected && threadIdx.x == 0) {\n    asm volatile(\"cp.async.bulk.tensor"}, {".shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster", ".shared::cta.global.tile.mbarrier::complete_tx::bytes"}, {fmt.Sprintf("}], [%%%d], %%%d;", c.Rank+2, c.Rank+3), fmt.Sprintf("}], [%%%d];", c.Rank+2)}, {`, "h"(mask) : "memory"`, ` : "memory"`}} {
		ordinary, e = qualificationReplace(ordinary, row[0], row[1])
		if e != nil {
			return "", e
		}
	}
	// Independent CUDA consumer uses the primary table's pattern-line/cell
	// relation, not the production i>>7 XOR expression.
	old := fmt.Sprintf("const uint64_t physical = i ^ (((i >> 7) & %du) << 4);", swizzleMask(c.Swizzle))
	expression := "const uint64_t physical = i;"
	if c.Swizzle != 0 {
		expression = fmt.Sprintf("const uint64_t physical = (i / 128) * 128 + 16 * ((i / 16 %% 8) ^ (i / 128 %% %d)) + i %% 16;", c.Swizzle/16)
	}
	ordinary, e = qualificationReplace(ordinary, old, expression)
	if e != nil {
		return "", e
	}
	assertions := `#include <stddef.h>
static_assert(sizeof(CUtensorMap) == 128, "tensor map ABI");
static_assert(sizeof(multicast_status) == 32, "status ABI");
static_assert(offsetof(multicast_status, generation) == 0, "generation ABI");
static_assert(offsetof(multicast_status, rank) == 8, "rank ABI");
static_assert(offsetof(multicast_status, selected) == 12, "selected ABI");
static_assert(offsetof(multicast_status, phase) == 16, "phase ABI");
static_assert(offsetof(multicast_status, bytes) == 20, "bytes ABI");
static_assert(offsetof(multicast_status, retired) == 24, "retirement ABI");
static_assert(offsetof(multicast_status, tile_address) == 28, "shared address ABI");
`
	if c.Control == "single-destination" {
		// Truthful single-CTA completion: remote output remains a canary. This
		// completed control changes participation as well as the qualifier; it is
		// not mislabeled an isolated qualifier mutant with an unbounded wait.
		for _, row := range [][2]string{{"const bool selected = (mask & (uint16_t(1) << rank)) != 0;", "const bool selected = rank == 0;"}, {fmt.Sprintf("  if (rank == 1 && threadIdx.x == 0) {\n    asm volatile(\"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%0], %%1;\" :: \"r\"(bar), \"r\"(%du) : \"memory\");\n  }\n", r.TileBytes), ""}, {".multicast::cluster", ""}, {fmt.Sprintf("}], [%%%d], %%%d;", c.Rank+2, c.Rank+3), fmt.Sprintf("}], [%%%d];", c.Rank+2)}, {`, "h"(mask) : "memory"`, ` : "memory"`}} {
			source, e = qualificationReplace(source, row[0], row[1])
			if e != nil {
				return "", e
			}
		}
	}
	if c.Parity {
		body := source[start:]
		for _, row := range [][2]string{{"tma_multicast(", "tma_parity("}, {"__grid_constant__ const CUtensorMap map,", "__grid_constant__ const CUtensorMap map0, __grid_constant__ const CUtensorMap map1,"}, {"uint64_t generation,", "uint64_t generation0, uint64_t generation1,"}, {"  if (rank == 0 && threadIdx.x == 0) {\n    asm volatile(\"mbarrier.arrive.expect_tx", "  for (unsigned phase = 0; phase < 2; ++phase) {\n  if (rank == 0 && threadIdx.x == 0) {\n    asm volatile(\"mbarrier.arrive.expect_tx"}, {`"l"(&map)`, `"l"(phase == 0 ? &map0 : &map1)`}, {`"r"(bar), "r"(0)`, `"r"(bar), "r"(phase)`}, {"output[uint64_t(rank) *", "output[(uint64_t(phase) * 2 + rank) *"}, {"status[rank] = {generation, rank, uint32_t(selected), 0,", "status[phase * 2 + rank] = {phase == 0 ? generation0 : generation1, rank, uint32_t(selected), phase,"}} {
			// Remote-only parity has no rank0 expectation; insert the loop before
			// the first selected expectation, regardless of mask.
			if strings.HasPrefix(row[0], "  if (rank == 0") && c.Mask == 2 {
				row[0] = strings.Replace(row[0], "rank == 0", "rank == 1", 1)
				row[1] = strings.Replace(row[1], "rank == 0", "rank == 1", 1)
			}
			body, e = qualificationReplace(body, row[0], row[1])
			if e != nil {
				return "", e
			}
		}
		if !strings.HasSuffix(body, "}\n") {
			return "", fmt.Errorf("parity function terminator absent")
		}
		body = strings.TrimSuffix(body, "}\n") + "  } // both maps retained through successful stream completion\n}\n"
		source = source[:start] + body
	}
	return source + "\n" + assertions + "\n" + ordinary, nil
}
func swizzleMask(sw uint32) uint32 {
	if sw == 0 {
		return 0
	}
	return sw/16 - 1
}

func qualificationPTX(text string, c qualificationCase) error {
	mod, e := cudaptx.Parse(strings.NewReader(text))
	if e != nil {
		return e
	}
	if mod.Target != TargetSM100a || mod.AddressSize != 64 {
		return fmt.Errorf("PTX target/address ABI differs from exact sm_100a admission")
	}
	name := "tma_multicast"
	maps := 1
	if c.Parity {
		name = "tma_parity"
		maps = 2
	}
	for _, kernel := range []struct {
		name string
		maps int
	}{{name, maps}, {"tma_ordinary", 1}} {
		k := mod.Kernel(kernel.name)
		if k == nil {
			return fmt.Errorf("PTX kernel %s absent", kernel.name)
		}
		expected := kernel.maps + 2 + kernel.maps + c.Rank
		if len(k.Params) != expected {
			return fmt.Errorf("PTX %s parameter count %d want %d", kernel.name, len(k.Params), expected)
		}
		for i, p := range k.Params {
			size := ptxFixtureBytes(p.Type, p.Size)
			want := 4
			if i < kernel.maps {
				want = 128
				if p.Align == 0 || p.Align > 128 {
					return fmt.Errorf("tensor map alignment absent/unsupported")
				}
			} else if i < kernel.maps+2+kernel.maps {
				want = 8
			}
			if size != want {
				return fmt.Errorf("PTX %s parameter%d size%d want%d", kernel.name, i, size, want)
			}
		}
	}
	for _, row := range []struct {
		name      string
		maps      int
		multicast bool
	}{{name, maps, c.Control != "single-destination"}, {"tma_ordinary", 1, false}} {
		k := mod.Kernel(row.name)
		if e := qualificationKernelPTX(text, k, c, row.maps, row.multicast); e != nil {
			return fmt.Errorf("kernel %s: %w", row.name, e)
		}
	}
	// This inspection remains a compiler admission gate, not numerical or
	// hardware qualification. Exact sources and raw PTX are retained for review.
	for _, token := range []string{"cvta.param", "fence.mbarrier_init.release.cluster", "mbarrier.arrive.expect_tx", "mbarrier.try_wait.parity", "barrier.cluster"} {
		if !strings.Contains(text, token) {
			return fmt.Errorf("PTX missing %s", token)
		}
	}
	if c.Control != "single-destination" && !strings.Contains(text, ".multicast::cluster") {
		return fmt.Errorf("PTX multicast instruction absent")
	}
	if strings.Contains(text, ".local .align 64 .b8") || strings.Contains(text, ".local .align 128 .b8") {
		return fmt.Errorf("local tensor-map-sized storage requires independent inspection")
	}
	return nil
}
func ptxFixtureBytes(typ string, n int) int {
	size := 0
	switch typ {
	case "b8", "u8", "s8":
		size = 1
	case "b32", "u32", "s32", "f32":
		size = 4
	case "b64", "u64", "s64":
		size = 8
	}
	if n > 0 {
		size *= n
	}
	return size
}

func TestQualificationRealSources(t *testing.T) {
	for _, c := range qualificationCases() {
		s, e := qualificationSource(c)
		if e != nil {
			t.Fatalf("%s: %v", c.ID, e)
		}
		if !strings.Contains(s, "tma_ordinary(") || !strings.Contains(s, "sizeof(multicast_status) == 32") {
			t.Fatal("reference/ABI missing", c.ID)
		}
		if c.Control != "single-destination" && strings.Count(s, ".multicast::cluster") != 1 {
			t.Fatal("real multicast opcode absent", c.ID)
		}
		if c.Parity {
			for _, want := range []string{"phase < 2", `"r"(phase)`, "map0", "map1", "status[phase * 2 + rank]", "cluster.sync(); // retirement"} {
				if !strings.Contains(s, want) {
					t.Fatal("parity sequence missing", c.ID, want)
				}
			}
		}
	}
	if _, e := qualificationReplace("a a", "a", "b"); e == nil {
		t.Fatal("ambiguous source rewrite accepted")
	}
}

// This is a restricted inspection of the prepared inline-asm ABI and map
// address origins. Unsupported compiler lowering is refused for independent
// inspection, not treated as proof. It does not prove arbitrary PTX control
// flow or synchronization on a device.
func qualificationKernelPTX(text string, k *cudaptx.Kernel, c qualificationCase, maps int, multicast bool) error {
	lines := strings.Split(text, "\n")
	if k.StartLine < 1 || k.EndLine > len(lines) {
		return fmt.Errorf("PTX body bounds unavailable")
	}
	body := strings.Join(lines[k.StartLine-1:k.EndLine], "\n")
	declarations := regexp.MustCompile(`\.reg\s+\.(?:b|u|s)(16|32|64)\s+(%[A-Za-z_][A-Za-z_0-9]*)(?:<([0-9]+)>)?`)
	type reg struct {
		prefix       string
		width, count int
	}
	var regs []reg
	for _, m := range declarations.FindAllStringSubmatch(body, -1) {
		width, _ := strconv.Atoi(m[1])
		count := 0
		if m[3] != "" {
			count, _ = strconv.Atoi(m[3])
		}
		regs = append(regs, reg{m[2], width, count})
	}
	widthOf := func(token string) int {
		for _, r := range regs {
			if r.count == 0 && token == r.prefix {
				return r.width
			}
			if r.count > 0 && strings.HasPrefix(token, r.prefix) {
				i, e := strconv.Atoi(strings.TrimPrefix(token, r.prefix))
				if e == nil && i >= 0 && i < r.count {
					return r.width
				}
			}
		}
		return 0
	}
	paramBits := map[string]uint32{}
	for i := 0; i < maps; i++ {
		paramBits[k.Params[i].Name] = 1 << i
	}
	origins := map[string]uint32{}
	convertedOrigins := map[string]uint32{}
	constants := map[string]uint64{}
	converted := uint32(0)
	used := uint32(0)
	origin := func(s string) uint32 {
		s = strings.TrimSpace(s)
		if b := paramBits[s]; b != 0 {
			return b
		}
		return origins[s]
	}
	opcode := fmt.Sprintf("cp.async.bulk.tensor.%dd.shared::", c.Rank)
	operands := regexp.MustCompile(`^\s*\[(%[A-Za-z_0-9]+)\],\s*\[(%[A-Za-z_0-9]+),\s*\{([^}]*)\}\],\s*\[(%[A-Za-z_0-9]+)\](?:,\s*(%[A-Za-z_0-9]+))?\s*$`)
	seenIssue := false
	initIndex, fenceIndex, expectIndex, issueIndex, waitIndex, retireIndex := -1, -1, -1, -1, -1, -1
	for index, inst := range k.Instructions {
		fields := strings.Split(inst.Operands, ",")
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		if strings.HasPrefix(inst.Opcode, "mov.") && len(fields) == 2 {
			origins[fields[0]] = origin(fields[1])
			convertedOrigins[fields[0]] = convertedOrigins[fields[1]]
			if value, e := strconv.ParseUint(fields[1], 0, 64); e == nil {
				constants[fields[0]] = value
			} else {
				delete(constants, fields[0])
			}
		}
		if strings.HasPrefix(inst.Opcode, "cvta.param.") && len(fields) == 2 {
			origins[fields[0]] = origin(fields[1])
			convertedOrigins[fields[0]] = origins[fields[0]]
			converted |= origins[fields[0]]
		}
		if strings.HasPrefix(inst.Opcode, "selp.") && len(fields) >= 3 {
			origins[fields[0]] = origin(fields[1]) | origin(fields[2])
			convertedOrigins[fields[0]] = convertedOrigins[fields[1]] | convertedOrigins[fields[2]]
		}
		if len(fields) > 0 && widthOf(fields[0]) != 0 && !strings.HasPrefix(inst.Opcode, "mov.") && !strings.HasPrefix(inst.Opcode, "cvta.param.") && !strings.HasPrefix(inst.Opcode, "selp.") {
			delete(origins, fields[0])
			delete(convertedOrigins, fields[0])
			delete(constants, fields[0])
		}
		switch {
		case strings.HasPrefix(inst.Opcode, "mbarrier.init."):
			if initIndex < 0 {
				initIndex = index
			}
		case strings.HasPrefix(inst.Opcode, "fence.mbarrier_init.release.cluster"):
			if fenceIndex < 0 {
				fenceIndex = index
			}
		case strings.HasPrefix(inst.Opcode, "mbarrier.arrive.expect_tx."):
			if expectIndex < 0 {
				expectIndex = index
			}
			if len(fields) != 3 {
				return fmt.Errorf("expect_tx operand count mismatch")
			}
			count, e := strconv.ParseUint(fields[2], 0, 64)
			if e != nil {
				count = constants[fields[2]]
			}
			tileBytes := uint64(c.Width)
			for axis := 1; axis < c.Rank; axis++ {
				tileBytes *= uint64(c.config().Descriptor.Box[axis])
			}
			if count != tileBytes || count > maxMulticastTransactionBytes {
				return fmt.Errorf("expect_tx byte count mismatch")
			}
		case strings.HasPrefix(inst.Opcode, "mbarrier.try_wait.parity.acquire.cta"):
			if waitIndex < 0 {
				waitIndex = index
			}
		case strings.HasPrefix(inst.Opcode, "barrier.cluster."):
			if seenIssue {
				retireIndex = index
			}
		}
		if !strings.HasPrefix(inst.Opcode, opcode) {
			continue
		}
		expectedSpace := "cluster"
		if k.Name == "tma_ordinary" {
			expectedSpace = "cta"
		}
		expected := fmt.Sprintf("cp.async.bulk.tensor.%dd.shared::%s.global.tile.mbarrier::complete_tx::bytes", c.Rank, expectedSpace)
		if multicast {
			expected += ".multicast::cluster"
		}
		if inst.Opcode != expected {
			return fmt.Errorf("unexpected tensor opcode %s", inst.Opcode)
		}
		m := operands.FindStringSubmatch(inst.Operands)
		if m == nil {
			return fmt.Errorf("tensor operands not inspectable")
		}
		coords := strings.Split(m[3], ",")
		if len(coords) != c.Rank || widthOf(m[1]) != 32 || widthOf(m[2]) != 64 || widthOf(m[4]) != 32 {
			return fmt.Errorf("tensor tile/map/barrier/coordinate operand ABI mismatch")
		}
		for _, coordinate := range coords {
			if widthOf(strings.TrimSpace(coordinate)) != 32 {
				return fmt.Errorf("coordinate is not a32-bit register")
			}
		}
		if multicast {
			if widthOf(m[5]) != 16 || constants[m[5]] != uint64(c.Mask) {
				return fmt.Errorf("actual multicast mask width/value mismatch")
			}
		} else if m[5] != "" {
			return fmt.Errorf("ordinary tensor unexpectedly has a mask")
		}
		if origins[m[2]] == 0 || convertedOrigins[m[2]] != origins[m[2]] || origins[m[2]]&^uint32((1<<maps)-1) != 0 {
			return fmt.Errorf("tensor-map operand lacks grid-constant parameter origin")
		}
		used |= origins[m[2]]
		seenIssue = true
		if issueIndex < 0 {
			issueIndex = index
		}
	}
	if !seenIssue || converted&uint32((1<<maps)-1) != uint32((1<<maps)-1) || used != uint32((1<<maps)-1) {
		return fmt.Errorf("required param-space tensor-map use absent")
	}
	if initIndex < 0 || fenceIndex <= initIndex || expectIndex <= fenceIndex || issueIndex <= expectIndex || waitIndex <= issueIndex || retireIndex <= waitIndex {
		return fmt.Errorf("prepared barrier/issue/wait/retirement instruction ordering absent")
	}
	return nil
}

func qualificationSyntheticPTX(c qualificationCase) string {
	// Deliberately synthetic parser/control input, never compiler or native
	// evidence. Ordinary and multicast kernels have separate instruction bodies.
	var b strings.Builder
	b.WriteString(".version 9.4\n.target sm_100a\n.address_size 64\n")
	for _, name := range []string{"tma_multicast", "tma_ordinary"} {
		fmt.Fprintf(&b, ".visible .entry %s(\n.param .align 64 .b8 %s_map[128],\n.param .u64 out,\n.param .u64 status,\n.param .u64 generation", name, name)
		for axis := 0; axis < c.Rank; axis++ {
			fmt.Fprintf(&b, ",\n.param .u32 c%d", axis)
		}
		fmt.Fprintf(&b, "\n) {\n.reg .b16 %%rs<1>;\n.reg .b32 %%r<16>;\n.reg .b64 %%rd<2>;\n.reg .pred p;\nmov.u64 %%rd0, %s_map;\ncvta.param.u64 %%rd1, %%rd0;\nmov.b16 %%rs0, %d;\n", name, c.Mask)
		tile := uint64(c.Width)
		for axis := 1; axis < c.Rank; axis++ {
			tile *= uint64(c.config().Descriptor.Box[axis])
		}
		fmt.Fprintf(&b, "mov.u32 %%r2, %d;\nmbarrier.init.shared::cta.b64 [%%r1], 1;\nfence.mbarrier_init.release.cluster;\nmbarrier.arrive.expect_tx.shared::cta.b64 _, [%%r1], %%r2;\nbarrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n", tile)
		space, qualifier := "cluster", ".multicast::cluster"
		if name == "tma_ordinary" {
			space, qualifier = "cta", ""
		}
		fmt.Fprintf(&b, "cp.async.bulk.tensor.%dd.shared::%s.global.tile.mbarrier::complete_tx::bytes%s [%%r0], [%%rd1, {", c.Rank, space, qualifier)
		for axis := 0; axis < c.Rank; axis++ {
			if axis > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%%r%d", axis+3)
		}
		b.WriteString("}], [%r1]")
		if qualifier != "" {
			b.WriteString(", %rs0")
		}
		b.WriteString(";\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%r1], 0;\nbarrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\nret;\n}\n")
	}
	return b.String()
}
func TestQualificationPTXABIControls(t *testing.T) {
	c := qualificationCase{Rank: 2, Type: ElementTypeFloat32, ElementBytes: 4, Width: 128, Swizzle: 128, Mask: 3}
	text := qualificationSyntheticPTX(c)
	if e := qualificationPTX(text, c); e != nil {
		t.Fatal(e)
	}
	rows := []struct{ name, old, new string }{
		{"mask width", ".reg .b16 %rs<1>", ".reg .b32 %rs<1>"},
		{"mask bit", "mov.b16 %rs0, 3", "mov.b16 %rs0, 1"},
		{"coordinate width", ".reg .b32 %r<16>", ".reg .b64 %r<16>"},
		{"map copied from local", "cvta.param.u64 %rd1", "cvta.local.u64 %rd1"},
		{"descriptor extent", "tma_multicast_map[128]", "tma_multicast_map[64]"},
		{"descriptor alignment", ".align 64 .b8 tma_multicast_map", ".b8 tma_multicast_map"},
		{"transaction count", "mov.u32 %r2, 384", "mov.u32 %r2, 1048576"},
		{"multicast qualifier", ".multicast::cluster", ""},
		{"init release fence", "fence.mbarrier_init.release.cluster;", ""},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			mutant := strings.Replace(text, row.old, row.new, 1)
			if mutant == text {
				t.Fatal("mutation absent")
			}
			if qualificationPTX(mutant, c) == nil {
				t.Fatal("ABI omission escaped independent parser control")
			}
		})
	}
}
