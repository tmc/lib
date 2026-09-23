package cudabpfdev

import (
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
)

// Translation contains verified PTX and its kernel ABI. DeclaredRegisters is
// the conservative 32-bit register footprint of the emitted declarations,
// including scratch registers and predicates. It is not the compiler's final
// register allocation. Injection still requires the compiled resource report.
type Translation struct {
	Artifact          *cudakernel.Artifact
	Analysis          Analysis
	DeclaredRegisters uint32
	digest, target    string
	schema            cudakernel.Schema
	slots             uint32
	limits            resourceLimits
}

// Translate emits a standalone one-warp oracle kernel for a verified program.
// It is the device bytecode execution path, not a trampoline injector. Launch
// exactly one block of 32 threads. Arguments are a zeroed Slots*4-byte map,
// a 128-byte per-lane return array, a zeroed 8-byte miss counter and an active
// lane mask. All storage must be disjoint and private to the invocation, and
// remain live until kernel completion. Inactive result slots are not written.
//
// PTX ISA 9.0 is required. Shared staging is not emitted yet; a nonzero request
// is rejected. Unknown target limits are rejected, never filled with defaults.
func Translate(p Program, b Budget, target string) (Translation, error) {
	p.Code = slices.Clone(p.Code)
	a, err := Verify(p, b)
	if err != nil {
		return Translation{}, err
	}
	info, err := cudakernel.ProbedTargetSM(target)
	if err != nil {
		return Translation{}, fmt.Errorf("cudabpfdev: target: %w", err)
	}
	if info.Major < 9 {
		return Translation{}, fmt.Errorf("cudabpfdev: target %s is outside the emitter contract", target)
	}
	if p.StagingBytes > info.MaxOptInSharedBytes {
		return Translation{}, fmt.Errorf("cudabpfdev: shared staging exceeds target limit")
	}
	if p.StagingBytes != 0 {
		return Translation{}, fmt.Errorf("cudabpfdev: shared staging is not emitted")
	}
	// Seven b32, six b64 and four predicate scratch declarations.
	declared := a.Registers + 23
	if declared > b.MaxRegisters || declared > 255 {
		return Translation{}, fmt.Errorf("cudabpfdev: emitted register budget exceeded")
	}
	var out strings.Builder
	fmt.Fprintf(&out, ".version 9.0\n.target %s\n.address_size 64\n\n", info.Arch)
	out.WriteString(`.visible .entry cudabpfdev_main(
    .param .u64 map_ptr,
    .param .u64 result_ptr,
    .param .u64 miss_ptr,
    .param .u32 active_mask
)
.reqntid 32, 1, 1
{
    .reg .b32 lane, leader, reverse_mask, mask, bit, discarded, block_id;
    .reg .b64 map_addr, result_addr, miss_addr, offset, addr, ignored64;
    .reg .pred pred, is_leader, in_bounds, write_map;
`)
	fmt.Fprintf(&out, "    .reg .b32 r<%d>;\n", p.Registers)
	for pc, in := range p.Code {
		if in.Op == Repeat {
			fmt.Fprintf(&out, "    .reg .b32 loop%d;\n", pc)
		}
	}
	out.WriteString(`    mov.u32 block_id, %ctaid.x;
    setp.ne.u32 pred, block_id, 0;
    @pred bra done;
    mov.u32 block_id, %ctaid.y;
    setp.ne.u32 pred, block_id, 0;
    @pred bra done;
    mov.u32 block_id, %ctaid.z;
    setp.ne.u32 pred, block_id, 0;
    @pred bra done;
    ld.param.u32 mask, [active_mask];
    mov.u32 lane, %laneid;
    shl.b32 bit, 1, lane;
    and.b32 bit, bit, mask;
    setp.eq.u32 pred, bit, 0;
    @pred bra done;
    brev.b32 reverse_mask, mask;
    bfind.u32 leader, reverse_mask;
    sub.u32 leader, 31, leader;
    setp.eq.u32 is_leader, lane, leader;
    ld.param.u64 map_addr, [map_ptr];
    ld.param.u64 result_addr, [result_ptr];
    ld.param.u64 miss_addr, [miss_ptr];
    cvta.to.global.u64 map_addr, map_addr;
    cvta.to.global.u64 result_addr, result_addr;
    cvta.to.global.u64 miss_addr, miss_addr;
`)
	for r := uint8(0); r < p.Registers; r++ {
		fmt.Fprintf(&out, "    mov.u32 r%d, 0;\n", r)
	}
	for pc, in := range p.Code {
		if in.Op == Repeat {
			fmt.Fprintf(&out, "    mov.u32 loop%d, 0;\n", pc)
		}
	}
	for pc, in := range p.Code {
		fmt.Fprintf(&out, "pc%d:\n", pc)
		emit := emitters[in.Op]
		if emit == nil {
			return Translation{}, &VerifyError{PC: pc, Reason: "opcode has no PTX emitter"}
		}
		emit(&out, pc, in, p.Slots)
	}
	out.WriteString("done:\n    ret;\n}\n")
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "map_ptr", Type: "u64", IsPointer: true},
		cudakernel.Param{Name: "result_ptr", Type: "u64", IsPointer: true},
		cudakernel.Param{Name: "miss_ptr", Type: "u64", IsPointer: true},
		cudakernel.Param{Name: "active_mask", Type: "u32"},
	)
	if err != nil {
		return Translation{}, err
	}
	artifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(out.String()), "cudabpfdev_main", schema, info.Arch)
	if err != nil {
		return Translation{}, err
	}
	sealedSchema := schema
	sealedSchema.Params = slices.Clone(schema.Params)
	return Translation{Artifact: artifact, Analysis: a, DeclaredRegisters: declared, digest: artifact.Digest, target: artifact.Target, schema: sealedSchema, slots: p.Slots,
		limits: resourceLimits{registers: min(b.MaxRegisters, 255), sharedBytes: min(b.MaxSharedBytes, info.MaxSharedMemoryBytes)}}, nil
}

type emitter func(*strings.Builder, int, Instruction, uint32)

var emitters = map[Op]emitter{
	Const: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    mov.u32 r%d, %d;\n", in.Dst, in.Imm)
	},
	Move: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    mov.u32 r%d, r%d;\n", in.Dst, in.A)
	},
	Add: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    add.u32 r%d, r%d, r%d;\n", in.Dst, in.A, in.B)
	},
	And: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    and.b32 r%d, r%d, r%d;\n", in.Dst, in.A, in.B)
	},
	Lane: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    mov.u32 r%d, lane;\n", in.Dst)
	},
	Ballot: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    setp.ne.u32 pred, r%d, 0;\n    vote.sync.ballot.b32 r%d, pred, mask;\n", in.A, in.Dst)
	},
	Leader: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    shfl.sync.idx.b32 r%d, r%d, leader, 31, mask;\n", in.Dst, in.A)
	},
	JumpNonzero: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    setp.ne.u32 pred, r%d, 0;\n    @pred bra pc%d;\n", in.A, in.Target)
	},
	Repeat: func(w *strings.Builder, pc int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    add.u32 loop%d, loop%d, 1;\n    setp.lt.u32 pred, loop%d, %d;\n    @pred bra pc%d;\n    mov.u32 loop%d, 0;\n", pc, pc, pc, in.Imm, in.Target, pc)
	},
	MapAdd: func(w *strings.Builder, _ int, in Instruction, slots uint32) {
		fmt.Fprintf(w, "    setp.lt.u32 in_bounds, r%d, %d;\n    and.pred write_map, is_leader, in_bounds;\n    mul.wide.u32 offset, r%d, 4;\n    add.u64 addr, map_addr, offset;\n    @write_map atom.global.add.u32 discarded, [addr], r%d;\n    not.pred in_bounds, in_bounds;\n    and.pred write_map, is_leader, in_bounds;\n    @write_map atom.global.add.u64 ignored64, [miss_addr], 1;\n", in.A, slots, in.A, in.B)
	},
	Exit: func(w *strings.Builder, _ int, in Instruction, _ uint32) {
		fmt.Fprintf(w, "    mul.wide.u32 offset, lane, 4;\n    add.u64 addr, result_addr, offset;\n    st.global.u32 [addr], r%d;\n    bra done;\n", in.A)
	},
}

// Assemble links translated PTX to CUBIN using nvJitLink, never NVRTC. It does
// not load or launch the output. A result accompanying an error may own native
// cleanup and must be closed according to nvjitlinkx.Result's contract.
func Assemble(t Translation) (*nvjitlinkx.Result, error) { return assemble(t, nvjitlinkx.Link) }

func assemble(t Translation, link func([]nvjitlinkx.Input, ...string) (*nvjitlinkx.Result, error)) (*nvjitlinkx.Result, error) {
	if err := t.Artifact.Validate(); err != nil {
		return nil, fmt.Errorf("cudabpfdev: PTX artifact: %w", err)
	}
	if t.Artifact.Kind != cudakernel.KindPTX {
		return nil, fmt.Errorf("cudabpfdev: expected PTX artifact")
	}
	if t.digest == "" || t.Artifact.Digest != t.digest || t.Artifact.Target != t.target || t.Artifact.EntryPoint != "cudabpfdev_main" || !reflect.DeepEqual(t.Artifact.Schema, t.schema) {
		return nil, fmt.Errorf("cudabpfdev: translation identity changed or unavailable")
	}
	in, err := nvjitlinkx.InputFromKernelArtifact("cudabpfdev.ptx", t.Artifact)
	if err != nil {
		return nil, err
	}
	return link([]nvjitlinkx.Input{in}, "-arch="+t.Artifact.Target)
}
