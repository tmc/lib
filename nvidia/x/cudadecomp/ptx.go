package cudadecomp

import (
	"fmt"
	"regexp"
	"strings"
)

// ptxVersions maps the targets [EmitBytePlaneMergePTX] accepts to the
// first PTX ISA version that supports them.
var ptxVersions = map[string]string{
	"sm_80":   "7.0",
	"sm_90":   "7.8",
	"sm_90a":  "8.0",
	"sm_100a": "8.6",
	"sm_120a": "8.7",
	"sm_121a": "8.8",
}

// Merge selectors for prmt.b32 with the low-byte word as its first operand
// and the high-byte word as its second: byte k of the result is byte s_k of
// the eight-byte pair {hi, lo}, s_k being nibble k of the selector.
const (
	mergeSelectLow  = 0x5140 // elements 0 and 1: lo0 hi0 lo1 hi1
	mergeSelectHigh = 0x7362 // elements 2 and 3: lo2 hi2 lo3 hi3
)

var ptxIdent = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// EmitBytePlaneMergePTX returns a PTX module for target holding the kernel
// name(hi, lo, out, n). Thread i, for i < n, loads the 32-bit words hi[i]
// and lo[i] (the high and low bytes of BF16 elements 4i to 4i+3, as
// [SplitBytePlanes] lays them out) and stores the two bf16x2 words of those
// elements at out[2i] and out[2i+1], merging the bytes with two prmt.b32.
func EmitBytePlaneMergePTX(name, target string) (string, error) {
	if !ptxIdent.MatchString(name) {
		return "", fmt.Errorf("cudadecomp: bad kernel name %q", name)
	}
	version, ok := ptxVersions[target]
	if !ok {
		return "", fmt.Errorf("cudadecomp: unsupported target %q", target)
	}
	var b strings.Builder
	fmt.Fprintf(&b, ".version %s\n.target %s\n.address_size 64\n\n", version, target)
	fmt.Fprintf(&b, ".visible .entry %[1]s(\n\t.param .u64 %[1]s_hi,\n\t.param .u64 %[1]s_lo,\n\t.param .u64 %[1]s_out,\n\t.param .u32 %[1]s_n\n)\n{\n", name)
	fmt.Fprintf(&b, `	.reg .pred %%p<2>;
	.reg .b32 %%r<10>;
	.reg .b64 %%rd<10>;

	ld.param.u64 %%rd1, [%[1]s_hi];
	ld.param.u64 %%rd2, [%[1]s_lo];
	ld.param.u64 %%rd3, [%[1]s_out];
	ld.param.u32 %%r1, [%[1]s_n];
	mov.u32 %%r2, %%ctaid.x;
	mov.u32 %%r3, %%ntid.x;
	mov.u32 %%r4, %%tid.x;
	mad.lo.s32 %%r5, %%r2, %%r3, %%r4;
	setp.ge.u32 %%p1, %%r5, %%r1;
	@%%p1 bra DONE;
	cvta.to.global.u64 %%rd4, %%rd1;
	cvta.to.global.u64 %%rd5, %%rd2;
	cvta.to.global.u64 %%rd6, %%rd3;
	mul.wide.u32 %%rd7, %%r5, 4;
	add.s64 %%rd8, %%rd4, %%rd7;
	ld.global.u32 %%r6, [%%rd8];
	add.s64 %%rd8, %%rd5, %%rd7;
	ld.global.u32 %%r7, [%%rd8];
	prmt.b32 %%r8, %%r7, %%r6, %#04x;
	prmt.b32 %%r9, %%r7, %%r6, %#04x;
	mul.wide.u32 %%rd9, %%r5, 8;
	add.s64 %%rd9, %%rd6, %%rd9;
	st.global.v2.u32 [%%rd9], {%%r8, %%r9};
DONE:
	ret;
}
`, name, mergeSelectLow, mergeSelectHigh)
	return b.String(), nil
}
