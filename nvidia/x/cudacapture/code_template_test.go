package cudacapture_test

import (
	"encoding/binary"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// templateSymbol is one instantiation of a template kernel as nvcc writes it
// into the CUBIN: down_q2k_d2r_kernel<64, 64>, Itanium-mangled. The source
// spells only "down_q2k_d2r_kernel", so a literal comparison against the
// symbol finds no definition.
const templateSymbol = "_Z19down_q2k_d2r_kernelILi64ELi64EEvPi"

// cubinWithTextSection builds the smallest CUBIN LoadCode accepts: an ELF64
// header, a section-name table, and one .text section holding two 16-byte
// instructions.
//
// The artifacts in this test are written rather than compiled, which the
// fixtures in testdata are not: a template instantiation's symbol is chosen by
// the instantiation, so pinning one here states which symbol the join is being
// asked about. Every other property of the join is covered by the compiled
// fixtures.
func cubinWithTextSection(symbol string, text []byte) []byte {
	const (
		ehsize    = 64
		shentsize = 64
		shnum     = 3
	)
	names := []string{"", ".shstrtab", ".text." + symbol}
	var shstrtab []byte
	offsets := make([]uint32, len(names))
	for i, n := range names {
		offsets[i] = uint32(len(shstrtab))
		shstrtab = append(shstrtab, n...)
		shstrtab = append(shstrtab, 0)
	}

	strOff := uint64(ehsize)
	textOff := strOff + uint64(len(shstrtab))
	if pad := textOff % 32; pad != 0 { // keep instruction offsets 16-byte aligned
		textOff += 32 - pad
	}
	shoff := textOff + uint64(len(text))

	buf := make([]byte, shoff+shentsize*shnum)
	copy(buf, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	binary.LittleEndian.PutUint16(buf[16:], 2)   // ET_EXEC
	binary.LittleEndian.PutUint16(buf[18:], 190) // EM_CUDA
	binary.LittleEndian.PutUint32(buf[20:], 1)
	binary.LittleEndian.PutUint64(buf[0x28:], shoff)
	binary.LittleEndian.PutUint16(buf[0x34:], ehsize)
	binary.LittleEndian.PutUint16(buf[0x3a:], shentsize)
	binary.LittleEndian.PutUint16(buf[0x3c:], shnum)
	binary.LittleEndian.PutUint16(buf[0x3e:], 1) // shstrndx
	copy(buf[strOff:], shstrtab)
	copy(buf[textOff:], text)

	put := func(i int, name uint32, typ uint32, off, size uint64) {
		o := shoff + uint64(i*shentsize)
		binary.LittleEndian.PutUint32(buf[o:], name)
		binary.LittleEndian.PutUint32(buf[o+4:], typ)
		binary.LittleEndian.PutUint64(buf[o+24:], off)
		binary.LittleEndian.PutUint64(buf[o+32:], size)
	}
	put(1, offsets[1], 3 /*SHT_STRTAB*/, strOff, uint64(len(shstrtab)))
	put(2, offsets[2], 1 /*SHT_PROGBITS*/, textOff, uint64(len(text)))
	return buf
}

// TestLoadCodeTemplateKernel checks that a template kernel resolves to its
// single source definition. Every instantiation carries a distinct mangled
// symbol and they all share the one "void down_q2k_d2r_kernel(" line.
func TestLoadCodeTemplateKernel(t *testing.T) {
	const source = `// Sample CUDA kernel
template <int TileN, int BaseTileN>
__global__ void down_q2k_d2r_kernel(int *a) {
    *a = 1;
}
`
	const ptx = `.version 8.0
.target sm_90
.address_size 64

.file 1 "kernels.cu"

.visible .entry ` + templateSymbol + `(
	.param .u64 a
)
{
	.loc 1 3 0;
	mov.u64 %rd1, %rd2;
	.loc 1 4 0;
	ret;
}
`
	const sass = `.section .text.` + templateSymbol + `
//## File "kernels.cu", line 3
/* 0x0000 */   MOV R1, c[0x0][0x28] ;
//## File "kernels.cu", line 4
/* 0x0010 */   RET ;
`
	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte(source),
			"kernels.ptx":   []byte(ptx),
			"kernels.cubin": cubinWithTextSection(templateSymbol, make([]byte, 32)),
			"kernels.sass":  []byte(sass),
		},
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	if len(code.Kernels) != 1 {
		t.Fatalf("got %d kernels, want 1", len(code.Kernels))
	}
	k := code.Kernels[0]
	if k.Name != templateSymbol {
		t.Errorf("Kernel.Name = %q, want %q", k.Name, templateSymbol)
	}
	if k.SourceLine != 3 {
		t.Errorf("Kernel.SourceLine = %d, want 3", k.SourceLine)
	}
}
