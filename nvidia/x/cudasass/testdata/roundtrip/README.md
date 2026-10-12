# Round-trip fixtures for lane 34 step 4

Provenance of every file here, and of the encoding facts asserted in
`../../dependency.go` and `../../roundtrip.go`. The GPU launch measurement
recorded in `roundtrip.go` is not among them; see the mirrored text section
below.

## Identity record (step 1)

| | |
|---|---|
| host | `gpu-host`, reached as `ssh gpu-host` (192.0.2.1) |
| device | NVIDIA GB10, compute capability 12.1 |
| driver | 580.173.02 |
| target | `sm_121a` -- the architecture-specific target, not the `sm_121` family |
| toolchain | CUDA 13.4, `nvcc`/`nvdisasm` V13.4.49, build `cuda_13.4.r13.4/compiler.38536908_0` |
| date | 2026-09-18 |

The `nvcc` first on the DGX's `PATH` reports 12.0 and rejects `sm_121a` with
"Unsupported .target 'sm_121a'". Everything here used
`/usr/local/cuda-13.4/bin` explicitly.

No GPU work was run: every command below is a compile or a disassembly, and
neither touches the device. Nothing in this directory required the GPU lock,
and nothing here qualifies any claim about GPU behaviour or performance.

## Files

- `kernel.cu` -- the fixture source. Written for this lane, owned here, so the
  built artifacts are redistributable. No vendor CUBIN is committed.

The other three files are toolchain output captured by the commands below and
not edited since; nothing in them is constructed.

- `sm121a-roundtrip.sass` -- disassembly with encodings. Printed with `-c`, it
  holds only the code sections `.text.intmix` and `.text.addk`; neither
  capmerc section below appears in it.
- `sm121a-roundtrip.bbcfg.dot` -- nvdisasm's own basic-block control-flow
  graph, used as the authority in `TestBasicBlockReconstructionAgainstNvdisasm`.
- `sm121a-roundtrip.cubin.hex` -- the CUBIN the two text artifacts were
  disassembled from, kept for the step 5 patch path, which needs the container
  and not only the text. Built here from `kernel.cu`; md5
  `1ee1a13df438e27deea643c990d938e3`, 8440 bytes. No vendor CUBIN is
  committed. It is stored hex-encoded, as the other CUBIN fixtures in this
  package are, so that the repository holds no binary files; the tests decode
  it on read.

## Commands

    export PATH=/usr/local/cuda-13.4/bin:$PATH
    nvcc -arch=sm_121a -cubin -o k.cubin kernel.cu
    nvdisasm -c -hex   k.cubin > sm121a-roundtrip.sass
    nvdisasm -bbcfg -poff k.cubin > sm121a-roundtrip.bbcfg.dot
    xxd -p -c 1000000 k.cubin        > sm121a-roundtrip.cubin.hex

Step 5 keeps the CUBIN as well, because the patch path operates on the ELF
container: section lookup, alignment, relocations and the mirrored text
section below are all properties of the file, not of the disassembly.

## The mirrored text section (step 5)

CUDA 13.4's `ptxas` emits, beside every `.text.<kernel>`, a second section
`.nv.capmerc.text.<kernel>` in an undocumented form. What it encodes has not
been measured. In the committed CUBIN:

| kernel | `.text.<kernel>` | `.nv.capmerc.text.<kernel>` | distinct byte values in capmerc |
|---|---|---|---|
| `intmix` | 0x180 bytes, 24 instructions | 0xaa bytes | 32 |
| `addk` | 0x200 bytes, 32 instructions | 0x102 bytes | 31 |

Neither capmerc section contains any 16-byte instruction of its kernel's text,
nor either 8-byte half of one. When the fixture was built, the pair was
observed to come from `ptxas` itself, not from `nvcc`: building through
`nvcc -cubin` and building `ptxas -arch=sm_121a` straight from PTX produced the
same pair, and `--compress-mode`/`-no-compress` govern fatbinaries rather than
this. That observation has not been repeated.

The patch path cannot update that copy, so it reports it and the load gate
refuses. Nothing in this directory decides which copy a driver executes, and a
load that succeeds is consistent with either. The comment on
`CubinImage.Mirrors` in `../../roundtrip.go` records a separate measurement on
the GB10 host: one byte flipped at stride 16, the image reloaded and launched,
40 of 53 flips in `.text.<kernel>` failed the launch and 0 of 25 in the capmerc
section did. Under that driver and launch configuration the executed bytes
come from `.text`. No command or output of that measurement is kept here, and
nothing here repeats it.

What this directory establishes is narrower: both `nvdisasm -c -hex` and
`cuobjdump -sass` read the patched `.text.<kernel>` and print the mutated
instruction, so the patch is coherent to the toolchain that produced the file.

    # on the DGX, after writing the patched image
    nvdisasm -c -hex patched.cubin | sed -n '/text.intmix/,/0x0030/p'
    cuobjdump -sass patched.cubin | grep -A3 S2R
    # /*0010*/  S2R R7, SR_TID.Y ;  /* 0x0000000000077919 */
    #                               /* 0x000e220000002200 */

## How the encoding facts were obtained

A probe took a known-good encoding, flipped each of the 128 bits in turn, and
disassembled each variant with `nvdisasm --binary SM121a`. A bit whose flip
changes the printed text names a field; a bit whose flip is rejected is a field
whose value became illegal; a bit whose flip changes nothing is invisible in the
text.

Each variant is disassembled on its own. Batching them into one file does not
work: `nvdisasm` aborts the whole file on the first illegal encoding and emits
nothing, so a batched probe silently loses every legal variant that follows one
illegal one. The first version of this probe did exactly that and reported 94 of
128 bits as illegal.

### Fields observed on sm_121a

Word bits 0-63 are `Lo`, 64-127 are `Hi`.

| bits | field | evidence |
|---|---|---|
| Lo 0-11 | opcode | flipping any of them yields a different mnemonic or an illegal encoding |
| Lo 12-14 | guarding predicate register, 7 meaning unguarded | sweeping 0-7 prints `@P0`..`@P6` then nothing |
| Lo 15 | predicate negated | prints `@!PT` |
| Lo 16-23 | destination register | sweeping prints `R0`..`R255` |
| Lo 24-31 | first source register | confirmed on `IADD Rd, Ra, URb` |
| Lo 32-39 | second source register | uniform numbering in the uniform forms |
| Hi 8-15 | special-register selector (`S2R`) | sweeping 0x00-0x4f prints `SR_LANEID`, `SR_CLOCK`, `SR_TID.X` and the rest |
| 105-108 | stall count | see below |
| 109 | yield | clearing it suppresses the `.reuse` suffix, so reuse is only meaningful while it is set |
| 110-112 | write barrier index, 7 meaning none | |
| 113-115 | read barrier index, 7 meaning none | |
| 116-121 | wait mask, one bit per barrier | |
| 122-125 | reuse flags, one per source slot | flipping 122 removes `.reuse`; 123 and 124 are rejected on a form with fewer reusable slots |
| 126-127 | no observed effect on any instruction probed | |

### The stall field is not a four-bit range, and the range is per-instruction

Sweeping all sixteen values of the stall field:

- on `IADD R0, R7.reuse, UR6`, nvdisasm accepts 1-11 and rejects 0 and 12-15;
- on `NOP`, it accepts all sixteen -- and `ptxas` itself emits `NOP` with
  stall 0, in all 21 occurrences in this fixture.

The first version of this model took the `IADD` sweep for an architecture fact
and rejected stall 0 everywhere. The round-trip gate then failed on every real
`NOP`, which is how the error was found. A range measured on one instruction is
a fact about that instruction. `ObservedStallRange` records the two swept forms
and reports absence for the others rather than substituting either.

## Mutation receipts

Encodings produced by `ReEncode` from mutated operand text, then disassembled
on the target:

    0x00000000001f7919 0x000e220000002100  ->  S2R R31, SR_TID.X
    0x0000000000c87919 0x000e220000002100  ->  S2R R200, SR_TID.X
    0x00000006071f7c35 0x041fe2000f8e0000  ->  IADD R31, R7.reuse, UR6
    0x0000000607c87c35 0x041fe2000f8e0000  ->  IADD R200, R7.reuse, UR6

Each is a register the compiler never emitted at that site.
