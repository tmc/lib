//go:build !windows

package moonep

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvlshm"
)

// Constants shared with the device kernels (moonep/constants.py).
const (
	// RankBits and KidxBits are the field widths of the packed int32
	// dedup encodings; they bound R <= 128 and K <= 32 (kmask is one b32).
	RankBits = 7
	KidxBits = 7

	// DedupBuilderWarps is the number of dedup-builder warps per dispatch CTA.
	DedupBuilderWarps = 4

	// blockSizeP2 is the planning phase-2 block size (BLOCK_SIZE_P2).
	blockSizeP2 = 2048

	// barrierSlots is the per-rank int32 footprint of the cross-rank
	// barrier: two phase signals plus one phase/sign counter.
	barrierSlots = 3

	bf16Bytes  = 2
	int32Bytes = 4

	int32Max = 1<<31 - 1
)

// Config are the MoonEP model/communication parameters.
type Config struct {
	S int // max tokens per rank
	H int // hidden size
	K int // top-k experts per token
	E int // total experts
	R int // expert-parallel ranks (world size)

	// B is the number of prefetch slots; 0 means E/R.
	B int

	// NumSMs is the cooperative grid size the kernels were built for.
	// It is required; there is no default, because no value this package
	// could invent would be right. MoonEP's kernels are JIT-specialized
	// on num_sms (see the export procedure in the README), so the correct
	// value is the one the loaded cubins were compiled with, and only the
	// caller knows that. Nor is it necessarily the device's SM count:
	// reading that attribute would produce a second invention, one that
	// can disagree with the cubin. The kernels launch cooperatively (see
	// [LaunchCooperative]), so a value above the resident capacity is
	// rejected by the driver at launch while one below it silently idles
	// the rest of the device; that quiet direction is why this field may
	// not be guessed.
	NumSMs int

	// TokenPadding pads each expert segment to a multiple of this many
	// token slots; 0 means 128.
	TokenPadding int
}

// A Layout holds every derived size and meta-buffer offset, mirroring
// MoonEP's _create_context (moonep/api.py). All meta offsets are int32
// element indices into a rank's meta chunk.
type Layout struct {
	Config          // with defaults applied
	N           int // S*K flat top-k entries per rank
	NvSCapacity int // S*K real slots
	NvS         int // logical slots incl. token-padding headroom
	NvSPadded   int // hidden_buf chunk dim0, VMM-granularity aligned
	NumVBlocks  int // ceil(N / 2048) planning vblocks
	VMMGran     uintptr
	McGran      uintptr

	// Meta chunk layout (int32 element offsets).
	WeightsOff       int
	TpeOff           int
	PlanOff          int
	PlanElems        int // planning output region size
	Topk0Off         int
	OrderOff         int
	Order0Off        int
	BarrierOff       int
	SrcInfoOff       int
	MetaChunkLogical int // used int32 elements per chunk
	MetaChunkPadded  int // allocated int32 elements per chunk
}

// HiddenChunkBytes returns one rank's hidden_buf chunk size:
// NvSPadded rows of H bf16 elements.
func (l *Layout) HiddenChunkBytes() uintptr {
	return uintptr(l.NvSPadded) * uintptr(l.H) * bf16Bytes
}

// MetaChunkBytes returns one rank's meta chunk size in bytes.
func (l *Layout) MetaChunkBytes() uintptr {
	return uintptr(l.MetaChunkPadded) * int32Bytes
}

// alignUp rounds n up to a multiple of a, mirroring MoonEP's _align_up
// (moonep/api.py). It is the package's one spelling of that rounding.
func alignUp(n, a int) int { return (n + a - 1) / a * a }

// NewLayout computes the buffer layout for cfg given the device's VMM
// allocation granularity and the multicast granularity for cfg.R devices
// (both in bytes; see nvlshm.Granularity and nvlshm.MulticastGranularity).
// The computation is pure so it can be checked without a GPU.
func NewLayout(cfg Config, vmmGran, mcGran uintptr) (*Layout, error) {
	if cfg.S <= 0 || cfg.H <= 0 || cfg.K <= 0 || cfg.E <= 0 || cfg.R <= 0 {
		return nil, fmt.Errorf("moonep: S, H, K, E, R must be positive: %+v", cfg)
	}
	if vmmGran == 0 || mcGran == 0 {
		return nil, fmt.Errorf("moonep: granularities must be positive")
	}
	if cfg.E%cfg.R != 0 {
		return nil, fmt.Errorf("moonep: E (%d) must be divisible by R (%d)", cfg.E, cfg.R)
	}
	epn := cfg.E / cfg.R
	if cfg.B == 0 {
		cfg.B = epn
	}
	if cfg.B < 0 {
		return nil, fmt.Errorf("moonep: B must be positive, got %d", cfg.B)
	}
	if cfg.NumSMs <= 0 {
		return nil, fmt.Errorf("moonep: NumSMs must be set to the cooperative grid size the kernels were built for, got %d", cfg.NumSMs)
	}
	if cfg.TokenPadding == 0 {
		cfg.TokenPadding = 128
	}
	if cfg.TokenPadding < 0 {
		return nil, fmt.Errorf("moonep: TokenPadding must be positive, got %d", cfg.TokenPadding)
	}
	if cfg.R > 1<<RankBits {
		return nil, fmt.Errorf("moonep: R (%d) exceeds RankBits capacity (%d)", cfg.R, 1<<RankBits)
	}
	if cfg.K > 32 {
		return nil, fmt.Errorf("moonep: K (%d) exceeds the 32-bit kmask", cfg.K)
	}

	l := &Layout{Config: cfg, VMMGran: vmmGran, McGran: mcGran}
	l.N = cfg.S * cfg.K
	if l.N <= 0 || l.N >= int32Max {
		return nil, fmt.Errorf("moonep: S*K (%d) out of int32 range", l.N)
	}
	l.NvSCapacity = l.N

	// Each destination rank receives at most epn remote and epn local
	// expert segments, each padded up to a multiple of TokenPadding, so
	// the logical slot headroom is (TokenPadding-1) * 2 * epn.
	l.NvS = l.NvSCapacity + (cfg.TokenPadding-1)*2*epn
	l.NumVBlocks = (l.N + blockSizeP2 - 1) / blockSizeP2

	// hidden_buf chunk dim0 padded so chunk bytes hit VMM granularity
	// exactly (port of pad_dim0_for_alignment).
	rowBytes := cfg.H * bf16Bytes
	nbytes := l.NvS * rowBytes
	padded, err := nvlshm.PadTo(uintptr(nbytes), vmmGran)
	if err != nil || padded > uintptr(int32Max) {
		return nil, fmt.Errorf("moonep: pad hidden chunk: %w", err)
	}
	paddedBytes := int(padded)
	dim0 := paddedBytes / rowBytes
	for dim0*rowBytes%int(vmmGran) != 0 {
		dim0++
	}
	l.NvSPadded = dim0

	// Meta chunk regions, each 4-element aligned for 16-byte vector ops.
	l.WeightsOff = 0
	l.TpeOff = alignUp(l.NvS, 4)
	l.PlanOff = alignUp(l.TpeOff+cfg.R*cfg.E, 4)
	broadcastElems := 3 * cfg.E * cfg.R
	if broadcastElems%4 != 0 {
		return nil, fmt.Errorf("moonep: broadcast elems (%d) must be divisible by 4", broadcastElems)
	}
	l.PlanElems = broadcastElems +
		cfg.R*(cfg.E+cfg.B) +
		2*cfg.R*(cfg.E+cfg.B) +
		cfg.B*cfg.R +
		2*cfg.R
	n4 := alignUp(l.N, 4)
	l.Topk0Off = alignUp(l.PlanOff+l.PlanElems, 4)
	l.OrderOff = l.Topk0Off + n4
	l.Order0Off = l.OrderOff + n4
	l.BarrierOff = l.Order0Off + n4
	l.SrcInfoOff = l.BarrierOff + barrierSlots
	l.MetaChunkLogical = l.SrcInfoOff + l.NvS

	// The meta chunk must satisfy both the VMM mapping and the multicast
	// binding granularity.
	chunkAlign := vmmGran
	if mcGran > chunkAlign {
		chunkAlign = mcGran
	}
	metaPadded, err := nvlshm.PadTo(uintptr(l.MetaChunkLogical)*int32Bytes, chunkAlign)
	if err != nil || metaPadded > uintptr(int32Max)*int32Bytes {
		return nil, fmt.Errorf("moonep: pad meta chunk: %w", err)
	}
	l.MetaChunkPadded = int(metaPadded) / int32Bytes

	// Device kernels multiply a runtime rank by these constexpr strides in
	// int32 arithmetic; guard the largest reachable non-negative index.
	if maxMeta := (cfg.R-1)*l.MetaChunkPadded + l.MetaChunkLogical - 1; maxMeta > int32Max {
		return nil, fmt.Errorf(
			"moonep: meta_buf rank-stride indexing would overflow int32: max index %d (R=%d NvS=%d chunk=%d)",
			maxMeta, cfg.R, l.NvS, l.MetaChunkPadded)
	}
	if maxHidden := (cfg.R-1)*l.NvSPadded + l.NvS - 1; maxHidden > int32Max {
		return nil, fmt.Errorf(
			"moonep: hidden_buf rank-stride indexing would overflow int32: max index %d (R=%d NvS=%d NvSPadded=%d)",
			maxHidden, cfg.R, l.NvS, l.NvSPadded)
	}
	// The dst encoding packs drank*NvS+loff into one int32; src_info mirrors it.
	if maxDst := cfg.R*l.NvS - 1; maxDst > int32Max {
		return nil, fmt.Errorf("moonep: R*NvS (%d) exceeds the int32 dst encoding", cfg.R*l.NvS)
	}
	return l, nil
}
