package cudatma

import (
	"fmt"
	"math"
	"math/bits"
)

const (
	TargetSM90a  = "sm_90a"
	TargetSM120a = "sm_120a"
)

// L2EvictionPolicy names a PTX L2 cache eviction priority hint used with
// createpolicy.fractional.L2.
type L2EvictionPolicy string

const (
	L2EvictNormal    L2EvictionPolicy = "evict_normal"
	L2EvictFirst     L2EvictionPolicy = "evict_first"
	L2EvictLast      L2EvictionPolicy = "evict_last"
	L2EvictUnchanged L2EvictionPolicy = "evict_unchanged"
)

func (p L2EvictionPolicy) valid() bool {
	switch p {
	case L2EvictNormal, L2EvictFirst, L2EvictLast, L2EvictUnchanged:
		return true
	}
	return false
}

// L2CachePolicySpec describes a 64-bit L2 cache eviction policy descriptor
// constructed by PTX createpolicy.fractional.L2 and passed to TMA transfers via
// .L2::cache_hint.
//
// For persistent expert weights or KV cache blocks, set PrimaryPolicy to
// L2EvictLast with Fraction = 1.0. For streaming activations that should not
// displace resident weights, set PrimaryPolicy to L2EvictFirst (and optionally
// SecondaryPolicy to L2EvictUnchanged or L2EvictNormal) with a fractional
// quota such as 0.25.
type L2CachePolicySpec struct {
	PrimaryPolicy   L2EvictionPolicy
	SecondaryPolicy L2EvictionPolicy
	Fraction        float32
}

// Validate checks that the policy names and fraction are admitted by PTX
// createpolicy.fractional.L2.
func (s L2CachePolicySpec) Validate() error {
	if !s.PrimaryPolicy.valid() {
		return fmt.Errorf("unsupported primary L2 eviction policy %q", s.PrimaryPolicy)
	}
	if s.PrimaryPolicy == L2EvictUnchanged {
		return fmt.Errorf("primary L2 eviction policy cannot be %q", s.PrimaryPolicy)
	}
	if s.SecondaryPolicy != "" && !s.SecondaryPolicy.valid() {
		return fmt.Errorf("unsupported secondary L2 eviction policy %q", s.SecondaryPolicy)
	}
	if math.IsNaN(float64(s.Fraction)) || math.IsInf(float64(s.Fraction), 0) || s.Fraction <= 0 || s.Fraction > 1.0 {
		return fmt.Errorf("L2 policy fraction %v must be in (0.0, 1.0]", s.Fraction)
	}
	return nil
}

// FormatPTX emits the PTX createpolicy.fractional instruction writing a 64-bit
// L2 cache policy descriptor into dstReg from srcFractionReg.
func (s L2CachePolicySpec) FormatPTX(dstReg, srcFractionReg string) (string, error) {
	if err := s.Validate(); err != nil {
		return "", err
	}
	if dstReg == "" || srcFractionReg == "" {
		return "", fmt.Errorf("destination and source registers must be non-empty")
	}
	if s.SecondaryPolicy != "" {
		return fmt.Sprintf("createpolicy.fractional.L2::%s.L2::%s.b64 %s, %s;",
			s.PrimaryPolicy, s.SecondaryPolicy, dstReg, srcFractionReg), nil
	}
	return fmt.Sprintf("createpolicy.fractional.L2::%s.b64 %s, %s;",
		s.PrimaryPolicy, dstReg, srcFractionReg), nil
}

// TMAGather4Mode selects between a 4-row non-contiguous TMA 2D gather load
// (global to shared::cta) and a 4-row non-contiguous scatter store
// (shared::cta to global).
type TMAGather4Mode string

const (
	TMAGather4Load   TMAGather4Mode = "gather4"
	TMAScatter4Store TMAGather4Mode = "scatter4"
)

// TMAGather4Spec describes a Blackwell (sm_100a, sm_103a, sm_100f, sm_120a,
// sm_121a) 2D TMA .gather4 or .scatter4 transfer that gathers or scatters four
// non-contiguous global rows of width ColDim elements in a single instruction.
type TMAGather4Spec struct {
	Target     string
	Mode       TMAGather4Mode
	ColDim     int
	ElemBytes  int
	RowIndices [4]int32
	L2Policy   *L2CachePolicySpec
}

func validGather4Target(target string) bool {
	switch target {
	case TargetSM100a, TargetSM103a, TargetSM100f, TargetSM120a, TargetSM121a:
		return true
	}
	return false
}

// Validate checks target support, mode, element width, 16-byte row alignment,
// non-negative row coordinates, and optional L2 cache policy.
func (s TMAGather4Spec) Validate() error {
	if !validGather4Target(s.Target) {
		return fmt.Errorf("unsupported gather4/scatter4 target %q", s.Target)
	}
	if s.Mode != TMAGather4Load && s.Mode != TMAScatter4Store {
		return fmt.Errorf("unsupported gather4 mode %q", s.Mode)
	}
	if s.ElemBytes != 1 && s.ElemBytes != 2 && s.ElemBytes != 4 {
		return fmt.Errorf("element bytes %d must be 1, 2, or 4", s.ElemBytes)
	}
	if s.ColDim <= 0 {
		return fmt.Errorf("column dimension %d must be positive", s.ColDim)
	}
	rowBytes := s.ColDim * s.ElemBytes
	if rowBytes%16 != 0 {
		return fmt.Errorf("contiguous row width %d bytes (%d * %d) must be a multiple of 16", rowBytes, s.ColDim, s.ElemBytes)
	}
	for i, idx := range s.RowIndices {
		if idx < 0 {
			return fmt.Errorf("row index %d (%d) must be non-negative", i, idx)
		}
	}
	if s.L2Policy != nil {
		if err := s.L2Policy.Validate(); err != nil {
			return fmt.Errorf("L2 policy: %w", err)
		}
	}
	return nil
}

// TransferBytes returns the total payload size in bytes across all 4 gathered
// or scattered rows.
func (s TMAGather4Spec) TransferBytes() int {
	return 4 * s.ColDim * s.ElemBytes
}

// FormatPTX emits the PTX cp.async.bulk.tensor.2d instruction for a 4-row
// .gather4 load or .scatter4 store, appending .L2::cache_hint when L2Policy is
// non-nil.
func (s TMAGather4Spec) FormatPTX() (string, error) {
	if err := s.Validate(); err != nil {
		return "", err
	}
	switch s.Mode {
	case TMAGather4Load:
		if s.L2Policy != nil {
			return "cp.async.bulk.tensor.2d.shared::cta.global.tile.gather4.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;", nil
		}
		return "cp.async.bulk.tensor.2d.shared::cta.global.tile.gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];", nil
	case TMAScatter4Store:
		if s.L2Policy != nil {
			return "cp.async.bulk.tensor.2d.global.shared::cta.tile.scatter4.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], [%6], %7;", nil
		}
		return "cp.async.bulk.tensor.2d.global.shared::cta.tile.scatter4 [%0, {%1, %2, %3, %4, %5}], [%6];", nil
	default:
		return "", fmt.Errorf("unsupported gather4 mode %q", s.Mode)
	}
}

// TMAMulticastSpec describes a 2D TMA cluster multicast transfer
// (cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster)
// that broadcasts a global tile into the shared memory of every CTA selected by
// CTAMask within a thread block cluster.
type TMAMulticastSpec struct {
	Target      string
	ClusterSize int
	CTAMask     uint16
	TileBytes   int
	L2Policy    *L2CachePolicySpec
}

// Validate checks target support, cluster size, non-zero recipient bitmask,
// 16-byte tile alignment, and optional L2 cache policy.
func (s TMAMulticastSpec) Validate() error {
	switch s.ClusterSize {
	case 2, 4, 8, 16:
	default:
		return fmt.Errorf("cluster size %d must be 2, 4, 8, or 16", s.ClusterSize)
	}
	switch s.Target {
	case TargetSM90a, TargetSM100a, TargetSM103a, TargetSM100f:
	case TargetSM120a, TargetSM121a:
		if s.ClusterSize > 2 {
			return fmt.Errorf("target %q does not support cluster size %d (max 2)", s.Target, s.ClusterSize)
		}
	default:
		return fmt.Errorf("unsupported multicast target %q", s.Target)
	}
	maxMask := uint32(1)<<s.ClusterSize - 1
	if s.CTAMask == 0 || uint32(s.CTAMask) > maxMask {
		return fmt.Errorf("CTA mask 0x%x must be in [0x1..0x%x] for cluster size %d", s.CTAMask, maxMask, s.ClusterSize)
	}
	if s.TileBytes <= 0 || s.TileBytes%16 != 0 {
		return fmt.Errorf("tile bytes %d must be a positive multiple of 16", s.TileBytes)
	}
	if s.TileBytes > maxMulticastTransactionBytes {
		return fmt.Errorf("tile bytes %d exceed mbarrier tx-count limit %d", s.TileBytes, maxMulticastTransactionBytes)
	}
	if s.L2Policy != nil {
		if err := s.L2Policy.Validate(); err != nil {
			return fmt.Errorf("L2 policy: %w", err)
		}
	}
	return nil
}

// RecipientCount returns the number of destination CTAs selected in CTAMask.
func (s TMAMulticastSpec) RecipientCount() int {
	return bits.OnesCount16(s.CTAMask)
}

// FormatPTX emits the PTX cp.async.bulk.tensor.2d.shared::cluster.global.tile
// instruction with .multicast::cluster and optional .L2::cache_hint.
func (s TMAMulticastSpec) FormatPTX() (string, error) {
	if err := s.Validate(); err != nil {
		return "", err
	}
	if s.L2Policy != nil {
		return "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5, %6;", nil
	}
	return "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;", nil
}
