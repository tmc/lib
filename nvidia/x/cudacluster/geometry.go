package cudacluster

import "fmt"

// Cluster geometry is read from PTX special registers rather than computed.
//
// Only %cluster_ctarank is verified in this tree: it is asserted by
// cudadevicetable's cluster.ptx fixture, which ptxas accepted for every target
// at or above sm_90 and refused at sm_80 with a named diagnostic. The
// remaining registers below are spelled per the PTX ISA and are NOT covered by
// any fixture here, so an assembler failure on them would be a new fact rather
// than a regression. They are marked individually.

// EmitBlockRank returns PTX reading the calling CTA's rank within its cluster
// into dst, a .u32 register.
//
// Verified: %cluster_ctarank is asserted by cudadevicetable/ptxprobe/cluster.ptx.
func EmitBlockRank(dst string) string {
	return "mov.u32 " + dst + ", %cluster_ctarank;"
}

// EmitClusterSize returns PTX reading the number of CTAs in the cluster into
// dst, a .u32 register.
//
// Not covered by any fixture in this tree. %cluster_nctarank is the ISA
// spelling for the cluster's CTA count.
func EmitClusterSize(dst string) string {
	return "mov.u32 " + dst + ", %cluster_nctarank;"
}

// EmitClusterCTAID returns PTX reading the calling CTA's three-dimensional id
// within its cluster into dst, a .u32 register, for the named dimension.
//
// Not covered by any fixture in this tree.
func EmitClusterCTAID(dst string, dim Dim) (string, error) {
	axis, err := dim.axis()
	if err != nil {
		return "", err
	}
	return "mov.u32 " + dst + ", %cluster_ctaid." + axis + ";", nil
}

// EmitClusterNCTAID returns PTX reading the cluster's three-dimensional
// extent in CTAs into dst, a .u32 register, for the named dimension.
//
// Not covered by any fixture in this tree.
func EmitClusterNCTAID(dst string, dim Dim) (string, error) {
	axis, err := dim.axis()
	if err != nil {
		return "", err
	}
	return "mov.u32 " + dst + ", %cluster_nctaid." + axis + ";", nil
}

// Dim names one axis of a cluster's CTA grid.
type Dim int

// The three cluster axes.
const (
	DimX Dim = iota
	DimY
	DimZ
)

// String returns the PTX suffix for the axis.
func (d Dim) String() string {
	axis, _ := d.axis()
	return axis
}

func (d Dim) axis() (string, error) {
	switch d {
	case DimX:
		return "x", nil
	case DimY:
		return "y", nil
	case DimZ:
		return "z", nil
	}
	return "", fmt.Errorf("cudacluster: invalid cluster dimension %d", d)
}
