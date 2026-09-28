package cublasltx

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
)

// Heuristic returns up to n candidate algorithms for the product described by
// desc and the layouts of A, B, C and D, in the order cuBLASLt's heuristic
// ranks them, fastest estimate first. It returns an empty slice if the
// heuristic lists none.
//
// Heuristic does not make a CUDA context current; the caller must, as for
// any call on h.
func Heuristic(h cublaslt.Handle, desc cublaslt.MatmulDesc, a, b, c, d cublaslt.MatrixLayout, pref cublaslt.MatmulPreference, n int) ([]cublaslt.MatmulHeuristicResult, error) {
	if h == 0 {
		return nil, errors.New("cublasltx: nil handle")
	}
	if n <= 0 || n > math.MaxInt32 {
		return nil, fmt.Errorf("cublasltx: candidate count %d out of range", n)
	}
	rs := make([]cublaslt.MatmulHeuristicResult, n)
	got, err := h.MatmulAlgoGetHeuristic(desc, a, b, c, d, pref, int32(n), &rs[0])
	if err != nil {
		return nil, fmt.Errorf("cublasltx: get heuristic: %w", err)
	}
	return rs[:got], nil
}

// AlgoConfig is the configuration of a cuBLASLt algorithm: the fields that
// decide which kernel a product runs.
type AlgoConfig struct {
	ID              int32
	Tile            cublaslt.MatmulTile
	Stages          cublaslt.MatmulStages
	CustomOption    int32
	SplitK          int32 // number of split-k slices; 0 or 1 means none
	ReductionScheme cublaslt.ReductionScheme
}

// AlgoConfigOf returns the configuration of algo.
func AlgoConfigOf(algo *cublaslt.MatmulAlgo) (AlgoConfig, error) {
	if algo == nil {
		return AlgoConfig{}, errors.New("cublasltx: nil algorithm")
	}
	var c AlgoConfig
	var pin runtime.Pinner
	pin.Pin(&c)
	defer pin.Unpin()
	for _, f := range []struct {
		attr cublaslt.MatmulAlgoConfigAttributes
		v    unsafe.Pointer
	}{
		{cublaslt.AlgoConfigId, unsafe.Pointer(&c.ID)},
		{cublaslt.AlgoConfigTileId, unsafe.Pointer(&c.Tile)},
		{cublaslt.AlgoConfigStagesId, unsafe.Pointer(&c.Stages)},
		{cublaslt.AlgoConfigCustomOption, unsafe.Pointer(&c.CustomOption)},
		{cublaslt.AlgoConfigSplitkNum, unsafe.Pointer(&c.SplitK)},
		{cublaslt.AlgoConfigReductionScheme, unsafe.Pointer(&c.ReductionScheme)},
	} {
		if _, err := cublaslt.MatmulAlgoConfigGetAttribute(algo, f.attr, f.v, 4); err != nil {
			return AlgoConfig{}, fmt.Errorf("cublasltx: get algorithm attribute %v: %w", f.attr, err)
		}
	}
	return c, nil
}

// Pick returns the index of the first of cands whose configuration match
// accepts, or 0 if none does or match is nil. A candidate whose
// configuration cannot be read never matches. The result depends only on
// cands, so a product planned twice on the same device and library runs
// the same kernel. Pick returns an error if cands is empty.
func Pick(cands []cublaslt.MatmulHeuristicResult, match func(AlgoConfig) bool) (int, error) {
	return pick(len(cands), func(i int) (AlgoConfig, error) { return AlgoConfigOf(&cands[i].Algo) }, match)
}

func pick(n int, config func(int) (AlgoConfig, error), match func(AlgoConfig) bool) (int, error) {
	if n == 0 {
		return 0, errors.New("cublasltx: no candidate algorithms")
	}
	if match == nil {
		return 0, nil
	}
	for i := range n {
		if c, err := config(i); err == nil && match(c) {
			return i, nil
		}
	}
	return 0, nil
}

// Version returns the version of the loaded cuBLASLt library.
func Version() (major, minor, patch int, err error) {
	var v [3]int32
	for i, p := range []cublaslt.LibraryPropertyType{cublaslt.MajorVersion, cublaslt.MinorVersion, cublaslt.PatchLevel} {
		if err := cublaslt.GetProperty(p, &v[i]); err != nil {
			return 0, 0, 0, fmt.Errorf("cublasltx: get version: %w", err)
		}
	}
	return int(v[0]), int(v[1]), int(v[2]), nil
}
