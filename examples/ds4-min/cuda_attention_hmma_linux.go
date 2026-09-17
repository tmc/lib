//go:build linux

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const (
	tokentileTileTokens = 4
	tokentileG          = 8
	tokentileStageRows  = 32
	tokentileRawWindow  = 128
	tokentileHeadDim    = 512
	tokentileWarps      = 16
	tokentileThreads    = 512
	tokentileSmemBudget = 88576
)

func (r *cudaFoundation) encodeTokentileDenseBuild(
	records cudamem.Slice[byte],
	counts cudamem.Slice[uint32],
	pos0, nTokens, ratio, nComp, recStride int,
) error {
	if nTokens <= 0 || records.Len() < ((nTokens+tokentileTileTokens-1)/tokentileTileTokens)*recStride*8 {
		return fmt.Errorf("cuda: tokentile dense build: invalid shape")
	}
	nTiles := uint32((nTokens + tokentileTileTokens - 1) / tokentileTileTokens)
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.tokentileDenseBuildFn).
			Grid(nTiles).
			Block(tokentileThreads).
			On(r.stream()).
			Launch(records, counts, uint32(pos0), uint32(nTokens), uint32(ratio), uint32(nComp), uint32(recStride)); err != nil {
			return fmt.Errorf("cuda: launch tokentile dense build: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeTokentileRawMirror(
	dst cudamem.Slice[byte],
	raw, out cudamem.Slice[float32],
	basePosition, nTokens, rawCap, rawRowMin int,
) error {
	nMirrorRows := nTokens + tokentileRawWindow - 1
	if nTokens <= 0 || dst.Len() < nMirrorRows*tokentileHeadDim*2 {
		return fmt.Errorf("cuda: tokentile raw mirror: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.tokentileRawMirrorFn).
			Grid(uint32(nMirrorRows)).
			Block(256).
			On(r.stream()).
			Launch(dst, raw, out, uint32(basePosition), uint32(nTokens), uint32(rawCap), uint32(rawRowMin)); err != nil {
			return fmt.Errorf("cuda: launch tokentile raw mirror: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeTokentileCompMirror(
	dst cudamem.Slice[byte],
	compKV cudamem.Slice[float32],
	nComp int,
) error {
	if nComp <= 0 {
		return nil
	}
	if dst.Len() < nComp*tokentileHeadDim*2 || compKV.Len() < nComp*tokentileHeadDim {
		return fmt.Errorf("cuda: tokentile comp mirror: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.tokentileCompMirrorFn).
			Grid(uint32(nComp)).
			Block(256).
			On(r.stream()).
			Launch(dst, compKV, uint32(nComp)); err != nil {
			return fmt.Errorf("cuda: launch tokentile comp mirror: %w", err)
		}
		return nil
	})
}

func (r *cudaFoundation) encodeTokentileHMMA(
	heads cudamem.Slice[float32],
	sinks cudaModelTensor,
	q cudamem.Slice[float32],
	rawKV, compKV, records cudamem.Slice[byte],
	counts cudamem.Slice[uint32],
	basePosition, qStride, headsStride, recStride, nTokens, nHead, rawRowMin int,
) error {
	if nTokens <= 0 || nHead%tokentileG != 0 {
		return fmt.Errorf("cuda: tokentile hmma: invalid shape")
	}
	nTiles := uint32((nTokens + tokentileTileTokens - 1) / tokentileTileTokens)
	gridY := uint32(nHead / tokentileG)
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.tokentileHMMAFn).
			Grid(nTiles, gridY).
			Block(tokentileThreads).
			SharedMem(tokentileSmemBudget).
			On(r.stream()).
			Launch(
				heads,
				sinks.device,
				q,
				rawKV,
				compKV,
				records,
				counts,
				uint32(basePosition),
				uint32(qStride),
				uint32(headsStride),
				uint32(recStride),
				uint32(nTokens),
				uint32(nHead),
				uint32(rawRowMin),
			); err != nil {
			return fmt.Errorf("cuda: launch tokentile hmma: %w", err)
		}
		return nil
	})
}
