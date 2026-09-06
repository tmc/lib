//go:build !darwin

package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// UploadRetainedModelToGPU uploads a model that was loaded in its source
// dtypes, storing float32 weights on the device.
//
// This is the reference upload, and it is what the engine uses unless a half
// mode is selected explicitly.
func UploadRetainedModelToGPU(dev *cuda.Device, tracker *cudamem.Tracker, m *RetainedModel, cfg *ModelConfig) (*GPUModelWeights, error) {
	return UploadRetainedModelToGPUAs(dev, tracker, m, cfg, WeightF32)
}

// UploadRetainedModelToGPUAs uploads a retained model with its weights stored
// on the device in dtype d.
//
// A tensor whose checkpoint dtype already matches d is transferred as its
// source bytes, so a BF16 checkpoint uploaded as bf16 is never widened and
// never rounded: the bytes on the device are the bytes in the file. A tensor in
// any other dtype is widened once and rounded once here, at the upload
// boundary. Either way the device holds exactly d.Bytes() bytes per weight, and
// no float32 copy of the model is materialized for a half upload.
func UploadRetainedModelToGPUAs(dev *cuda.Device, tracker *cudamem.Tracker, m *RetainedModel, cfg *ModelConfig, d WeightDType) (*GPUModelWeights, error) {
	if err := d.check(); err != nil {
		return nil, err
	}
	arena := cudamem.NewTrackedArena(dev, tracker)
	committed := false
	defer func() {
		if !committed {
			_ = arena.Free()
		}
	}()
	upload := func(name string, t *RetainedTensor) (WeightSlice, error) {
		s, err := uploadRetainedWeights(arena, d, t)
		if err != nil {
			return WeightSlice{}, fmt.Errorf("upload %s: %w", name, err)
		}
		return s, nil
	}

	embed, err := upload("embed_tokens", m.EmbedTokens)
	if err != nil {
		return nil, err
	}
	norm, err := upload("norm", m.Norm)
	if err != nil {
		return nil, err
	}
	lmHead := embed
	if m.LMHead != m.EmbedTokens {
		if lmHead, err = upload("lm_head", m.LMHead); err != nil {
			return nil, err
		}
	}

	layers := make([]GPULayerWeights, len(m.Layers))
	for i := range m.Layers {
		l := &m.Layers[i]
		for _, f := range []struct {
			dst  *WeightSlice
			name string
			src  *RetainedTensor
		}{
			{&layers[i].InputNorm, "input_norm", l.InputNorm},
			{&layers[i].PostNorm, "post_norm", l.PostNorm},
			{&layers[i].WQ, "wq", l.WQ},
			{&layers[i].WK, "wk", l.WK},
			{&layers[i].WV, "wv", l.WV},
			{&layers[i].QBias, "q_bias", l.QBias},
			{&layers[i].KBias, "k_bias", l.KBias},
			{&layers[i].VBias, "v_bias", l.VBias},
			{&layers[i].WO, "wo", l.WO},
			{&layers[i].WGate, "wgate", l.WGate},
			{&layers[i].WUp, "wup", l.WUp},
			{&layers[i].WDown, "wdown", l.WDown},
		} {
			s, err := upload(fmt.Sprintf("layer %d %s", i, f.name), f.src)
			if err != nil {
				return nil, err
			}
			*f.dst = s
		}
	}

	committed = true
	return &GPUModelWeights{
		EmbedTokens: embed,
		Layers:      layers,
		Norm:        norm,
		LMHead:      lmHead,
		dtype:       d,
		arena:       arena,
	}, nil
}
