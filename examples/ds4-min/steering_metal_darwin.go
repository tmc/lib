//go:build darwin && arm64

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

//go:embed metal_steering.metal
var metalSteeringSource string

type metalSteering struct {
	r              *metalRuntime
	directions     metalmem.Slice[float32]
	library        apple.MTLLibrary
	pipeline       apple.MTLComputePipelineState
	attention, ffn float32
	identity       steeringIdentity
}

func (s *metalRequestState) loadSteering(v *steeringVectors, attention, ffn float32) error {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	if s.runtime.failed != nil {
		return s.runtime.failed
	}
	if err := steeringScale(attention); err != nil {
		return err
	}
	if err := steeringScale(ffn); err != nil {
		return err
	}
	if v == nil {
		if attention != 0 || ffn != 0 {
			return errors.New("steering requires loaded directions")
		}
		return nil
	}
	if v.layers != ds4Layers || v.width != ds4Width || len(v.values) != ds4Layers*ds4Width {
		return errors.New("steering vector shape mismatch")
	}
	if s.runtime.steering != nil {
		return errors.New("steering directions already loaded")
	}
	g := &metalSteering{r: s.runtime, attention: attention, ffn: ffn, identity: identityForSteering(v, attention, ffn)}
	if err := g.load(v); err != nil {
		return errors.Join(err, g.close())
	}
	s.runtime.steering = g
	return nil
}

func (g *metalSteering) load(v *steeringVectors) (err error) {
	g.directions, err = metalmem.AllocFromSlice(g.r.dev, v.values)
	if err != nil {
		return err
	}
	g.library, err = g.r.dev.CompileLibrary(metalSteeringSource)
	if err != nil {
		return err
	}
	fn := g.library.NewFunctionWithName("ds4_directional_project")
	if fn == nil || fn.GetID() == 0 {
		return errors.New("missing Metal steering function")
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	g.pipeline, err = g.r.dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	return err
}

func (g *metalSteering) project(b *metalBatch, x metalmem.Slice[float32], layer int, scale float32) error {
	if g == nil || scale == 0 {
		return nil
	}
	if b == nil || b.ops != g.r.ops || layer < 0 || layer >= ds4Layers || x.Len() != ds4Width || g.directions.Len() != ds4Layers*ds4Width {
		return errors.New("invalid Metal steering projection")
	}
	if err := b.ready(); err != nil {
		return err
	}
	nth := min(uint32(256), uint32(g.pipeline.MaxTotalThreadsPerThreadgroup()))
	if nth == 0 || nth&(nth-1) != 0 {
		return fmt.Errorf("invalid steering reduction width %d", nth)
	}
	args := struct {
		Width, Rows, Layer, Threads uint32
		Scale                       float32
	}{ds4Width, 1, uint32(layer), nth, scale}
	enc := b.encoder()
	enc.SetComputePipelineState(g.pipeline)
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&args)), unsafe.Sizeof(args)), 0)
	enc.SetBufferWithOffsetAtIndex(x.Buffer(), x.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(g.directions.Buffer(), g.directions.Offset(), 2)
	enc.SetThreadgroupMemoryLengthAtIndex(uint(nth*4), 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(int(nth)))
	b.encoded++
	return nil
}

func (g *metalSteering) close() error {
	if g == nil {
		return nil
	}
	if g.pipeline != nil && g.pipeline.GetID() != 0 {
		objectivec.ObjectFromID(g.pipeline.GetID()).Release()
		g.pipeline = nil
	}
	if g.library != nil && g.library.GetID() != 0 {
		objectivec.ObjectFromID(g.library.GetID()).Release()
		g.library = nil
	}
	if g.directions.Len() > 0 {
		g.directions.Free()
		g.directions = metalmem.Slice[float32]{}
	}
	return nil
}

func (s *metalRequestState) steeringFFN() float32 {
	if s == nil || s.runtime == nil || s.runtime.steering == nil {
		return 0
	}
	return s.runtime.steering.ffn
}
func (s *metalRequestState) setSteeringFFN(scale float32) error {
	if s == nil || s.closed || s.runtime == nil {
		return dsv4.ErrClosed
	}
	if s.runtime.failed != nil {
		return s.runtime.failed
	}
	if err := steeringScale(scale); err != nil {
		return err
	}
	if s.runtime.steering == nil {
		if scale != 0 {
			return errors.New("live FFN steering needs a directional vector loaded at startup")
		}
		return nil
	}
	if err := s.runtime.synchronizeState(); err != nil {
		return err
	}
	s.runtime.steering.ffn = scale
	s.runtime.steering.identity.ffn = scale
	return nil
}

func (r *metalRuntime) steeringStateIdentity() steeringIdentity {
	if r == nil || r.steering == nil {
		return steeringIdentity{}
	}
	return r.steering.identity
}
