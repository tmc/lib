//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_steering.cu
var cudaSteeringSource string

type cudaSteering struct {
	r                 *cudaRuntime
	directions        cudamem.Slice[float32]
	module            driver.Module
	function          driver.Function
	attention, ffn    float32
	identity          steeringIdentity
	releaseDirections func() error
}

func (s *cudaRequestState) loadSteering(v *steeringVectors, attention, ffn float32) error {
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
	if v.layers != cudaLayers || v.width != cudaWidth || len(v.values) != cudaLayers*cudaWidth {
		return errors.New("steering vector shape mismatch")
	}
	if s.runtime.steering != nil {
		return errors.New("steering directions already loaded")
	}
	g := &cudaSteering{r: s.runtime, attention: attention, ffn: ffn, identity: identityForSteering(v, attention, ffn)}
	if err := g.load(v); err != nil {
		return s.retainFailedSteering(g, err, g.close())
	}
	s.runtime.steering = g
	return nil
}

func (g *cudaSteering) load(v *steeringVectors) (err error) {
	g.directions, err = cudamem.Alloc[float32](g.r.device, len(v.values))
	if g.directions.Ptr() != 0 {
		g.releaseDirections = g.directions.Free
	}
	if err != nil {
		return err
	}
	if err = g.directions.CopyFromHost(v.values); err != nil {
		return err
	}
	compiled, err := nvrtcx.Compile(cudaSteeringSource, nvrtcx.WithName("cuda_steering.cu"), nvrtcx.WithDeviceArch(g.r.device))
	if err != nil {
		return err
	}
	g.module, _, err = compiled.LoadOutputWithLog(g.r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		return err
	}
	return g.r.device.WithContext(func() error { g.function, err = g.module.Function("ds4_directional_project"); return err })
}

func (g *cudaSteering) project(x cudamem.Slice[float32], layer int, scale float32) error {
	return g.projectRows(x, layer, 1, scale)
}

func (g *cudaSteering) projectRows(x cudamem.Slice[float32], layer, rows int, scale float32) error {
	if g == nil || scale == 0 {
		return nil
	}
	if layer < 0 || layer >= cudaLayers || rows <= 0 || rows > cudaPrefillGroupWidth || x.Len()%cudaWidth != 0 || x.Len()/cudaWidth != rows || g.directions.Len() != cudaLayers*cudaWidth {
		return errors.New("invalid CUDA steering projection")
	}
	return g.r.device.WithContext(func() error {
		return cudalaunch.Kernel(g.function).Grid(uint32(rows), 1, 1).Block(256, 1, 1).On(g.r.stream()).Launch(x, g.directions, uint32(layer), uint32(cudaWidth), uint32(rows), scale)
	})
}

func (g *cudaSteering) close() error {
	return g.closeWithContext(g.r.device.WithContext, func(module driver.Module) error { return module.Close() })
}

func (g *cudaSteering) closeWithContext(withContext func(func() error) error, unload func(driver.Module) error) error {
	return g.closeWithModuleUnload(func(module driver.Module) error {
		return withContext(func() error {
			err := unload(module)
			if err == nil {
				// Context restoration can fail after Unload succeeds. The
				// released handle must not survive that unrelated failure.
				g.module = 0
			}
			return err
		})
	})
}

func (g *cudaSteering) closeWithModuleUnload(unload func(driver.Module) error) error {
	if g == nil {
		return nil
	}
	var err error
	if g.module != 0 {
		if releaseErr := unload(g.module); releaseErr != nil {
			err = errors.Join(err, releaseErr)
		} else {
			g.module = 0
		}
	}
	if g.releaseDirections != nil {
		if releaseErr := g.releaseDirections(); releaseErr != nil {
			err = errors.Join(err, releaseErr)
		} else {
			g.directions = cudamem.Slice[float32]{}
			g.releaseDirections = nil
		}
	}
	return err
}

func (s *cudaRequestState) retainFailedSteering(g *cudaSteering, loadErr, cleanupErr error) error {
	err := errors.Join(loadErr, cleanupErr)
	if cleanupErr != nil {
		s.runtime.steering = g
		s.runtime.failed = errors.Join(err, dsv4.ErrDesynchronized)
		return s.runtime.failed
	}
	return err
}

func (r *cudaRuntime) closeSteering(close func(*cudaSteering) error) error {
	if r.steering == nil {
		return nil
	}
	if err := close(r.steering); err != nil {
		r.failed = errors.Join(err, dsv4.ErrClosed, dsv4.ErrDesynchronized)
		return err
	}
	r.steering = nil
	return nil
}

func (s *cudaRequestState) steeringFFN() float32 {
	if s == nil || s.runtime == nil || s.runtime.steering == nil {
		return 0
	}
	return s.runtime.steering.ffn
}
func (s *cudaRequestState) setSteeringFFN(scale float32) error {
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
			return fmt.Errorf("live FFN steering needs a directional vector loaded at startup")
		}
		return nil
	}
	return changeSteeringFFN(s.runtime.steering.ffn, scale, func() error { return syncStream(s.runtime.stream()) }, func() error {
		if err := s.runtime.invalidateSteeringGraphs(); err != nil {
			s.runtime.failed = errors.Join(err, dsv4.ErrDesynchronized)
			return s.runtime.failed
		}
		return nil
	}, func(scale float32) { s.runtime.steering.ffn = scale; s.runtime.steering.identity.ffn = scale })
}

func (r *cudaRuntime) invalidateSteeringGraphs() error {
	var errs []error
	sets := []*[cudaLayers]cudaLayerGraphs{&r.graphs}
	if r.batch != nil {
		sets = append(sets, &r.batch.graphs)
	}
	for _, set := range sets {
		for i := range set {
			if set[i].qkv != nil {
				if err := set[i].qkv.Destroy(); err != nil {
					errs = append(errs, err)
				} else {
					set[i].qkv = nil
				}
			}
			if set[i].ffn != nil {
				if err := set[i].ffn.Destroy(); err != nil {
					errs = append(errs, err)
				} else {
					set[i].ffn = nil
				}
			}
		}
	}
	return errors.Join(errs...)
}

func (r *cudaRuntime) steeringStateIdentity() steeringIdentity {
	if r == nil || r.steering == nil {
		return steeringIdentity{}
	}
	return r.steering.identity
}
