//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_experts_mxfp4.cu
var cudaMXFP4SelectedSource string

// cudaMXFP4Selected owns only its module. Operand allocations and the foundation
// remain caller-owned; callers must retain them until stream completion.
// This primitive does not admit a model or implement a complete FFN.
type cudaMXFP4Selected struct {
	r        *cudaFoundation
	module   driver.Module
	fn       driver.Function
	lifetime mxfp4ModuleLifetime
}

func newCUDAMXFP4Selected(r *cudaFoundation) (*cudaMXFP4Selected, error) {
	return compileCUDAMXFP4Selected(r, cudaMXFP4SelectedSource)
}

func compileCUDAMXFP4Selected(r *cudaFoundation, source string) (*cudaMXFP4Selected, error) {
	if r == nil || r.device == nil {
		return nil, errors.New("cuda MXFP4: missing foundation")
	}
	s := &cudaMXFP4Selected{r: r}
	program, err := nvrtcx.Compile(source, nvrtcx.WithName("cuda_experts_mxfp4.cu"), nvrtcx.WithDeviceArch(r.device))
	if err != nil {
		return nil, err
	}
	s.module, _, err = program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if s.module != 0 {
		s.lifetime.synchronize = func() error { return r.device.WithContext(func() error { return syncStream(r.stream()) }) }
		s.lifetime.unload = func() error {
			if s.module == 0 {
				return nil
			}
			return mxfp4UnloadInContext(r.device.WithContext, s.module.Close, func() { s.module = 0; s.fn = 0 })
		}
	}
	if err == nil {
		err = r.device.WithContext(func() error { s.fn, err = s.module.Function("ds4_mxfp4_selected_q8_1"); return err })
	}
	if err != nil {
		if closeErr := s.Close(); closeErr != nil {
			return s, errors.Join(err, closeErr)
		}
		return nil, err
	}
	return s, nil
}

func (s *cudaMXFP4Selected) Close() error {
	if s == nil {
		return nil
	}
	if err := s.lifetime.close(); err != nil {
		return err
	}
	s.module = 0
	s.fn = 0
	return nil
}

func (s *cudaMXFP4Selected) Encode(t cudaModelTensor, ids cudamem.Slice[int32], input cudamem.Slice[byte], output cudamem.Slice[float32], slotted bool) error {
	if s == nil || s.lifetime.closed || s.module == 0 {
		return errors.New("cuda MXFP4: closed consumer")
	}
	if t.typeID != typeMXFP4 || t.device == 0 || len(t.dimensions) != 3 {
		return errors.New("cuda MXFP4: invalid tensor")
	}
	k, m, e := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	n := uint64(ids.Len())
	inputs := uint64(1)
	if slotted {
		inputs = n
	}
	if k == 0 || k%32 != 0 || k > math.MaxUint32 || m == 0 || m > math.MaxUint32 || e == 0 || e > math.MaxInt32 || n == 0 || n > 6 || m > math.MaxInt32/n || k/32 > math.MaxUint64/17/m/e {
		return errors.New("cuda MXFP4: invalid geometry")
	}
	if t.bytes != k/32*17*m*e || uint64(input.Len()) != k/32*36*inputs || uint64(output.Len()) != m*n {
		return errors.New("cuda MXFP4: invalid buffer shape")
	}
	if err := checkMXFP4DeviceRanges(
		mxfp4DeviceRange{uint64(t.device), t.bytes},
		mxfp4DeviceRange{uint64(ids.Ptr()), n * 4},
		mxfp4DeviceRange{uint64(input.Ptr()), uint64(input.Len())},
		mxfp4DeviceRange{uint64(output.Ptr()), uint64(output.Len()) * 4},
	); err != nil {
		return err
	}
	var perSlot uint32
	if slotted {
		perSlot = 1
	}
	return s.r.device.WithContext(func() error {
		if err := cudalaunch.Kernel(s.fn).Grid(uint32(m), uint32(n), 1).Block(32, 1, 1).On(s.r.stream()).Launch(t.device, ids, input, output, uint32(k), uint32(m), uint32(e), uint32(n), perSlot); err != nil {
			return fmt.Errorf("cuda MXFP4: launch: %w", err)
		}
		return nil
	})
}
