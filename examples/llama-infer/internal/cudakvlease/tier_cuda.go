package cudakvlease

import (
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaPages struct {
	*cudakv.PagedCache
	dev *cuda.Device
}

type cudaHostBuffer struct {
	allocation cudamem.HostAllocation[float32]
	dev        *cuda.Device
}

func (b *cudaHostBuffer) Slice() []float32 { return b.allocation.Slice() }
func (b *cudaHostBuffer) Free() error {
	err := b.dev.WithContext(b.allocation.Free)
	if err == nil {
		b.allocation = cudamem.HostAllocation[float32]{}
	}
	return err
}

func (p *cudaPages) AllocHost(n int) (hostBuffer, error) {
	var b cudamem.HostAllocation[float32]
	err := p.dev.WithContext(func() error {
		var err error
		b, err = cudamem.AllocHost[float32](n)
		return err
	})
	if err != nil {
		return nil, err
	}
	return &cudaHostBuffer{allocation: b, dev: p.dev}, nil
}

func (p *cudaPages) CopyPageToHost(layer, slot int, k, v []float32) error {
	page, err := p.Page(layer, slot)
	if err != nil {
		return err
	}
	return p.dev.WithContext(func() error {
		if err := page.K.CopyToHost(k); err != nil {
			return err
		}
		return page.V.CopyToHost(v)
	})
}

func (p *cudaPages) CopyHostToPage(layer, slot int, k, v []float32) error {
	page, err := p.Page(layer, slot)
	if err != nil {
		return err
	}
	return p.dev.WithContext(func() error {
		if err := page.K.CopyFromHost(k); err != nil {
			return err
		}
		return page.V.CopyFromHost(v)
	})
}
