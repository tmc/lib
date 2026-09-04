//go:build darwin && arm64

package main

import (
	"fmt"
	"os"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

type metalModelView struct {
	fileOffset uint64
	data       metalmem.Slice[byte]
}

type metalModelViews struct {
	views []metalModelView
}

type boundMetalTensor struct {
	name       string
	typeID     uint32
	dimensions []uint64
	data       metalmem.Slice[byte]
}

func (t boundMetalTensor) Rows(start, count int) (boundMetalTensor, error) {
	if len(t.dimensions) != 2 || start < 0 || count <= 0 || uint64(start+count) > t.dimensions[1] {
		return boundMetalTensor{}, fmt.Errorf("tensor %q: invalid row range", t.name)
	}
	total, err := tensorBytes(ggufTensor{name: t.name, typeID: t.typeID, dimensions: t.dimensions})
	if err != nil {
		return boundMetalTensor{}, err
	}
	rowBytes := int(total / t.dimensions[1])
	data, err := t.data.Slice(start*rowBytes, (start+count)*rowBytes)
	if err != nil {
		return boundMetalTensor{}, err
	}
	return boundMetalTensor{t.name, t.typeID, []uint64{t.dimensions[0], uint64(count)}, data}, nil
}

func (t boundMetalTensor) Expert(id int32) (boundMetalTensor, error) {
	if len(t.dimensions) != 3 || id < 0 || uint64(id) >= t.dimensions[2] {
		return boundMetalTensor{}, fmt.Errorf("tensor %q: invalid expert %d", t.name, id)
	}
	total, err := tensorBytes(ggufTensor{name: t.name, typeID: t.typeID, dimensions: t.dimensions})
	if err != nil {
		return boundMetalTensor{}, err
	}
	expertBytes := int(total / t.dimensions[2])
	data, err := t.data.Slice(int(id)*expertBytes, (int(id)+1)*expertBytes)
	if err != nil {
		return boundMetalTensor{}, err
	}
	return boundMetalTensor{t.name, t.typeID, append([]uint64(nil), t.dimensions[:2]...), data}, nil
}

func (m *metalModelViews) Bind(g *ggufFile, name string) (boundMetalTensor, error) {
	if g == nil {
		return boundMetalTensor{}, fmt.Errorf("bind tensor %q: missing GGUF", name)
	}
	for _, tensor := range g.tensors {
		if tensor.name != name {
			continue
		}
		n, err := tensorBytes(tensor)
		if err != nil {
			return boundMetalTensor{}, err
		}
		data, err := m.Tensor(g.dataOffset+tensor.offset, n)
		if err != nil {
			return boundMetalTensor{}, fmt.Errorf("bind tensor %q: %w", name, err)
		}
		return boundMetalTensor{name, tensor.typeID, append([]uint64(nil), tensor.dimensions...), data}, nil
	}
	return boundMetalTensor{}, fmt.Errorf("bind tensor %q: not found", name)
}

func mapMetalModel(dev *libmetal.Device, f *os.File, fileBytes, maxTensorBytes uint64) (*metalModelViews, error) {
	if dev == nil || f == nil {
		return nil, fmt.Errorf("map model: missing device or file")
	}
	page := uint64(os.Getpagesize())
	maxBuffer := uint64(dev.Raw().MaxBufferLength()) &^ (page - 1)
	overlap, err := checkedRoundPage(maxTensorBytes, page)
	if err != nil || overlap > ^uint64(0)-page {
		return nil, fmt.Errorf("map model: tensor span overflow")
	}
	overlap += page
	if fileBytes == 0 || maxBuffer <= overlap {
		return nil, fmt.Errorf("map model: Metal maximum %d cannot cover tensor span %d", maxBuffer, maxTensorBytes)
	}
	step := maxBuffer - overlap
	m := &metalModelViews{}
	for off := uint64(0); off < fileBytes; {
		length := min(maxBuffer, fileBytes-off)
		if off > uint64(^uint64(0)>>1) || length > uint64(^uint(0)>>1) {
			m.Close()
			return nil, fmt.Errorf("map model: view range overflows int")
		}
		data, err := metalmem.AllocMMapZeroCopy[byte](dev, f, int64(off), int(length))
		if err != nil {
			m.Close()
			return nil, fmt.Errorf("map model view at %d: %w", off, err)
		}
		m.views = append(m.views, metalModelView{fileOffset: off, data: data})
		if length == fileBytes-off {
			break
		}
		off += step
	}
	return m, nil
}

func checkedRoundPage(n, page uint64) (uint64, error) {
	if page == 0 || page&(page-1) != 0 || n > ^uint64(0)-(page-1) {
		return 0, errShape
	}
	return (n + page - 1) &^ (page - 1), nil
}

func (m *metalModelViews) Tensor(fileOffset, length uint64) (metalmem.Slice[byte], error) {
	if m == nil || length == 0 || fileOffset > ^uint64(0)-length {
		return metalmem.Slice[byte]{}, fmt.Errorf("model tensor: invalid range")
	}
	end := fileOffset + length
	for _, view := range m.views {
		viewEnd := view.fileOffset + uint64(view.data.Len())
		if fileOffset < view.fileOffset || end > viewEnd {
			continue
		}
		start := int(fileOffset - view.fileOffset)
		return view.data.Slice(start, start+int(length))
	}
	return metalmem.Slice[byte]{}, fmt.Errorf("model tensor: range [%d,%d) crosses mapped views", fileOffset, end)
}

func (m *metalModelViews) Close() {
	if m == nil {
		return
	}
	for i := range m.views {
		m.views[i].data.Free()
	}
	m.views = nil
}
