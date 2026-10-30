package main

import (
	"bytes"
	"errors"
	"io"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

var errAfmoeTest = errors.New("injected afmoe failure")

type afmoeTestBuffer struct {
	data     []byte
	failFree bool
	freed    int
}

func (b *afmoeTestBuffer) Ptr() driver.DevicePtr { return 1 }
func (b *afmoeTestBuffer) Free() error {
	if b.failFree {
		b.failFree = false
		return errAfmoeTest
	}
	b.freed++
	return nil
}

func afmoeToyCheckpoint(t *testing.T) (*safetensors.File, []afmoeTensor) {
	t.Helper()
	var data bytes.Buffer
	var tensors []safetensors.Tensor
	for _, tt := range []struct {
		name  string
		shape []int
		v     []float32
	}{{"matrix", []int{2, 3}, []float32{1, -2, 3, 4, 5, 6}}, {"norm", []int{3}, []float32{1, 1.5, .25}}} {
		b, err := safetensors.EncodeFloat32(safetensors.BF16, tt.v)
		if err != nil {
			t.Fatal(err)
		}
		tensors = append(tensors, safetensors.Tensor{Name: tt.name, Dtype: safetensors.BF16, Shape: tt.shape, Data: b})
	}
	if err := safetensors.Write(&data, tensors, nil); err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.NewFile(bytes.NewReader(data.Bytes()), int64(data.Len()))
	if err != nil {
		t.Fatal(err)
	}
	return f, []afmoeTensor{{"matrix", []int{2, 3}}, {"norm", []int{3}}}
}

func TestAfmoeSharedShardStream(t *testing.T) {
	ck, specs := afmoeToyCheckpoint(t)
	d := newMemDevice()
	w, err := streamAfmoeWeightsWith(ck, specs, make([]byte, 8), d.allocMatrix, d.write)
	if err != nil {
		t.Fatal(err)
	}
	if w.Tensors["matrix"].dtype != weightBF16 || w.Tensors["norm"].dtype != weightF32 {
		t.Fatal("matrices must stay BF16 while norm values widen")
	}
	wantMatrix := wantBytes(t, safetensors.BF16, weightBF16, []float32{1, -2, 3, 4, 5, 6})
	wantNorm := wantBytes(t, safetensors.BF16, weightF32, []float32{1, 1.5, .25})
	if !bytes.Equal(bytesOf(w.Tensors["matrix"]), wantMatrix) || !bytes.Equal(bytesOf(w.Tensors["norm"]), wantNorm) {
		t.Fatal("streaming changed tensor values or layout")
	}
	if len(d.mats) != 2 || len(d.vecs) != 0 || len(d.bad) != 0 {
		t.Fatalf("unexpected storage: matrices=%d vectors=%d errors=%v", len(d.mats), len(d.vecs), d.bad)
	}
	if err := w.Free(); err != nil {
		t.Fatal(err)
	}
	if err := w.Free(); err != nil || len(w.Tensors) != 0 || len(d.freed) != 2 {
		t.Fatalf("free retry: tensors=%d frees=%v error=%v", len(w.Tensors), d.freed, err)
	}
}

func TestAfmoeStreamPreflight(t *testing.T) {
	ck, specs := afmoeToyCheckpoint(t)
	specs[1].Shape = []int{4}
	d := newMemDevice()
	if _, err := streamAfmoeWeightsWith(ck, specs, make([]byte, 8), d.allocMatrix, d.write); err == nil {
		t.Fatal("accepted a shape mismatch")
	}
	if len(d.mats) != 0 {
		t.Fatal("allocated before checking every tensor header")
	}
}

func TestAfmoeStreamRetainsFailedCleanup(t *testing.T) {
	ck, specs := afmoeToyCheckpoint(t)
	buf := &afmoeTestBuffer{failFree: true}
	w, err := streamAfmoeWeightsWith(ck, specs[:1], make([]byte, 8),
		func(n int, dtype weightDType) (deviceWeight, error) {
			return deviceWeight{buf: buf, dtype: dtype}, errAfmoeTest
		}, func(deviceBuffer, int, []byte) error { t.Fatal("write after allocation error"); return nil })
	if !errors.Is(err, errAfmoeTest) || w == nil || len(w.Tensors) != 1 {
		t.Fatalf("lost failed owner: weights=%v error=%v", w, err)
	}
	if err := w.Free(); err != nil || buf.freed != 1 || len(w.Tensors) != 0 {
		t.Fatalf("cleanup retry: freed=%d weights=%v error=%v", buf.freed, w, err)
	}
}

func TestAfmoeStreamWriteFailureCleans(t *testing.T) {
	ck, specs := afmoeToyCheckpoint(t)
	d := newMemDevice()
	d.writeFail = 1
	w, err := streamAfmoeWeightsWith(ck, specs, make([]byte, 8), d.allocMatrix, d.write)
	if err == nil || w != nil || !slices.Equal(d.freed, []string{"m0"}) {
		t.Fatalf("write cleanup: weights=%v error=%v frees=%v", w, err, d.freed)
	}
}

// afmoeChangedReader models a shard truncated after metadata inspection.
type afmoeChangedReader struct {
	checkpoint
	data []byte
	size int64
}

func (c afmoeChangedReader) TensorReader(name string) (*io.SectionReader, error) {
	return io.NewSectionReader(bytes.NewReader(c.data), 0, c.size), nil
}

func TestAfmoeStreamRejectsChangedPayload(t *testing.T) {
	ck, specs := afmoeToyCheckpoint(t)
	for _, tt := range []struct {
		name string
		data int
		size int64
	}{{"truncated_after_open", 8, 12}, {"extra_section", 14, 14}} {
		t.Run(tt.name, func(t *testing.T) {
			d := newMemDevice()
			changed := afmoeChangedReader{ck, make([]byte, tt.data), tt.size}
			w, err := streamAfmoeWeightsWith(changed, specs[:1], make([]byte, 4), d.allocMatrix, d.write)
			if err == nil || w != nil || !slices.Equal(d.freed, []string{"m0"}) {
				t.Fatalf("payload mutation accepted or owner lost: weights=%v error=%v frees=%v", w, err, d.freed)
			}
		})
	}
}

func TestAfmoeStreamMidUploadFailureRetainsOwner(t *testing.T) {
	ck, specs := afmoeToyCheckpoint(t)
	buf := &afmoeTestBuffer{data: make([]byte, 12), failFree: true}
	writes := 0
	w, err := streamAfmoeWeightsWith(ck, specs[:1], make([]byte, 4),
		func(int, weightDType) (deviceWeight, error) { return deviceWeight{buf: buf, dtype: weightBF16}, nil },
		func(_ deviceBuffer, off int, b []byte) error {
			writes++
			if writes == 2 {
				return errAfmoeTest
			}
			copy(buf.data[off:], b)
			return nil
		})
	if !errors.Is(err, errAfmoeTest) || writes != 2 || w == nil || len(w.Tensors) != 1 {
		t.Fatalf("mid-upload owner: weights=%v writes=%d error=%v", w, writes, err)
	}
	if err := w.Free(); err != nil || buf.freed != 1 || len(w.Tensors) != 0 {
		t.Fatalf("retry owner: freed=%d weights=%v error=%v", buf.freed, w, err)
	}
}
