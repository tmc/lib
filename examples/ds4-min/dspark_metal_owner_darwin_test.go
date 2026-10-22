//go:build darwin && arm64

package main

import (
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// ownerTestSlice supplies only shape metadata. A rejected owner must never
// inspect its nil buffer or issue a device command.
func ownerTestSlice[T any](n int) metalmem.Slice[T] {
	var s metalmem.Slice[T]
	f := reflect.ValueOf(&s).Elem().FieldByName("len")
	*(*int)(unsafe.Pointer(f.UnsafeAddr())) = n
	return s
}

func TestMetalDSparkTerminalOwners(t *testing.T) {
	w := &metalDSparkWeights{model: &dsparkModel{block: 1}}
	for _, tc := range []struct {
		name string
		d    *metalDSparkDense
		a    *metalDSparkAttention
	}{
		{"nil dense", nil, &metalDSparkAttention{weights: w}},
		{"closed dense", &metalDSparkDense{weights: w, closed: true}, &metalDSparkAttention{weights: w}},
		{"nil attention", &metalDSparkDense{weights: w}, nil},
		{"closed attention", &metalDSparkDense{weights: w}, &metalDSparkAttention{weights: w, closed: true}},
		{"nil weights", &metalDSparkDense{}, &metalDSparkAttention{}},
		{"nil model", &metalDSparkDense{weights: &metalDSparkWeights{}}, &metalDSparkAttention{weights: w}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("owner preflight panicked: %v", r)
				}
			}()
			err := tc.d.EncodeProposalHead(nil, tc.a, boundMetalTensor{dimensions: []uint64{4096, uint64(Flash.Vocab)}}, ownerTestSlice[float32](4096), ownerTestSlice[float32](Flash.Vocab), ownerTestSlice[int32](1), ownerTestSlice[float32](1), ownerTestSlice[float32](1), ownerTestSlice[uint32](2), 1, 0, .5)
			if err == nil || !strings.Contains(err.Error(), "owner") {
				t.Errorf("owner not refused first: %v", err)
			}
		})
	}
	for _, d := range []*metalDSparkDense{nil, {}, {weights: w, closed: true}, {weights: &metalDSparkWeights{}}} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("direct dense preflight panicked: %v", r)
				}
			}()
			err := d.encodeTensor(&metalBatch{ops: &metalPinnedOps{}}, boundMetalTensor{dimensions: []uint64{32, 1}}, ownerTestSlice[float32](32), ownerTestSlice[float32](1), 1)
			if err == nil || !strings.Contains(err.Error(), "owner") {
				t.Errorf("direct dense owner accepted: %v", err)
			}
		}()
	}
}
