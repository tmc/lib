//go:build darwin && arm64

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

type ssdAddressOwner struct {
	device    *libmetal.Device
	arena     *metalmem.Arena
	weights   *metalStreamingWeights
	batch     *metalBatch
	encoder   apple.MTLComputeCommandEncoder
	submitted bool
	wait      func() error
	terminal  func() bool
}

var retainedSSDAddressOwners []*ssdAddressOwner

func (o *ssdAddressOwner) close() error {
	if o.submitted {
		if o.wait == nil || o.terminal == nil {
			return errors.New("missing SSD address consumer completion")
		}
		if err := o.wait(); err != nil {
			return err
		}
		if !o.terminal() {
			return errors.New("nonterminal SSD address consumer")
		}
	} else if o.batch != nil {
		o.batch.endEncoding()
	}
	// Consumer ownership drains before the retained command buffer is released.
	if err := o.weights.close(); err != nil {
		return err
	}
	o.weights = nil
	if o.arena != nil {
		o.arena.Free()
		o.arena = nil
	}
	if o.encoder != nil {
		objectivec.ObjectFromID(o.encoder.GetID()).Release()
		o.encoder = nil
	}
	if o.batch != nil {
		objectivec.ObjectFromID(o.batch.buffer.GetID()).Release()
		o.batch = nil
	}
	if o.device != nil {
		if err := o.device.Close(); err != nil {
			return err
		}
		o.device = nil
	}
	return nil
}

type ssdAddressResult struct {
	Cols, Rows int
	IDs        []int32
	Input      []float32
	DownInput  []float32
	Packed     [3][]byte
	Reference  [3][]float64
	Adjacent   []float64
	Output     [3][]float32
	Canaries   bool
}

func TestMetalSSDSelectedAddresses(t *testing.T) {
	if os.Getenv("DS4_METAL_SSD_ADDRESS_TEST") != "1" {
		t.Skip("requires separately reviewed SSD address numerical packet")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" || os.Getpagesize() != 16384 {
		t.Fatal("owning GPU lock and 16384-byte host pages required")
	}
	dir := os.Getenv("DS4_METAL_SSD_ADDRESS_OUTPUT")
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		t.Fatal("fresh artifact directory required")
	}
	for _, shape := range [][2]int{{256, 3}, {1024, 7}, {2048, 9}} {
		t.Run(fmt.Sprintf("%dx%d", shape[0], shape[1]), func(t *testing.T) {
			runSSDAddressFixture(t, dir, shape[0], shape[1])
		})
	}
}

func runSSDAddressFixture(t *testing.T, dir string, cols, rows int) {
	t.Helper()
	f, err := newSSDAddressFixture(cols, rows)
	if err != nil {
		t.Fatal(err)
	}
	o := new(ssdAddressOwner)
	defer func() {
		if err := o.close(); err != nil {
			retainedSSDAddressOwners = append(retainedSSDAddressOwners, o)
			t.Errorf("retained SSD address owner: %v", err)
		}
	}()
	o.device, err = libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	o.arena = metalmem.NewArena(o.device)
	source, cache, strides, err := f.source()
	if err != nil {
		t.Fatal(err)
	}
	o.weights, err = initMetalStreaming(o.device, source, cache, strides, 5, 5*16384, 8)
	if err != nil {
		t.Fatal(err)
	}
	var tensors [3]boundMetalTensor
	var mapped [3]metalmem.Slice[byte]
	for component, packed := range f.packed {
		mapped[component] = metalmem.AllocIn[byte](o.arena, "mapped canaries", len(packed)+32)
		if err := o.arena.Err(); err != nil {
			t.Fatal(err)
		}
		for i := range mapped[component].HostSlice() {
			mapped[component].HostSlice()[i] = 0xa5
		}
		copy(mapped[component].HostSlice()[16:], packed)
		data, err := mapped[component].Slice(16, 16+len(packed))
		if err != nil {
			t.Fatal(err)
		}
		typeID := uint32(16)
		if component == 2 {
			typeID = 10
		}
		tensors[component] = boundMetalTensor{typeID: typeID, dimensions: []uint64{uint64(cols), uint64(rows), 8}, data: data}
	}
	// Protect adjacent expert 2 in the same five-slot lease. It is absent from
	// all six actual consumer routes, but is a valid bounded mutant target.
	routes := append(append([]int32(nil), f.ids...), 2)
	if err := o.weights.prepare(0, tensors, routes); err != nil {
		t.Fatal(err)
	}
	selection := o.weights.selection
	var views [][3]metalmem.Slice[byte]
	var snapshots [][3][]byte
	for route := range 5 {
		v, err := selection.Views(route)
		if err != nil {
			t.Fatal(err)
		}
		var snapshot [3][]byte
		for component := range v {
			snapshot[component] = append([]byte(nil), v[component].HostSlice()...)
		}
		views, snapshots = append(views, v), append(snapshots, snapshot)
	}
	ids := metalmem.UploadIn(o.arena, "selected IDs", f.ids)
	in := metalmem.UploadIn(o.arena, "gate/up input", f.input)
	downIn := metalmem.UploadIn(o.arena, "slot-major down input", f.downInput)
	var full, output [3]metalmem.Slice[float32]
	const guard = uint32(0x4b2a1234)
	for component := range full {
		full[component] = metalmem.AllocIn[float32](o.arena, "output canaries", 6*rows+8)
		if err := o.arena.Err(); err != nil {
			t.Fatal(err)
		}
		for i := range full[component].HostSlice() {
			full[component].HostSlice()[i] = math.Float32frombits(guard)
		}
		output[component], err = full[component].Slice(4, 4+6*rows)
		if err != nil {
			t.Fatal(err)
		}
	}
	ops := &metalPinnedOps{dev: o.device, hc: new(metalHC)}
	o.batch, err = newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	objectivec.ObjectFromID(o.batch.buffer.GetID()).Retain()
	// Retain the encoder before any encode error can leave it with the owner.
	o.encoder = o.batch.encoder()
	objectivec.ObjectFromID(o.encoder.GetID()).Retain()
	if err := o.weights.encodeGate(o.batch, tensors, ids, in, output[0], output[1]); err != nil {
		t.Fatal(err)
	}
	if err := o.weights.encodeDown(o.batch, tensors[2], ids, downIn, output[2]); err != nil {
		t.Fatal(err)
	}
	if err := o.batch.Commit(); err != nil {
		t.Fatal(err)
	}
	o.submitted = true
	o.weights.consumer = o.batch.buffer
	o.wait = func() error { return libmetal.WaitCommandBuffer(context.Background(), o.batch.buffer) }
	o.terminal = func() bool {
		status := o.batch.buffer.Status()
		return status == apple.MTLCommandBufferStatusCompleted || status == apple.MTLCommandBufferStatusError
	}
	if err := o.batch.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	result := ssdAddressResult{Cols: cols, Rows: rows, IDs: f.ids, Input: f.input, DownInput: f.downInput, Packed: f.packed, Reference: f.want, Adjacent: f.adjacent, Canaries: true}
	for component := range output {
		result.Output[component] = append([]float32(nil), output[component].HostSlice()...)
		for _, i := range []int{0, 1, 2, 3, 6*rows + 4, 6*rows + 5, 6*rows + 6, 6*rows + 7} {
			result.Canaries = result.Canaries && math.Float32bits(full[component].HostSlice()[i]) == guard
		}
		raw := mapped[component].HostSlice()
		result.Canaries = result.Canaries && bytes.Equal(raw[16:16+len(f.packed[component])], f.packed[component])
		for _, v := range append(append([]byte(nil), raw[:16]...), raw[len(raw)-16:]...) {
			result.Canaries = result.Canaries && v == 0xa5
		}
		for route := range views {
			result.Canaries = result.Canaries && bytes.Equal(views[route][component].HostSlice(), snapshots[route][component])
		}
	}
	if err := o.weights.finish(); err != nil {
		t.Fatal(err)
	}
	if _, err := selection.Views(0); err == nil {
		t.Fatal("released SSD address view accepted")
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, fmt.Sprintf("%dx%d.json", cols, rows))
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, err = file.Write(encoded)
	if err := errors.Join(err, file.Close()); err != nil {
		t.Fatal(err)
	}
	if !result.Canaries {
		t.Fatal("SSD address canary failure")
	}
	for component := range output {
		for i, value := range result.Output[component] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || float64(value) != result.Reference[component][i] {
				t.Fatalf("SSD address numerical component=%d coordinate=%d got=%g want=%g", component, i, value, result.Reference[component][i])
			}
		}
	}
	t.Logf("device=%s fixture=%dx%d coordinates=%d dispatches=2 canaries=PASS output_sha256=%x", o.device.Name(), cols, rows, 18*rows, sha256.Sum256(encoded))
}

func TestSSDAddressOwnerLifecycle(t *testing.T) {
	t.Run("before-device", func(t *testing.T) {
		o := new(ssdAddressOwner)
		if err := o.close(); err != nil {
			t.Fatal(err)
		}
	})
	for _, scenario := range []string{"failed-wait", "nonterminal"} {
		t.Run(scenario, func(t *testing.T) {
			failure := errors.New("consumer pending")
			pending, terminal := scenario == "failed-wait", scenario != "nonterminal"
			w := new(metalStreamingWeights)
			o := &ssdAddressOwner{weights: w, submitted: true, wait: func() error {
				if pending {
					return failure
				}
				return nil
			}, terminal: func() bool { return terminal }}
			if err := o.close(); err == nil || o.weights != w {
				t.Fatal("pending consumer discarded owner", err)
			}
			pending, terminal = false, true
			if err := o.close(); err != nil || o.weights != nil {
				t.Fatal("terminal consumer did not release owner", err)
			}
		})
	}
}
