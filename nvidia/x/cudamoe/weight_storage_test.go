package cudamoe

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestMoEWeightRowPermutation(t *testing.T) {
	for _, intermediate := range []int{8, 16, 24} {
		for _, columns := range []int{1, 17, 32} {
			src := make([]int, 2*intermediate*columns)
			for i := range src {
				src[i] = i
			}
			got, err := interleaveMoEWeightRows(src, intermediate, columns)
			if err != nil {
				t.Fatal(err)
			}
			// Construct the specified eight-row blocks independently, rather than
			// using the production mapping to define its expected permutation.
			var want []int
			for block := 0; block < intermediate; block += 8 {
				for _, half := range []int{0, intermediate} {
					want = append(want, src[(half+block)*columns:(half+block+8)*columns]...)
				}
			}
			if !slices.Equal(got, want) {
				t.Fatalf("I=%d H=%d: incorrect permutation", intermediate, columns)
			}
			for row := 0; row < 2*intermediate; row++ {
				physical := moeWeightPhysicalRow(row, intermediate)
				if moeWeightLogicalRow(physical, intermediate) != row {
					t.Fatal("row mapping is not reversible")
				}
			}
		}
	}
}

func TestMoEWeightScalePermutation(t *testing.T) {
	weights := make([]int, 32*17)
	scales := make([]int, 32)
	for row := range scales {
		scales[row] = row + 1
		for column := 0; column < 17; column++ {
			weights[row*17+column] = scales[row]
		}
	}
	w, err := interleaveMoEWeightRows(weights, 16, 17)
	if err != nil {
		t.Fatal(err)
	}
	s, err := interleaveMoEWeightRows(scales, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	for row, scale := range s {
		for column := 0; column < 17; column++ {
			if w[row*17+column] != scale {
				t.Fatal("weight and scale rows differ")
			}
		}
	}
	if slices.Equal(s, scales) {
		t.Fatal("scale permutation control is vacuous")
	}
}

func TestMoEWeightViewOffsets(t *testing.T) {
	l, err := newMoEWeightLayout(MoEWeightShape{3, 2, 16, 17, []int{0, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if l.weightBytes != 3*2*32*17 || l.scaleRows != 3*2*32 {
		t.Fatal("slab does not preserve padded global layer")
	}
	for _, tt := range []struct{ layer, expert, weights, scales int }{
		{0, 0, 0, 0}, {0, 1, 544, 32}, {2, 0, 2176, 128}, {2, 1, 2720, 160},
	} {
		w, s, err := l.offsets(tt.layer, tt.expert)
		if err != nil || w != tt.weights || s != tt.scales {
			t.Fatalf("layer=%d expert=%d: got %d,%d,%v", tt.layer, tt.expert, w, s, err)
		}
	}
}

func TestMoEWeightGlobalLayerMap(t *testing.T) {
	active := []int{2, 0}
	l, err := newMoEWeightLayout(MoEWeightShape{3, 2, 8, 32, active})
	if err != nil {
		t.Fatal(err)
	}
	active[0] = 1
	if l.shape.ActiveLayers[0] != 2 || l.active[1] {
		t.Fatal("caller mutation changes admitted global layers")
	}
	for _, tt := range []struct{ layer, expert int }{{1, 0}, {-1, 0}, {3, 0}, {0, -1}, {0, 2}} {
		if _, _, err := l.offsets(tt.layer, tt.expert); err == nil {
			t.Fatalf("admitted unreadable layer/expert %v", tt)
		}
	}
}

func TestMoEWeightInvalidShape(t *testing.T) {
	for _, shape := range []MoEWeightShape{
		{}, {3, 2, 7, 17, []int{0}}, {3, 2, 8, 17, nil},
		{3, 2, 8, 17, []int{0, 0}}, {3, 2, 8, 17, []int{3}},
		{math.MaxInt, 2, 8, 17, []int{0}}, {3, math.MaxInt, 8, 17, []int{0}},
		{3, 2, math.MaxInt - 7, 17, []int{0}}, {3, 2, 8, math.MaxInt, []int{0}},
	} {
		if _, err := newMoEWeightLayout(shape); err == nil {
			t.Fatalf("accepted invalid shape %+v", shape)
		}
	}
	for _, tt := range []struct{ rows, columns, length int }{{0, 1, 0}, {7, 1, 14}, {8, 0, 0}, {8, 1, 15}, {math.MaxInt - 7, math.MaxInt, 0}} {
		if _, err := interleaveMoEWeightRows(make([]byte, tt.length), tt.rows, tt.columns); err == nil {
			t.Fatalf("accepted invalid permutation %v", tt)
		}
	}
}

func TestMoEWeightE4M3FN(t *testing.T) {
	for raw := 0; raw < 256; raw++ {
		got, err := decodeMoEWeight(byte(raw))
		if raw&127 == 127 {
			if err == nil {
				t.Fatalf("accepted NaN %02x", raw)
			}
			continue
		}
		// Independent IEEE FP32 construction of the normal significand; the
		// subnormal FP8 mantissas are exact multiples of 1/512 in FP32.
		e, fraction := (raw>>3)&15, raw&7
		var want float32
		if e == 0 {
			want = float32(fraction) / 512
		} else {
			want = math.Float32frombits(uint32(e-7+127)<<23 | uint32(fraction)<<20)
		}
		bits := math.Float32bits(want) | uint32(raw&128)<<24
		if err != nil || math.Float32bits(got) != bits {
			t.Fatalf("decode %02x: got %08x,%v want %08x", raw, math.Float32bits(got), err, bits)
		}
	}
	for _, tt := range []struct {
		raw  byte
		want float32
	}{{0x01, 1.0 / 512}, {0x08, 1.0 / 64}, {0x38, 1}, {0x7e, 448}, {0xfe, -448}} {
		got, err := decodeMoEWeight(tt.raw)
		if err != nil || got != tt.want {
			t.Fatalf("FP8 landmark %02x: got %v,%v", tt.raw, got, err)
		}
	}
	for _, scale := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if err := validateMoEWeightPayload([]byte{0x38}, []float32{scale}); err == nil {
			t.Fatal("accepted nonfinite scale")
		}
	}
	for _, raw := range []byte{0x7f, 0xff} {
		if err := validateMoEWeightPayload([]byte{raw}, []float32{1}); err == nil {
			t.Fatal("accepted nonfinite payload")
		}
	}
}

// This backend is a labeled host ownership simulation. Native tests must also
// observe real cudamem allocations, borrows and successful CUDA completion.
type weightStorageHostMemory struct {
	bytes                []byte
	ptr                  driver.DevicePtr
	users, frees, writes int
	failWrite            error
}

func (m *weightStorageHostMemory) buffer() *moeWeightBuffer {
	return &moeWeightBuffer{
		ptr: func() driver.DevicePtr { return m.ptr },
		write: func(offset int, b []byte) error {
			m.writes++
			if m.failWrite != nil {
				return m.failWrite
			}
			if m.ptr == 0 || offset < 0 || offset > len(m.bytes) || len(b) > len(m.bytes)-offset {
				return errors.New("invalid simulated write")
			}
			copy(m.bytes[offset:], b)
			return nil
		},
		borrow: func() (func(), error) {
			if m.ptr == 0 {
				return nil, errors.New("simulated storage freed")
			}
			m.users++
			return func() { m.users-- }, nil
		},
		free: func() error {
			if m.users != 0 {
				return cudamem.ErrInUse
			}
			if m.ptr != 0 {
				m.frees++
				m.ptr = 0
			}
			return nil
		},
	}
}

func weightStorageHost(t *testing.T) (*MoEWeightStorage, []*weightStorageHostMemory) {
	t.Helper()
	var buffers []*weightStorageHostMemory
	s, err := newMoEWeightStorage(&cuda.Device{}, MoEWeightShape{3, 2, 16, 17, []int{0, 2}}, func(n int) (*moeWeightBuffer, error) {
		m := &weightStorageHostMemory{bytes: make([]byte, n), ptr: driver.DevicePtr(0x100000 + len(buffers)*0x100000)}
		buffers = append(buffers, m)
		return m.buffer(), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return s, buffers
}

func weightStoragePayload(s *MoEWeightStorage) ([]byte, []byte) {
	rows := 2 * s.layout.shape.Intermediate
	w := bytes.Repeat([]byte{0x38}, rows*s.layout.shape.Hidden)
	scale := make([]byte, rows*4)
	for i := 0; i < rows; i++ {
		binary.LittleEndian.PutUint32(scale[i*4:], math.Float32bits(float32(i+1)))
	}
	return w, scale
}

func TestMoEWeightSharedBacking(t *testing.T) {
	s, mem := weightStorageHost(t)
	w, scale := weightStoragePayload(s)
	var views []*MoEWeightView
	for _, layer := range []int{0, 2} {
		for expert := 0; expert < 2; expert++ {
			if err := s.Load(layer, expert, bytes.NewReader(w), bytes.NewReader(scale)); err != nil {
				t.Fatal(err)
			}
			v, err := s.View(layer, expert)
			if err != nil {
				t.Fatal(err)
			}
			views = append(views, v)
			d, err := v.Descriptor()
			if err != nil {
				t.Fatal(err)
			}
			if d.WeightBase != mem[0].ptr || d.ScaleBase != mem[1].ptr || d.Device != s.device {
				t.Fatal("view uses private or different device storage")
			}
		}
	}
	if err := s.Close(); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatalf("premature Close: %v", err)
	}
	for _, v := range views {
		if err := v.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	for _, m := range mem {
		if m.frees != 1 || m.users != 0 {
			t.Fatalf("base cleanup: %+v", m)
		}
	}
}

func TestMoEWeightLoadFailureOwnership(t *testing.T) {
	first := &weightStorageHostMemory{bytes: make([]byte, 3264), ptr: 0x100000}
	injected := errors.New("injected second allocation failure")
	count := 0
	s, err := newMoEWeightStorage(&cuda.Device{}, MoEWeightShape{3, 2, 16, 17, []int{0, 2}}, func(int) (*moeWeightBuffer, error) {
		count++
		if count == 2 {
			return nil, injected
		}
		return first.buffer(), nil
	})
	if !errors.Is(err, injected) || s == nil || first.frees != 0 {
		t.Fatal("partial allocation abandoned")
	}
	if err := s.Close(); err != nil || first.frees != 1 {
		t.Fatalf("partial cleanup: %v", err)
	}
	s, mem := weightStorageHost(t)
	w, scale := weightStoragePayload(s)
	mem[1].failWrite = injected
	if err := s.Load(2, 1, bytes.NewReader(w), bytes.NewReader(scale)); !errors.Is(err, injected) {
		t.Fatal(err)
	}
	if _, err := s.View(2, 1); err == nil {
		t.Fatal("partial scale failure published ready view")
	}
	if mem[0].writes != 1 || mem[0].frees != 0 {
		t.Fatal("partial payload ownership changed")
	}
	mem[1].failWrite = nil
	if err := s.Load(2, 1, bytes.NewReader(w), bytes.NewReader(scale)); err != nil {
		t.Fatal(err)
	}
	v, err := s.View(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	d, _ := v.Descriptor()
	if d.Generation != 2 {
		t.Fatal("failed generation reused")
	}
	v.Close()
	if err := s.Load(2, 1, nil, bytes.NewReader(scale)); err == nil {
		t.Fatal("nil reader accepted")
	}
	if _, err := s.View(2, 1); err == nil {
		t.Fatal("failed reader left range ready")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMoEWeightViewLifetime(t *testing.T) {
	s, mem := weightStorageHost(t)
	w, scale := weightStoragePayload(s)
	if err := s.Load(0, 0, bytes.NewReader(w), bytes.NewReader(scale)); err != nil {
		t.Fatal(err)
	}
	v, err := s.View(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := v.retain(&cuda.Device{}); err == nil {
		t.Fatal("different context/device admitted")
	}
	_, release, err := v.retain(s.device)
	if err != nil {
		t.Fatal(err)
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := v.Descriptor(); err == nil {
		t.Fatal("closed view remains usable")
	}
	if err := s.Close(); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("consumer lost independent retention")
	}
	if err := s.Load(0, 0, bytes.NewReader(w), bytes.NewReader(scale)); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("retained range rewritten")
	}
	if mem[0].users != 1 || mem[1].users != 1 {
		t.Fatal("consumer does not pin both bases")
	}
	release()
	release()
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMoEWeightSessionAllocationBudget(t *testing.T) {
	s, mem := weightStorageHost(t)
	w, scale := weightStoragePayload(s)
	if err := s.Load(2, 0, bytes.NewReader(w), bytes.NewReader(scale)); err != nil {
		t.Fatal(err)
	}
	before := s.Stats()
	for i := 0; i < 20; i++ {
		v, err := s.View(2, 0)
		if err != nil {
			t.Fatal(err)
		}
		v.Close()
	}
	if s.Stats() != before || before.BaseAllocations != 2 || before.AllocatedBytes != uint64(3264+768) || mem[0].writes != 1 || mem[1].writes != 1 {
		t.Fatal("session restacked or reuploaded weights")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func weightStorageArtifact(t *testing.T, kind MoEWeightConsumerKind) *cudakernel.Artifact {
	t.Helper()
	source, entry, err := MoEWeightSource(kind)
	if err != nil {
		t.Fatal(err)
	}
	schema, err := MoEWeightSchema()
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("compiler image fixture"), entry, schema, "sm_121", cudakernel.WithDependencies([]cudakernel.Dependency{{Path: "cudamoe_weight_storage.cu", Tracked: true, Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(source)))}}), cudakernel.WithCompilerOptions([]string{"--fmad=false"}))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestMoEWeightArtifactBinding(t *testing.T) {
	for _, kind := range []MoEWeightConsumerKind{MoEWeightViewConsumer, MoEWeightSlabConsumer} {
		a := weightStorageArtifact(t, kind)
		owned, err := checkedMoEWeightArtifact(kind, a)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(owned.Raw, a.Raw) || owned.EntryPoint != a.EntryPoint {
			t.Fatal("valid binding changed")
		}
		a.Raw[0] = 'X'
		a.Schema.Params[0].Name = "changed"
		a.Dependencies[0].Digest = "changed"
		a.CompilerOpts[0] = "changed"
		if owned.Raw[0] == 'X' || owned.Schema.Params[0].Name == "changed" || owned.Dependencies[0].Digest == "changed" || owned.CompilerOpts[0] == "changed" {
			t.Fatal("artifact aliases caller slices")
		}
	}
	for _, tt := range []struct {
		name string
		edit func(*cudakernel.Artifact)
	}{
		{"wrong_entry", func(a *cudakernel.Artifact) { a.EntryPoint = "moe_weight_slab" }},
		{"wrong_schema", func(a *cudakernel.Artifact) { a.Schema.Params[0].Name = "different" }},
		{"missing_binding", func(a *cudakernel.Artifact) { a.Dependencies = nil }},
		{"duplicate_binding", func(a *cudakernel.Artifact) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) }},
		{"wrong_digest", func(a *cudakernel.Artifact) { a.Dependencies[0].Digest = "wrong" }},
		{"untracked", func(a *cudakernel.Artifact) { a.Dependencies[0].Tracked = false }},
		{"producer_error", func(a *cudakernel.Artifact) { a.Dependencies[0].ErrorMsg = "read failed" }},
		{"corrupt_image", func(a *cudakernel.Artifact) { a.Raw[0] = 'X' }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := weightStorageArtifact(t, MoEWeightViewConsumer)
			tt.edit(a)
			if _, err := checkedMoEWeightArtifact(MoEWeightViewConsumer, a); err == nil {
				t.Fatal("invalid source/image/schema admitted")
			}
		})
	}
	if _, err := checkedMoEWeightArtifact(MoEWeightViewConsumer, nil); err == nil {
		t.Fatal("nil artifact admitted")
	}
	if _, err := checkedMoEWeightArtifact(0, weightStorageArtifact(t, MoEWeightViewConsumer)); err == nil {
		t.Fatal("unknown consumer admitted")
	}
}

func TestMoEWeightConsumerSchema(t *testing.T) {
	schema, err := MoEWeightSchema()
	if err != nil {
		t.Fatal(err)
	}
	if len(schema.Params) != 9 || schema.TotalParamBytes != 56 {
		t.Fatalf("schema: %+v", schema)
	}
	for i, p := range schema.Params {
		if i < 4 {
			if !p.IsPointer || p.ByteWidth != 8 || p.Alignment != 8 {
				t.Fatal("pointer ABI differs")
			}
		} else if p.IsPointer || p.Type != "s32" || p.ByteWidth != 4 || p.Alignment != 4 {
			t.Fatal("dimension ABI differs")
		}
	}
}
