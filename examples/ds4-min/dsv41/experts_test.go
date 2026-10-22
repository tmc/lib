package dsv41

import (
	"crypto/sha256"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
	"github.com/tmc/lib/gguf"
)

// Pattern bytes depend on the absolute offset, including high offset bits.
// No enormous fixture file or complete-file hashing is implied by this reader.
type expertPatternReader struct{}

func (expertPatternReader) ReadAt(dst []byte, off int64) (int, error) {
	for i := range dst {
		dst[i] = expertPattern(uint64(off) + uint64(i))
	}
	return len(dst), nil
}

func expertPattern(offset uint64) byte {
	x := offset ^ (offset >> 17) ^ (offset >> 29)
	return byte((x*73+41)%251 + 1)
}

func expertFixture(t *testing.T) *Model {
	t.Helper()
	g := modelFixture(t)
	m, err := Bind(expertPatternReader{}, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	// Test-only synthetic identity; production requires BindVerified to read and
	// hash the whole bound file. A sparse fixture is not a released model asset.
	m.assetSHA = sha256.Sum256([]byte("synthetic expert directory control"))
	return m
}

func TestV41ExpertTripletsRead(t *testing.T) {
	m := expertFixture(t)
	layers, identity, err := m.ExpertTriplets()
	if err != nil {
		t.Fatal(err)
	}
	if len(layers) != 40 || identity.Model != m.AssetSHA256() || identity.Quant == ([32]byte{}) || identity.Layout == ([32]byte{}) {
		t.Fatalf("incomplete directory or identity: %d %+v", len(layers), identity)
	}
	source, err := dsv4expertcache.NewSource(expertPatternReader{}, m.fileSize, identity, layers)
	if err != nil {
		t.Fatal(err)
	}
	for component, suffix := range []string{"gate", "up", "down"} {
		t.Run(suffix, func(t *testing.T) {
			name := "blk.0.ffn_" + suffix + "_exps.weight"
			descriptor, _ := m.Tensor(name)
			stride := uint64(descriptor.Bytes) / descriptor.Dimensions[2]
			buf := make([]byte, stride)
			key := dsv4expertcache.Key{Identity: identity, Layer: 0, Expert: 383}
			if err := source.Read(key, component, buf); err != nil {
				t.Fatal(err)
			}
			start := uint64(descriptor.Offset) + 383*stride
			for i, got := range buf {
				if want := expertPattern(start + uint64(i)); got != want {
					t.Fatalf("component %d byte %d = %d, want %d", component, i, got, want)
				}
			}
			key.Expert = 384
			if source.Read(key, component, buf) == nil {
				t.Fatal("expert 384 admitted")
			}
			key.Expert = 383
			for _, field := range []string{"model", "quant", "layout"} {
				wrong := key
				switch field {
				case "model":
					wrong.Identity.Model[0]++
				case "quant":
					wrong.Identity.Quant[0]++
				case "layout":
					wrong.Identity.Layout[0]++
				}
				if source.Read(wrong, component, buf) == nil {
					t.Fatalf("wrong %s identity admitted", field)
				}
			}
		})
	}
	// Neither a caller-owned map mutation nor deletion changes a subsequent lookup.
	want := layers[0]
	layers[0] = dsv4expertcache.Triplet{}
	delete(layers, 1)
	again, againID, err := m.ExpertTriplets()
	if err != nil || again[0] != want || len(again) != 40 || againID != identity {
		t.Fatalf("caller mutation changed directory: %v", err)
	}
}

func TestV41ExpertTripletsEveryLayer(t *testing.T) {
	m := expertFixture(t)
	layers, _, err := m.ExpertTriplets()
	if err != nil {
		t.Fatal(err)
	}
	for layer := 0; layer < 40; layer++ {
		t.Run(fmt.Sprintf("layer-%02d", layer), func(t *testing.T) {
			tr, ok := layers[layer]
			if !ok {
				t.Fatal("layer missing")
			}
			for component, suffix := range []string{"gate", "up", "down"} {
				name := fmt.Sprintf("blk.%d.ffn_%s_exps.weight", layer, suffix)
				descriptor, ok := m.Tensor(name)
				if !ok || len(descriptor.Dimensions) != 3 {
					t.Fatalf("missing independently named descriptor %q", name)
				}
				want := dsv4expertcache.Tensor{
					Offset:  uint64(descriptor.Offset),
					Bytes:   uint64(descriptor.Bytes),
					Stride:  uint64(descriptor.Bytes) / descriptor.Dimensions[2],
					Experts: int(descriptor.Dimensions[2]),
				}
				got := []dsv4expertcache.Tensor{tr.Gate, tr.Up, tr.Down}[component]
				if got != want {
					t.Fatalf("%s extent = %+v, want %+v", name, got, want)
				}
			}
		})
	}
}

func TestV41ExpertTripletsReject(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*Model)
	}{
		{"unverified", func(m *Model) { m.assetSHA = [32]byte{} }},
		{"missing", func(m *Model) { delete(m.tensors, "blk.0.ffn_gate_exps.weight") }},
		{"wrong family axis", func(m *Model) {
			d := m.tensors["blk.0.ffn_gate_exps.weight"]
			d.Dimensions[2] = 256
			m.tensors[d.Name] = d
		}},
		{"encoding", func(m *Model) { d := m.tensors["blk.0.ffn_gate_exps.weight"]; d.Encoding = 1; m.tensors[d.Name] = d }},
		{"format confusion", func(m *Model) {
			// Keep gate/up encodings equal and the computed Q8_0 extents in
			// bounds, but retain their original IQ2_XXS byte counts. Only the
			// serialized-byte equality guard should reject these descriptors.
			m.fileSize = math.MaxInt64
			for _, suffix := range []string{"gate", "up"} {
				d := m.tensors["blk.0.ffn_"+suffix+"_exps.weight"]
				d.Encoding = 8
				m.tensors[d.Name] = d
			}
		}},
		{"gate up mismatch", func(m *Model) {
			d := m.tensors["blk.0.ffn_up_exps.weight"]
			d.Encoding = 8
			// Encoding 8 has 34 bytes per 32 elements, not IQ2_XXS's 66/256.
			d.Bytes = 5120 * 2304 * 384 / 32 * 34
			m.tensors[d.Name] = d
		}},
		{"negative offset", func(m *Model) { d := m.tensors["blk.0.ffn_gate_exps.weight"]; d.Offset = -1; m.tensors[d.Name] = d }},
		{"outside", func(m *Model) {
			d := m.tensors["blk.0.ffn_gate_exps.weight"]
			d.Offset = int64(m.fileSize)
			m.tensors[d.Name] = d
		}},
		{"offset overflow", func(m *Model) {
			d := m.tensors["blk.0.ffn_gate_exps.weight"]
			d.Offset = math.MaxInt64
			m.tensors[d.Name] = d
		}},
		{"wrong bytes", func(m *Model) { d := m.tensors["blk.0.ffn_gate_exps.weight"]; d.Bytes++; m.tensors[d.Name] = d }},
	} {
		t.Run(test.name, func(t *testing.T) {
			m := expertFixture(t)
			test.change(m)
			layers, id, err := m.ExpertTriplets()
			if err == nil || layers != nil || !reflect.DeepEqual(id, dsv4expertcache.Identity{}) {
				t.Fatalf("invalid layout published: layers=%d identity=%+v err=%v", len(layers), id, err)
			}
		})
	}
}

func TestV41ExpertIdentitySeparation(t *testing.T) {
	m := expertFixture(t)
	_, before, err := m.ExpertTriplets()
	if err != nil {
		t.Fatal(err)
	}
	m.assetSHA[0]++
	_, after, err := m.ExpertTriplets()
	if err != nil || after.Model == before.Model || after.Quant != before.Quant || after.Layout != before.Layout {
		t.Fatalf("model identity separation failed: %v", err)
	}
	m.assetSHA = before.Model
	name := "blk.0.ffn_down_exps.weight"
	d := m.tensors[name]
	d.Encoding = 8
	n, err := (gguf.Tensor{TypeID: d.Encoding, Dimensions: d.Dimensions}).Bytes()
	if err != nil {
		t.Fatal(err)
	}
	d.Bytes = int64(n)
	m.tensors[name] = d
	_, after, err = m.ExpertTriplets()
	if err != nil || after.Model != before.Model || after.Quant == before.Quant || after.Layout == before.Layout {
		t.Fatalf("quantized layout identity separation failed: %v", err)
	}
	d.Offset += 32
	m.tensors[name] = d
	_, shifted, err := m.ExpertTriplets()
	if err != nil || shifted.Model != after.Model || shifted.Quant != after.Quant || shifted.Layout == after.Layout {
		t.Fatalf("offset layout identity separation failed: %v", err)
	}
}

func TestV41ExpertSerializedEncodings(t *testing.T) {
	for _, encoding := range []uint32{8, 16, 10, 12, 13, 14, 39} {
		t.Run(fmt.Sprint(encoding), func(t *testing.T) {
			m := expertFixture(t)
			for _, suffix := range []string{"gate", "up", "down"} {
				name := "blk.0.ffn_" + suffix + "_exps.weight"
				d := m.tensors[name]
				d.Encoding = encoding
				n, err := (gguf.Tensor{TypeID: encoding, Dimensions: d.Dimensions}).Bytes()
				if err != nil {
					t.Fatal(err)
				}
				d.Bytes = int64(n)
				m.tensors[name] = d
			}
			layers, identity, err := m.ExpertTriplets()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := dsv4expertcache.NewSource(expertPatternReader{}, m.fileSize, identity, layers); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type expertDone struct{}

func (expertDone) Wait() error { return nil }

type expertReadTransfer struct {
	t       *testing.T
	triplet dsv4expertcache.Triplet
	calls   int
}

func (transfer *expertReadTransfer) Submit(upload dsv4expertcache.Upload, component int, data []byte) (dsv4expertcache.Completion, error) {
	transfer.t.Helper()
	part := []dsv4expertcache.Tensor{transfer.triplet.Gate, transfer.triplet.Up, transfer.triplet.Down}[component]
	start := part.Offset + uint64(upload.Key.Expert)*part.Stride
	for i, got := range data {
		if want := expertPattern(start + uint64(i)); got != want {
			transfer.t.Fatalf("staged component %d byte %d = %d, want %d", component, i, got, want)
		}
	}
	transfer.calls++
	return expertDone{}, nil
}

func TestV41ExpertLeaseRead(t *testing.T) {
	m := expertFixture(t)
	layers, identity, err := m.ExpertTriplets()
	if err != nil {
		t.Fatal(err)
	}
	source, err := dsv4expertcache.NewSource(expertPatternReader{}, m.fileSize, identity, layers)
	if err != nil {
		t.Fatal(err)
	}
	tr := layers[0]
	bytes := tr.Gate.Stride + tr.Up.Stride + tr.Down.Stride
	cache, err := dsv4expertcache.New(identity, 1, bytes, bytes)
	if err != nil {
		t.Fatal(err)
	}
	key := dsv4expertcache.Key{Identity: identity, Layer: 0, Expert: 383}
	lease, err := cache.Acquire([]dsv4expertcache.Key{key})
	if err != nil {
		t.Fatal(err)
	}
	transfer := &expertReadTransfer{t: t, triplet: tr}
	stageBytes := max(tr.Gate.Stride, tr.Up.Stride, tr.Down.Stride)
	if err := source.Stage(lease, make([]byte, stageBytes), transfer, false); err != nil {
		t.Fatal(err)
	}
	if transfer.calls != 3 {
		t.Fatalf("staged %d components, want 3", transfer.calls)
	}
	if err := lease.Release(expertDone{}); err != nil {
		t.Fatal(err)
	}
	hit, err := cache.Acquire([]dsv4expertcache.Key{key})
	if err != nil {
		t.Fatal(err)
	}
	if len(hit.Uploads()) != 0 {
		t.Fatal("qualified expert was not cached")
	}
	if err := hit.Abort(); err != nil {
		t.Fatal(err)
	}
}

func ExampleModel_ExpertTriplets() {
	var model *Model
	_, _, err := model.ExpertTriplets()
	fmt.Println(err)
	// Output: unverified V4.1 expert source
}
