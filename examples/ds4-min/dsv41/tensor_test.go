package dsv41

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestModelTensor(t *testing.T) {
	g := modelFixture(t)
	r := new(countReader)
	m, err := Bind(r, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range g.Tensors {
		t.Run(want.Name, func(t *testing.T) {
			got, ok := m.Tensor(want.Name)
			n, err := want.Bytes()
			if err != nil {
				t.Fatal(err)
			}
			if !ok || got.Name != want.Name || got.Encoding != want.TypeID ||
				!reflect.DeepEqual(got.Dimensions, want.Dimensions) ||
				got.Offset != int64(g.DataOffset+want.Offset) || got.Bytes != int64(n) {
				t.Fatalf("Tensor(%q) = %+v, %v; directory %+v", want.Name, got, ok, want)
			}
		})
	}
	if r.reads != 0 {
		t.Fatalf("descriptor lookup read payloads: %d reads", r.reads)
	}
	for _, model := range []*Model{nil, {}, m} {
		if got, ok := model.Tensor("not-a-tensor"); ok || !reflect.DeepEqual(got, TensorDescriptor{}) {
			t.Fatalf("missing tensor = %+v, %v", got, ok)
		}
	}
}

func TestModelTensorDetached(t *testing.T) {
	for _, name := range []string{"token_embd.weight", "blk.1.engram_embd.weight", "blk.14.engram_embd.weight"} {
		t.Run(name, func(t *testing.T) {
			for _, mutation := range []string{"lookup-return", "bound-directory"} {
				t.Run(mutation, func(t *testing.T) {
					g := modelFixture(t)
					index := -1
					for i := range g.Tensors {
						if g.Tensors[i].Name == name {
							index = i
							break
						}
					}
					if index < 0 {
						t.Fatal("fixture tensor missing")
					}
					original := g.Tensors[index]
					n, err := original.Bytes()
					if err != nil {
						t.Fatal(err)
					}
					// Capture the expectation before Bind or lookup, detached from
					// both directory and returned descriptor backing arrays.
					want := TensorDescriptor{
						Name: original.Name, Encoding: original.TypeID,
						Dimensions: append([]uint64(nil), original.Dimensions...),
						Offset:     int64(g.DataOffset + original.Offset), Bytes: int64(n),
					}
					m, err := Bind(new(countReader), int64(g.Size), g)
					if err != nil {
						t.Fatal(err)
					}
					got, ok := m.Tensor(name)
					if !ok || !reflect.DeepEqual(got, want) {
						t.Fatalf("initial descriptor: got %+v, want %+v", got, want)
					}
					switch mutation {
					case "lookup-return":
						got.Dimensions[0]++
						got.Name, got.Encoding, got.Offset, got.Bytes = "changed", 99, -1, -1
					case "bound-directory":
						g.Tensors[index].Dimensions[0]++
						g.Tensors[index].Name = "changed"
						g.Tensors[index].TypeID = 99
						g.Tensors[index].Offset++
						g.DataOffset++
					}
					again, ok := m.Tensor(name)
					if !ok || !reflect.DeepEqual(again, want) {
						t.Fatalf("%s changed descriptor: got %+v, want %+v", mutation, again, want)
					}
				})
			}
		})
	}
}

func TestModelTensorInvalidBind(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*gguf.File)
	}{
		{"offset overflow", func(g *gguf.File) { g.Tensors[0].Offset = math.MaxUint64 }},
		{"signed offset overflow", func(g *gguf.File) { g.Tensors[0].Offset = math.MaxInt64 }},
		{"outside file", func(g *gguf.File) { g.Tensors[0].Offset = g.Size }},
		{"dimension", func(g *gguf.File) { g.Tensors[0].Dimensions[0]++ }},
		{"encoding", func(g *gguf.File) { g.Tensors[0].TypeID = 0 }},
		{"missing", func(g *gguf.File) { g.Tensors = g.Tensors[1:] }},
		{"gate up mismatch", func(g *gguf.File) {
			for i := range g.Tensors {
				if g.Tensors[i].Name == "blk.0.ffn_up_exps.weight" {
					g.Tensors[i].TypeID = 8
				}
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			g := modelFixture(t)
			test.mutate(g)
			m, err := Bind(new(countReader), int64(g.Size), g)
			if err == nil || m != nil {
				t.Fatalf("invalid bind returned model %v, error %v", m != nil, err)
			}
			if _, ok := m.Tensor("token_embd.weight"); ok {
				t.Fatal("failed binding published descriptors")
			}
		})
	}
}

func ExampleModel_Tensor() {
	var model *Model
	tensor, ok := model.Tensor("token_embd.weight")
	fmt.Println(tensor.Name, ok)
	// Output: false
}

func ExampleTensorDescriptor() {
	// A descriptor uses GGUF dimension order and absolute file offsets.
	t := TensorDescriptor{Name: "norm.weight", Encoding: 0, Dimensions: []uint64{5120}, Offset: 4096, Bytes: 20480}
	fmt.Println(t.Name, t.Offset, t.Bytes)
	// Output: norm.weight 4096 20480
}
