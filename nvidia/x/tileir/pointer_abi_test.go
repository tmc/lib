package tileir_test

import (
	"encoding/binary"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// The generic and Tile C++ routes emit bare float* parameters, which the
// C ABI places in consecutive 8-byte slots. The schema must agree, or Pack
// writes every argument after the first where the kernel does not read it.
func TestSemanticPointerSlots(t *testing.T) {
	graphs := []struct {
		name  string
		build func(g *cudair.Graph)
	}{
		{"add2d", func(g *cudair.Graph) {
			g.AddBinary("add", g.AddInput("A", []int{4, 4}, cudair.Float32), g.AddInput("B", []int{4, 4}, cudair.Float32))
		}},
		{"matmul", func(g *cudair.Graph) {
			g.AddMatMul(g.AddInput("A", []int{4, 4}, cudair.Float32), g.AddInput("B", []int{4, 4}, cudair.Float32))
		}},
	}
	routes := []struct {
		name   string
		schema func(*cudair.Snapshot) (cudakernel.Schema, error)
	}{
		{"LowerSemanticRegion", func(s *cudair.Snapshot) (cudakernel.Schema, error) {
			r, err := tileir.LowerSemanticRegion(s, tileir.LowerOptions{})
			if err != nil {
				return cudakernel.Schema{}, err
			}
			return r.Schema, nil
		}},
		{"GenerateSemanticTileCPP", func(s *cudair.Snapshot) (cudakernel.Schema, error) {
			_, schema, err := tileir.GenerateSemanticTileCPP(s, tileir.LowerOptions{})
			return schema, err
		}},
	}
	for _, gr := range graphs {
		for _, rt := range routes {
			t.Run(gr.name+"/"+rt.name, func(t *testing.T) {
				g := cudair.NewGraph()
				gr.build(g)
				snap, err := g.Snapshot()
				if err != nil {
					t.Fatal(err)
				}
				s, err := rt.schema(snap)
				if err != nil {
					t.Fatal(err)
				}
				n := len(s.Params)
				if n != 3 || s.TotalParamBytes != 8*n {
					t.Fatalf("%d parameters, %d bytes, want 3 and 24", n, s.TotalParamBytes)
				}
				args := make([]any, n)
				for i, p := range s.Params {
					if !p.IsPointer || p.ByteWidth != 8 || p.Alignment != 8 || p.Offset != 8*i {
						t.Fatalf("parameter %d = %+v, want an 8-byte pointer at offset %d", i, p, 8*i)
					}
					args[i] = uint64(0x1000 + i)
				}
				buf, err := s.Pack(args...)
				if err != nil {
					t.Fatal(err)
				}
				if len(buf) != 8*n {
					t.Fatalf("Pack returned %d bytes, want %d", len(buf), 8*n)
				}
				for i := range n {
					if got := binary.LittleEndian.Uint64(buf[8*i:]); got != uint64(0x1000+i) {
						t.Errorf("slot %d = %#x, want %#x", i, got, 0x1000+i)
					}
				}
			})
		}
	}
}
