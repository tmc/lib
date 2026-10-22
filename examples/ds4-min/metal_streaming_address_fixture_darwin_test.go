//go:build darwin && arm64

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

type ssdAddressFixture struct {
	cols, rows int
	packed     [3][]byte
	input      []float32
	downInput  []float32
	ids        []int32
	want       [3][]float64
	adjacent   []float64
}

func newSSDAddressFixture(cols, rows int) (ssdAddressFixture, error) {
	f := ssdAddressFixture{cols: cols, rows: rows, ids: []int32{5, 1, 5, 3, 0, 1}}
	if cols < 256 || cols > 2048 || cols%256 != 0 || rows < 1 || rows > 9 {
		return f, fmt.Errorf("invalid SSD address fixture extent")
	}
	for component := range f.packed {
		blockBytes := 66
		if component == 2 {
			blockBytes = 84
		}
		f.packed[component] = make([]byte, 8*rows*(cols/256)*blockBytes)
		for expert := range 8 {
			for row := range rows {
				for block := range cols / 256 {
					p := f.packed[component][((expert*rows+row)*(cols/256)+block)*blockBytes:]
					for i := range blockBytes {
						p[i] = byte(17 + component*41 + expert*19 + row*11 + block*29 + i*23)
					}
					if component < 2 {
						binary.LittleEndian.PutUint16(p, 0x2400) // 2^-6
					} else {
						binary.LittleEndian.PutUint16(p[80:], 0x2400)
						binary.LittleEndian.PutUint16(p[82:], 0x2000) // 2^-7
					}
				}
			}
		}
	}
	f.input = make([]float32, cols)
	f.downInput = make([]float32, len(f.ids)*cols)
	for i := range f.input {
		f.input[i] = float32(i%5 - 2)
		for slot := range f.ids {
			f.downInput[slot*cols+i] = float32((i+slot*3)%7 - 3)
		}
	}
	for component := range f.want {
		var err error
		f.want[component], err = f.reference(component, f.ids)
		if err != nil {
			return f, err
		}
	}
	var err error
	f.adjacent, err = f.reference(0, []int32{2, 2, 2, 2, 2, 2})
	return f, err
}

// The independent CPU decoder uses separately pinned C codebooks. Dot products
// are accumulated in float64, without Metal code or its reduction schedule.
// All fixture operands are dyadic and the sum of absolute integer numerators
// is below 2^24, so every float32 multiplication/addition is exact in any order.
// The gate therefore has a fixed zero-error bound; it is an address/numeric
// fixture, not a qualification of ill-conditioned model reductions.
func (f ssdAddressFixture) reference(component int, ids []int32) ([]float64, error) {
	typeID := dsv4cpu.IQ2XXS
	quantum := 512.0
	if component == 2 {
		typeID = dsv4cpu.Q2K
		quantum = 128
	}
	packed := f.packed[component]
	m, err := dsv4cpu.NewMatrix(bytes.NewReader(packed), 0, int64(len(packed)), 8*f.rows, f.cols, typeID)
	if err != nil {
		return nil, err
	}
	row := make([]float32, f.cols)
	out := make([]float64, len(ids)*f.rows)
	for slot, id := range ids {
		x := f.input
		if component == 2 {
			x = f.downInput[slot*f.cols : (slot+1)*f.cols]
		}
		for r := range f.rows {
			if err := m.Row(row, int(id)*f.rows+r); err != nil {
				return nil, err
			}
			var sum, absolute float64
			for i, weight := range row {
				term := float64(weight) * float64(x[i])
				if term*quantum != math.Trunc(term*quantum) {
					return nil, fmt.Errorf("fixture term is not dyadic")
				}
				sum += term
				absolute += math.Abs(term * quantum)
			}
			if absolute >= 1<<24 || float64(float32(sum)) != sum {
				return nil, fmt.Errorf("fixture exceeds exact arithmetic bound")
			}
			out[slot*f.rows+r] = sum
		}
	}
	return out, nil
}

func (f ssdAddressFixture) source() (*dsv4expertcache.Source, *dsv4expertcache.Cache, [3]uint64, error) {
	var raw []byte
	var tensors [3]dsv4expertcache.Tensor
	var strides [3]uint64
	for i, data := range f.packed {
		strides[i] = uint64(len(data) / 8)
		tensors[i] = dsv4expertcache.Tensor{Offset: uint64(len(raw)), Bytes: uint64(len(data)), Stride: strides[i], Experts: 8}
		raw = append(raw, data...)
	}
	id := dsv4expertcache.Identity{Model: sha256.Sum256(raw), Quant: sha256.Sum256([]byte("IQ2_XXS/IQ2_XXS/Q2_K")), Layout: sha256.Sum256([]byte(fmt.Sprint(f.cols, f.rows)))}
	source, err := dsv4expertcache.NewSource(bytes.NewReader(raw), uint64(len(raw)), id, map[int]dsv4expertcache.Triplet{0: {Gate: tensors[0], Up: tensors[1], Down: tensors[2]}})
	if err != nil {
		return nil, nil, strides, err
	}
	cache, err := dsv4expertcache.New(id, 5, strides[0]+strides[1]+strides[2], 5*16384)
	return source, cache, strides, err
}

func TestSSDAddressFixture(t *testing.T) {
	for _, shape := range [][2]int{{256, 3}, {1024, 7}, {2048, 9}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			f, err := newSSDAddressFixture(shape[0], shape[1])
			if err != nil {
				t.Fatal(err)
			}
			if _, _, _, err := f.source(); err != nil {
				t.Fatal(err)
			}
			for component := range f.want {
				if len(f.want[component]) != 6*f.rows {
					t.Fatal("incomplete reference")
				}
			}
			for slot := range f.ids {
				different := false
				for row := range f.rows {
					different = different || f.want[0][slot*f.rows+row] != f.adjacent[slot*f.rows+row]
				}
				if !different {
					t.Fatalf("adjacent expert mutant invisible in slot %d", slot)
				}
			}
			if dir := os.Getenv("DS4_SSD_ADDRESS_FIXTURE_OUTPUT"); dir != "" {
				result := ssdAddressResult{Cols: f.cols, Rows: f.rows, IDs: f.ids, Input: f.input, DownInput: f.downInput, Packed: f.packed, Reference: f.want, Adjacent: f.adjacent, Canaries: true}
				for component := range result.Output {
					for _, v := range f.want[component] {
						result.Output[component] = append(result.Output[component], float32(v))
					}
				}
				encoded, err := json.Marshal(result)
				if err != nil {
					t.Fatal(err)
				}
				path := filepath.Join(dir, fmt.Sprintf("%dx%d.json", f.cols, f.rows))
				file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
				if err != nil {
					t.Fatal(err)
				}
				_, err = file.Write(encoded)
				if err := errors.Join(err, file.Close()); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
