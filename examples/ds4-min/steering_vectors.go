package main

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// The pinned Flash shape has 43 normal layers; its optional MTP layer is not
// in this file. Directions are raw little-endian float32, one width per layer.
type steeringVectors struct {
	values        []float32
	layers, width int
	digest        [32]byte
}

func loadSteeringVectors(path string) (vectors *steeringVectors, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open steering vectors: %w", err)
	}
	defer func() { err = errors.Join(err, f.Close()) }()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() != int64(Flash.Layers*Flash.Embed*4) {
		return nil, errors.New("steering vectors have wrong byte extent")
	}
	return readSteeringVectors(f, Flash.Layers, Flash.Embed)
}

func readSteeringVectors(r io.Reader, layers, width int) (*steeringVectors, error) {
	if r == nil || layers <= 0 || width <= 0 || layers > math.MaxInt/width/4 {
		return nil, errors.New("invalid steering vector shape")
	}
	data := make([]byte, layers*width*4)
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, fmt.Errorf("read steering vectors: %w", err)
	}
	var extra [1]byte
	if n, err := io.ReadFull(r, extra[:]); n != 0 || err != io.EOF {
		return nil, errors.New("steering vectors have trailing bytes")
	}
	v := &steeringVectors{values: make([]float32, layers*width), layers: layers, width: width, digest: sha256.Sum256(data)}
	for i := range v.values {
		v.values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		if math.IsNaN(float64(v.values[i])) || math.IsInf(float64(v.values[i]), 0) {
			return nil, errors.New("steering vectors contain nonfinite values")
		}
	}
	return v, nil
}

func steeringScale(scale float32) error {
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || scale < -100 || scale > 100 {
		return errors.New("steering scale must be between -100 and 100")
	}
	return nil
}

// steeringProject is the independent scalar CPU reference from the pinned
// source. Device kernels use a parallel reduction and have a separate gate.
func steeringProject(x, direction []float32, scale float32) error {
	if len(x) == 0 || len(x) != len(direction) {
		return errors.New("invalid steering projection shape")
	}
	if err := steeringScale(scale); err != nil {
		return err
	}
	if scale == 0 {
		return nil
	}
	var dot float32
	for i := range x {
		dot += x[i] * direction[i]
	}
	coefficient := scale * dot
	for i := range x {
		x[i] -= coefficient * direction[i]
	}
	return nil
}
