//go:build ds4cpumodel

package dsv4cpu

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"math"
	"os"

	"github.com/tmc/lib/gguf"
)

// openFlashBackend is the actual immutable model binding, kept private until
// the complete forward gates pass. It never delegates computation to C.
func openFlashBackend(path, expectedSHA256 string) (_ *flashBackend, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			f.Close()
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := info.Size()
	hash := sha256.New()
	if _, err := io.Copy(hash, io.NewSectionReader(f, 0, size)); err != nil {
		return nil, err
	}
	digest := hex.EncodeToString(hash.Sum(nil))
	if expectedSHA256 != "" && digest != expectedSHA256 {
		return nil, errors.New("CPU model digest differs")
	}
	g, err := gguf.Read(f, size)
	if err != nil {
		return nil, err
	}
	tensors := make([]Tensor, len(g.Tensors))
	for i, v := range g.Tensors {
		bytes, err := v.Bytes()
		if err != nil {
			return nil, err
		}
		if g.DataOffset > math.MaxInt64 || v.Offset > math.MaxInt64-g.DataOffset || bytes > math.MaxInt64 {
			return nil, errors.New("CPU tensor extent overflows")
		}
		dims := make([]int, len(v.Dimensions))
		for j, d := range v.Dimensions {
			if d > uint64(^uint(0)>>1) {
				return nil, errors.New("CPU tensor dimension overflows")
			}
			dims[j] = int(d)
		}
		tensors[i] = Tensor{Name: v.Name, Dimensions: dims, Encoding: Encoding(v.TypeID), Offset: int64(g.DataOffset + v.Offset), Bytes: int64(bytes)}
	}
	m, err := newFlashModel(f, size, tensors)
	if err != nil {
		return nil, err
	}
	copy(m.digest[:], hash.Sum(nil))
	return &flashBackend{model: m, source: f}, nil
}
