package dsv4cpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

type flashSnapshot struct{ state *flashState }

func (s *flashState) clone() (*flashState, error) {
	if s == nil || s.model == nil || s.closed || s.failed {
		return nil, errors.New("invalid Flash state for snapshot")
	}
	out, err := s.model.newState(s.context)
	if err != nil {
		return nil, err
	}
	out.position = s.position
	if s.support != nil {
		out.support, err = s.support.clone(s.position)
		if err != nil {
			out.close()
			return nil, err
		}
	}
	out.steering = append([]float32(nil), s.steering...)
	out.attentionScale, out.ffnScale = s.attentionScale, s.ffnScale
	copy(out.logits, s.logits)
	for il := range s.layers {
		src, dst := &s.layers[il], &out.layers[il]
		dst.raw.count = src.raw.count
		copy(dst.raw.rows, src.raw.rows)
		for _, pair := range [][2]*Compressor{{src.attention, dst.attention}, {src.index, dst.index}} {
			if pair[0] != nil {
				pair[1].position = pair[0].position
				copy(pair[1].kv, pair[0].kv)
				copy(pair[1].score, pair[0].score)
			}
		}
		dst.compressed = append(dst.compressed, src.compressed...)
		dst.indexRows = append(dst.indexRows, src.indexRows...)
	}
	return out, nil
}

func (s *flashState) snapshot() (*flashSnapshot, error) {
	copy, err := s.clone()
	if err != nil {
		return nil, err
	}
	return &flashSnapshot{copy}, nil
}

func (s *flashState) restore(snapshot *flashSnapshot) error {
	if s == nil || s.closed || snapshot == nil || snapshot.state == nil || snapshot.state.model != s.model || snapshot.state.context != s.context {
		return errors.New("foreign or closed Flash snapshot")
	}
	copy, err := snapshot.state.clone()
	if err != nil {
		return err
	}
	if (s.support == nil) != (copy.support == nil) || (s.support != nil && !s.support.sameIdentity(copy.support)) {
		copy.close()
		return errors.New("foreign CPU support snapshot")
	}
	old := *s
	*s = *copy
	*copy = old
	copy.close()
	return nil
}

func (s *flashState) close() {
	if s != nil && !s.closed {
		if s.support != nil {
			s.support.close()
			s.support = nil
		}
		clear(s.logits)
		clear(s.steering)
		s.steering = nil
		s.attentionScale, s.ffnScale = 0, 0
		s.logits = nil
		for i := range s.layers {
			s.layers[i].raw.Reset()
			if c := s.layers[i].attention; c != nil {
				clear(c.kv)
				clear(c.score)
			}
			if c := s.layers[i].index; c != nil {
				clear(c.kv)
				clear(c.score)
			}
			clear(s.layers[i].compressed)
			clear(s.layers[i].indexRows)
			s.layers[i] = flashLayerState{}
		}
		s.closed = true
	}
}

func (s *flashSnapshot) close() {
	if s != nil && s.state != nil {
		s.state.close()
		s.state = nil
	}
}

func (s *flashState) save(dst io.Writer) error {
	if s != nil && s.support != nil {
		if dst == nil || s.closed || s.failed || s.model == nil {
			return errors.New("invalid Flash state for save")
		}
		return s.saveSupport(dst)
	}
	return s.saveCore(dst)
}

func (s *flashState) saveCore(dst io.Writer) error {
	if dst == nil || s == nil || s.closed || s.failed || s.model == nil || s.model.digest == ([32]byte{}) {
		return errors.New("invalid Flash state for save")
	}
	hash := sha256.New()
	w := io.MultiWriter(dst, hash)
	write := func(v any) error { return binary.Write(w, binary.LittleEndian, v) }
	if err := write([8]byte{'D', 'S', '4', 'C', 'P', 'U', '0', '2'}); err != nil {
		return err
	}
	if err := write(s.model.digest); err != nil {
		return err
	}
	if err := write(uint32(s.context)); err != nil {
		return err
	}
	if err := write(uint32(s.position)); err != nil {
		return err
	}
	for il := range s.layers {
		l := &s.layers[il]
		if err := write(uint32(l.raw.count)); err != nil {
			return err
		}
		if err := write(l.raw.rows[:l.raw.count*512]); err != nil {
			return err
		}
		if l.attention != nil {
			if err := write(uint32(len(l.compressed) / 512)); err != nil {
				return err
			}
			if err := write(l.compressed); err != nil {
				return err
			}
			if err := write(l.attention.kv); err != nil {
				return err
			}
			if err := write(l.attention.score); err != nil {
				return err
			}
			if l.index != nil {
				if err := write(l.indexRows); err != nil {
					return err
				}
				if err := write(l.index.kv); err != nil {
					return err
				}
				if err := write(l.index.score); err != nil {
					return err
				}
			}
		}
	}
	if err := write(s.logits); err != nil {
		return err
	}
	if err := write(uint32(len(s.steering))); err != nil {
		return err
	}
	if err := write(s.attentionScale); err != nil {
		return err
	}
	if err := write(s.ffnScale); err != nil {
		return err
	}
	if err := write(s.steering); err != nil {
		return err
	}
	_, err := io.Copy(dst, bytes.NewReader(hash.Sum(nil)))
	return err
}

// load consumes one checksummed native frame. It leaves any trailing bytes in
// src for the caller; it does not validate an entire file or upstream KVC.
func (s *flashState) load(src io.Reader) error {
	if s != nil && s.support != nil {
		if src == nil || s.closed || s.model == nil {
			return errors.New("invalid Flash state for load")
		}
		return s.loadSupport(src)
	}
	return s.loadCore(src)
}

func (s *flashState) loadCore(src io.Reader) error {
	if src == nil || s == nil || s.closed || s.model == nil || s.model.digest == ([32]byte{}) {
		return errors.New("invalid Flash state for load")
	}
	hash := sha256.New()
	r := io.TeeReader(src, hash)
	read := func(v any) error { return binary.Read(r, binary.LittleEndian, v) }
	var magic [8]byte
	var digest [32]byte
	var context, pos uint32
	if err := read(&magic); err != nil {
		return err
	}
	if magic != ([8]byte{'D', 'S', '4', 'C', 'P', 'U', '0', '2'}) {
		return errors.New("unsupported CPU state format")
	}
	if err := read(&digest); err != nil {
		return err
	}
	if digest != s.model.digest {
		return errors.New("CPU state model digest differs")
	}
	if err := read(&context); err != nil {
		return err
	}
	if err := read(&pos); err != nil {
		return err
	}
	if int(context) != s.context || pos > context {
		return errors.New("CPU state context or position differs")
	}
	next, err := s.model.newState(s.context)
	if err != nil {
		return err
	}
	next.position = int(pos)
	defer func() {
		if next != nil {
			next.close()
		}
	}()
	readFloats := func(values []float32) error {
		if err := read(values); err != nil {
			return err
		}
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return errors.New("nonfinite CPU state payload")
			}
		}
		return nil
	}
	for il := range next.layers {
		l := &next.layers[il]
		var count uint32
		if err := read(&count); err != nil {
			return fmt.Errorf("layer %d raw count: %w", il, err)
		}
		want := int(pos)
		if want > l.raw.capacity {
			want = l.raw.capacity
		}
		if int(count) != want {
			return errors.New("CPU raw cache count differs from position")
		}
		l.raw.count = int(count)
		if err := readFloats(l.raw.rows[:l.raw.count*512]); err != nil {
			return err
		}
		if l.attention != nil {
			if err := read(&count); err != nil {
				return err
			}
			if int(count) != int(pos)/l.attention.ratio {
				return errors.New("CPU compressed count differs from position")
			}
			l.compressed = l.compressed[:int(count)*512]
			if err := readFloats(l.compressed); err != nil {
				return err
			}
			if err := readFloats(l.attention.kv); err != nil {
				return err
			}
			if err := readFloats(l.attention.score); err != nil {
				return err
			}
			l.attention.position = int(pos)
			if l.index != nil {
				l.indexRows = l.indexRows[:int(count)*128]
				if err := readFloats(l.indexRows); err != nil {
					return err
				}
				if err := readFloats(l.index.kv); err != nil {
					return err
				}
				if err := readFloats(l.index.score); err != nil {
					return err
				}
				l.index.position = int(pos)
			}
		}
	}
	if err := readFloats(next.logits); err != nil {
		return err
	}
	var steeringCount uint32
	if err := read(&steeringCount); err != nil {
		return err
	}
	if steeringCount != 0 && steeringCount != 43*4096 {
		return errors.New("invalid CPU steering size")
	}
	scales := make([]float32, 2)
	if err := readFloats(scales); err != nil {
		return err
	}
	next.attentionScale, next.ffnScale = scales[0], scales[1]
	next.steering = make([]float32, int(steeringCount))
	if err := readFloats(next.steering); err != nil {
		return err
	}
	if steeringCount == 0 && (scales[0] != 0 || scales[1] != 0) {
		return errors.New("CPU steering scale without directions")
	}
	var checksum [32]byte
	if _, err := io.ReadFull(src, checksum[:]); err != nil {
		return err
	}
	if !bytes.Equal(checksum[:], hash.Sum(nil)) {
		return errors.New("CPU state checksum differs")
	}
	*s = *next
	next = nil
	return nil
}
