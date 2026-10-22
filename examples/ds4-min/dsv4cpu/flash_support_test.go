package dsv4cpu

import (
	"bytes"
	"encoding/binary"
	"math"
	"reflect"
	"strings"
	"testing"
)

func supportFixture(t *testing.T, position int) *flashState {
	t.Helper()
	s := codecFixturePosition(t, position)
	s.position = 0
	if err := s.attachSupport([]int{1, 3}, 4); err != nil {
		t.Fatal(err)
	}
	hc := make([]float32, 4*4096)
	for p := 0; p < position; p++ {
		for i := range hc {
			hc[i] = float32((p+i)%31-15) / 8
		}
		for _, layer := range s.support.state.Targets {
			if err := s.support.capture(layer, hc); err != nil {
				t.Fatal(err)
			}
		}
		if err := s.support.publish(p, p+1); err != nil {
			t.Fatal(err)
		}
	}
	s.position = position
	return s
}

func TestFlashSupportSaveLoadClone(t *testing.T) {
	for _, position := range []int{0, 8, 128, 134} {
		s := supportFixture(t, position)
		wire := saveFixture(t, s)
		clone, err := s.clone()
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(wire, saveFixture(t, clone)) {
			t.Fatal("clone wire differs")
		}
		if position > 0 {
			s.support.state.History[0]++
			if bytes.Equal(saveFixture(t, s), saveFixture(t, clone)) {
				t.Fatal("clone aliases support history")
			}
		}
		dst := supportFixture(t, 0)
		dst.model = s.model
		if err := dst.load(bytes.NewReader(wire)); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(wire, saveFixture(t, dst)) {
			t.Fatal("load wire differs")
		}
		snap, err := clone.snapshot()
		if err != nil {
			t.Fatal(err)
		}
		clone.close()
		if err := dst.restore(snap); err != nil {
			t.Fatal(err)
		}
		snap.close()
		if !bytes.Equal(wire, saveFixture(t, dst)) {
			t.Fatal("snapshot close changed restored owner")
		}
		dst.close()
		s.close()
	}
}

func TestFlashSupportValidationNoMutation(t *testing.T) {
	src := supportFixture(t, 8)
	wire := saveFixture(t, src)
	for _, name := range []string{"platform", "model", "order", "context", "truncated", "journal", "checksum"} {
		t.Run(name, func(t *testing.T) {
			dst := supportFixture(t, 0)
			dst.model = src.model
			dst.steering = make([]float32, 43*4096)
			dst.steering[0], dst.attentionScale, dst.ffnScale = 3, 0.5, 0.75
			data := bytes.Clone(wire)
			switch name {
			case "platform":
				data[8] = 2
			case "model":
				dst.support.state.Model[0]++
			case "order":
				dst.support.state.Targets[0], dst.support.state.Targets[1] = 3, 1
			case "context":
				offset := 25 + int(binary.LittleEndian.Uint64(data[9:17]))
				data[offset+41] ^= 1
			case "truncated":
				data = data[:len(data)-1]
			case "journal":
				offset := 25 + int(binary.LittleEndian.Uint64(data[9:17]))
				data[offset+133+8+16] ^= 1
			case "checksum":
				data[25+int(binary.LittleEndian.Uint64(data[9:17]))-1] ^= 1
			}
			before := saveFixture(t, dst)
			steering := append([]float32(nil), dst.steering...)
			attention, ffn, owner := dst.attentionScale, dst.ffnScale, dst.support
			err := dst.load(bytes.NewReader(data))
			want := map[string]string{"platform": "missing or foreign support", "model": "destination identity mismatch", "order": "selected layer order mismatch", "context": "destination identity mismatch", "truncated": "invalid section lengths", "journal": "snapshot history mismatch", "checksum": "CPU state checksum differs"}[name]
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("got %v, want semantic rejection %q", err, want)
			}
			if !bytes.Equal(before, saveFixture(t, dst)) {
				t.Fatal("failed load partially mutated destination")
			}
			if dst.support != owner || dst.attentionScale != attention || dst.ffnScale != ffn || !reflect.DeepEqual(steering, dst.steering) {
				t.Fatal("failed load changed steering, scales or support owner")
			}
			dst.close()
		})
	}
	src.close()
}

func TestFlashSupportActualCombineBody(t *testing.T) {
	s := supportFixture(t, 0)
	block, after, dst := make([]float32, 4096), make([]float32, 4*4096), make([]float32, 4*4096)
	post, comb := []float32{1, 2, 3, 4}, make([]float32, 16)
	for i := range block {
		block[i] = 8
	}
	for lane := 0; lane < 4; lane++ {
		comb[lane+lane*4] = 1
		for d := 0; d < 4096; d++ {
			after[lane*4096+d] = float32(10 * (lane + 1))
		}
	}
	if err := s.combineLayer(dst, block, after, post, comb, 1); err != nil {
		t.Fatal(err)
	}
	// A pre-Combine observation gives 25; first-lane-only gives 18.
	// The literal float64 reference of the actual four finite terms is 45.
	var expected float64
	for lane := 0; lane < 4; lane++ {
		expected += (8*float64(post[lane]) + float64(10*(lane+1))) * 0.25
	}
	if s.support.staged[0] != float32(expected) || !s.support.captured[0] {
		t.Fatal("actual final Combine observation differs from literal reference")
	}
	if err := s.support.reset(); err != nil {
		t.Fatal(err)
	}
	if err := s.combineLayer(dst[:1], block, after, post, comb, 1); err == nil {
		t.Fatal("invalid Combine accepted")
	}
	if s.support.captured[0] || s.support.staged[0] != 0 {
		t.Fatal("failed Combine published a selected capture")
	}
	s.close()
}

func TestFlashSupportFailedPublishNoMutation(t *testing.T) {
	for _, fault := range []string{"foreign-epoch", "noncontiguous"} {
		t.Run(fault, func(t *testing.T) {
			s := supportFixture(t, 0)
			v := s.support
			hc := make([]float32, 4*4096)
			for i := range hc {
				hc[i] = 7
			}
			for _, layer := range v.state.Targets {
				if err := v.capture(layer, hc); err != nil {
					t.Fatal(err)
				}
			}
			before := append([]float32(nil), v.state.History...)
			frontier, prefix := v.state.Journal.Frontier, v.state.Journal.Prefix
			position := 1
			if fault == "foreign-epoch" {
				position = 0
				v.state.Journal.Epoch = 0
			}
			if err := v.publish(position, 1); err == nil {
				t.Fatal("invalid publication accepted")
			}
			if v.state.Journal.Frontier != frontier || v.state.Journal.Prefix != prefix || !reflect.DeepEqual(before, v.state.History) {
				t.Fatal("failed publish changed trusted history or frontier")
			}
			var dst bytes.Buffer
			if err := s.save(&dst); err == nil || dst.Len() != 0 {
				t.Fatal("pending row serialized")
			}
			s.close()
		})
	}
}

func TestFlashSupportCaptureAndPublication(t *testing.T) {
	s := supportFixture(t, 0)
	v := s.support
	hc := make([]float32, 4*4096)
	for lane := 0; lane < 4; lane++ {
		for d := 0; d < 4096; d++ {
			hc[lane*4096+d] = float32(lane + 1)
		}
	}
	if err := v.capture(1, hc); err != nil {
		t.Fatal(err)
	}
	if v.staged[0] != 2.5 {
		t.Fatal("capture is not four-lane mean")
	}
	if err := v.publish(0, 1); err == nil {
		t.Fatal("incomplete capture published")
	}
	if v.state.Journal.Frontier != 0 || v.state.History[0] != 0 {
		t.Fatal("partial row became trusted")
	}
	if err := v.capture(3, hc); err != nil {
		t.Fatal(err)
	}
	if err := v.publish(0, 1); err != nil {
		t.Fatal(err)
	}
	if v.state.Journal.Frontier != 1 || v.state.History[0] != 2.5 {
		t.Fatal("complete row not published")
	}
	hc[0] = float32(math.NaN())
	if err := v.capture(1, hc); err == nil {
		t.Fatal("nonfinite consumed capture accepted")
	}
	s.close()
}
