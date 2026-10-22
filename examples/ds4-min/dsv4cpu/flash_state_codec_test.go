package dsv4cpu

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"testing"
)

// This fixture fills native request buffers directly; it tests ownership and
// the codec, not a claim that a complete inference has run.
func codecFixture(t *testing.T) *flashState {
	return codecFixturePosition(t, 8)
}

func codecFixturePosition(t *testing.T, position int) *flashState {
	t.Helper()
	tensors, size := virtualFlashDescriptors()
	m, err := newFlashModel(&virtualFlashSource{}, size, tensors)
	if err != nil {
		t.Fatal(err)
	}
	m.digest = sha256.Sum256([]byte("virtual Flash ownership fixture"))
	s, err := m.newState(4096)
	if err != nil {
		t.Fatal(err)
	}
	s.position = position
	s.logits[0] = 7
	for il := range s.layers {
		l := &s.layers[il]
		l.raw.count = min(position, 128)
		for i := 0; i < l.raw.count*512; i++ {
			l.raw.rows[i] = float32((il+i)%17-8) / 4
		}
		if l.attention != nil {
			l.attention.position = position
			l.compressed = l.compressed[:position/l.attention.ratio*512]
			for i := range l.compressed {
				l.compressed[i] = float32(i%19-9) / 8
			}
			for i := range l.attention.kv {
				l.attention.kv[i] = float32(i%7 - 3)
			}
			if l.index != nil {
				l.index.position = position
				l.indexRows = l.indexRows[:position/4*128]
				for i := range l.indexRows {
					l.indexRows[i] = float32(i % 5)
				}
			}
		}
	}
	return s
}

func saveFixture(t *testing.T, s *flashState) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := s.save(&b); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestFlashSnapshotOwnership(t *testing.T) {
	s := codecFixture(t)
	snap, err := s.snapshot()
	if err != nil {
		t.Fatal(err)
	}
	raw := s.layers[0].raw.rows[0]
	s.logits[0] = 13
	s.layers[0].raw.rows[0] = 17
	if err := s.restore(snap); err != nil {
		t.Fatal(err)
	}
	if s.logits[0] != 7 || s.layers[0].raw.rows[0] != raw {
		t.Fatal("snapshot borrowed mutable source")
	}
	s.layers[0].raw.rows[0] = 19
	if snap.state.layers[0].raw.rows[0] != raw {
		t.Fatal("restore borrowed snapshot storage")
	}
	foreign := codecFixture(t)
	if err := foreign.restore(snap); err == nil {
		t.Fatal("foreign snapshot accepted")
	}
	snap.close()
	if err := s.restore(snap); err == nil {
		t.Fatal("closed snapshot accepted")
	}
	s.close()
	if _, err := s.snapshot(); err == nil {
		t.Fatal("closed state snapshot accepted")
	}
}

func TestFlashCodecRoundTrip(t *testing.T) {
	s := codecFixture(t)
	raw := saveFixture(t, s)
	dst, err := s.model.newState(s.context)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.load(bytes.NewReader(raw)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, saveFixture(t, dst)) {
		t.Fatal("native round trip changed exact state bits")
	}
	if dst.layers[2].attention.position != 8 || dst.layers[3].attention.position != 8 {
		t.Fatal("compressor timeline not restored")
	}
	dst.layers[0].raw.rows[0] = 100
	if s.layers[0].raw.rows[0] == 100 {
		t.Fatal("load shared source state")
	}
	t.Logf("native state payload %d bytes, all43raw/compressor/indexer/logit buffers round trip exactly", len(raw))
}

func TestFlashCodecBoundaryRoundTrips(t *testing.T) {
	for _, pos := range []int{127, 128, 129, 2048} {
		t.Run(fmt.Sprint(pos), func(t *testing.T) {
			s := codecFixturePosition(t, pos)
			defer s.close()
			raw := saveFixture(t, s)
			dst, err := s.model.newState(s.context)
			if err != nil {
				t.Fatal(err)
			}
			defer dst.close()
			stream := bytes.NewReader(append(raw, 7, 9))
			if err := dst.load(stream); err != nil {
				t.Fatal(err)
			}
			if stream.Len() != 2 {
				t.Fatal("load consumed trailing frame bytes")
			}
			if !bytes.Equal(raw, saveFixture(t, dst)) {
				t.Fatal("boundary round trip changed state")
			}
			if dst.layers[2].attention.position != pos || dst.layers[3].attention.position != pos {
				t.Fatal("boundary timeline differs")
			}
			snap, err := dst.snapshot()
			if err != nil {
				t.Fatal(err)
			}
			defer snap.close()
			dst.layers[0].raw.rows[0]++
			if err := dst.restore(snap); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(raw, saveFixture(t, dst)) {
				t.Fatal("boundary snapshot changed state")
			}
		})
	}
}

func TestFlashCodecChecksummedInvalidIsAtomic(t *testing.T) {
	s := codecFixture(t)
	defer s.close()
	raw := saveFixture(t, s)
	for _, tc := range []struct {
		name   string
		offset int
		bits   uint32
	}{
		{"position", 44, uint32(s.context + 1)},
		{"count", 48, 127},
		{"nan", 52, 0x7fc00000},
		{"infinity", 52, 0x7f800000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := append([]byte(nil), raw...)
			binary.LittleEndian.PutUint32(bad[tc.offset:], tc.bits)
			sum := sha256.Sum256(bad[:len(bad)-32])
			copy(bad[len(bad)-32:], sum[:])
			if err := s.load(bytes.NewReader(bad)); err == nil {
				t.Fatal("checksummed invalid payload accepted")
			}
			if !bytes.Equal(raw, saveFixture(t, s)) {
				t.Fatal("invalid load changed destination")
			}
		})
	}
}

func TestFlashCodecFailedLoadIsAtomic(t *testing.T) {
	s := codecFixture(t)
	raw := saveFixture(t, s)
	before := append([]byte(nil), raw...)
	cases := [][]byte{raw[:0], raw[:7], raw[:47], raw[:len(raw)-1]}
	corrupt := append([]byte(nil), raw...)
	corrupt[len(corrupt)-33] ^= 1
	cases = append(cases, corrupt)
	wrongModel := append([]byte(nil), raw...)
	wrongModel[8] ^= 1
	cases = append(cases, wrongModel)
	wrongCount := append([]byte(nil), raw...)
	wrongCount[48] = 255
	cases = append(cases, wrongCount)
	for i, b := range cases {
		if err := s.load(bytes.NewReader(b)); err == nil {
			t.Fatal("bad payload accepted", i)
		}
		if !bytes.Equal(before, saveFixture(t, s)) {
			t.Fatal("failed load changed destination", i)
		}
	}
}

type shortStateWriter struct{}

func (shortStateWriter) Write(p []byte) (int, error) { return len(p) / 2, nil }

func TestFlashCodecSaveAdmission(t *testing.T) {
	s := codecFixture(t)
	if err := s.save(shortStateWriter{}); err != io.ErrShortWrite {
		t.Fatalf("short writer got %v", err)
	}
	s.model.digest = [32]byte{}
	if err := s.save(io.Discard); err == nil {
		t.Fatal("unidentified model saved")
	}
}
