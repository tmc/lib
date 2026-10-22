package dsv41

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"reflect"
	"testing"
)

func TestAssetHash(t *testing.T) {
	data := []byte("complete immutable asset, not just metadata")
	got, err := hashAsset(bytes.NewReader(data), int64(len(data)))
	if err != nil || got != sha256.Sum256(data) {
		t.Fatal(got, err)
	}
	changed := append([]byte(nil), data...)
	changed[len(changed)-1] ^= 1
	other, err := hashAsset(bytes.NewReader(changed), int64(len(changed)))
	if err != nil || other == got {
		t.Fatal("asset content change invisible", err)
	}
	if _, err = hashAsset(bytes.NewReader(data), int64(len(data)+1)); err == nil {
		t.Fatal("short asset accepted")
	}
	if sealed, err := verifyAsset(bytes.NewReader(data), int64(len(data)), got); err != nil || sealed != got {
		t.Fatal("asset seal rejected", err)
	}
	if _, err = verifyAsset(bytes.NewReader(changed), int64(len(changed)), got); err == nil {
		t.Fatal("wrong asset seal accepted")
	}
}

func TestCommittedResultOwnership(t *testing.T) {
	g := modelFixture(t)
	m, err := Bind(&countReader{}, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	s, err := m.NewState(4)
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float32, 129280)
	values[0] = 7
	s.kv.position = 1
	s.commitResult(3, values)
	values[0] = 1000
	if s.logits[0] != 7 {
		t.Fatal("caller changed retained logits")
	}
	c, err := s.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	s.tokens[0] = 90
	s.logits[0] = 99
	if err = s.Restore(c); err != nil {
		t.Fatal(err)
	}
	if s.tokens[0] != 3 || s.logits[0] != 7 {
		t.Fatal("checkpoint history aliased")
	}
}

// payloadFixture populates synthetic request storage, not model inference. Its
// asset identity is planted only to exercise the codec's identity comparison.
func payloadFixture(t *testing.T, pos int) *State {
	t.Helper()
	g := modelFixture(t)
	mapValues := g.Metadata["deepseek41.engram.token_map"].Value.([]any)
	for i := range mapValues {
		mapValues[i] = uint32(i * 3 % 99092)
	}
	m, err := Bind(&countReader{}, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	m.assetSHA = sha256.Sum256([]byte("synthetic codec control"))
	s, err := m.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	s.position = pos
	s.kv.position = pos
	s.tokens = make([]int32, pos)
	s.logits = make([]float32, 129280)
	for i := range s.tokens {
		s.tokens[i] = int32(i % 129280)
	}
	fill := func(x []float32, tag int) {
		for i := range x {
			x[i] = float32(tag) + float32(i%31)/32
		}
	}
	fill(s.logits, 99)
	for layer := range s.kv.raw {
		fill(s.kv.raw[layer], layer)
	}
	for owner := range s.kv.owners {
		ratio := 2
		if owner == 3 {
			ratio = 1
		}
		s.kv.owners[owner].kv = make([]float32, pos/ratio*512)
		s.kv.owners[owner].index = make([]float32, pos/ratio*128)
		fill(s.kv.owners[owner].kv, 40+owner*4)
		fill(s.kv.owners[owner].index, 41+owner*4)
		if owner < 3 {
			s.pool[owner].position = pos
			if pos&1 != 0 {
				s.pool[owner].kv = make([]float32, 512)
				s.pool[owner].scores = make([]float32, 512)
				fill(s.pool[owner].kv, 42+owner*4)
				fill(s.pool[owner].scores, 43+owner*4)
			}
		}
	}
	return s
}

func TestPayloadRoundTripAndRefusal(t *testing.T) {
	for _, pos := range []int{1, 2, 127, 128, 129} {
		t.Run(fmt.Sprint(pos), func(t *testing.T) {
			s := payloadFixture(t, pos)
			var wire bytes.Buffer
			if err := s.WritePayload(&wire); err != nil {
				t.Fatal(err)
			}
			if int64(wire.Len()) != payloadBytes(pos) {
				t.Fatal("payload extent")
			}
			next, err := s.model.NewState(132)
			if err != nil {
				t.Fatal(err)
			}
			if err = next.RestorePayload(bytes.NewReader(wire.Bytes()), int64(wire.Len()), s.model.AssetSHA256()); err != nil {
				t.Fatal(err)
			}
			history := next.engram.history
			history.Reset()
			ids := make([]uint32, pos*48)
			if err = s.model.layout.Hash(&history, s.tokens, nil, ids); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(history, next.engram.history) {
				t.Fatal("restored Engram history differs from full token history")
			}
			var again bytes.Buffer
			if err = next.WritePayload(&again); err != nil || !bytes.Equal(wire.Bytes(), again.Bytes()) {
				t.Fatal("payload round trip differs", err)
			}
			before := append([]byte(nil), again.Bytes()...)
			checkUnchanged := func() {
				var current bytes.Buffer
				if err := next.WritePayload(&current); err != nil || !bytes.Equal(current.Bytes(), before) {
					t.Fatal("refusal mutated state", err)
				}
			}
			// Corrupt each serialized float family independently. Even a caller
			// sealing these modified bytes must not admit non-finite state.
			offsets := []int{52 + 4*pos}
			offset := 52 + 4*(pos+129280)
			for layer := 0; layer < 40; layer++ {
				offsets = append(offsets, offset)
				offset += 4 * min(pos, 128) * 512
			}
			for owner := 0; owner < 4; owner++ {
				ratio := 2
				if owner == 3 {
					ratio = 1
				}
				for _, width := range []int{512, 128} {
					if pos/ratio > 0 {
						offsets = append(offsets, offset)
					}
					offset += 4 * (pos / ratio) * width
				}
				if owner < 3 && pos&1 != 0 {
					for range 2 {
						offsets = append(offsets, offset)
						offset += 4 * 512
					}
				}
			}
			for _, off := range offsets {
				for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
					bad := append([]byte(nil), wire.Bytes()...)
					binary.LittleEndian.PutUint32(bad[off:], math.Float32bits(value))
					if err = next.RestorePayload(bytes.NewReader(bad), int64(len(bad)), s.model.AssetSHA256()); err == nil {
						t.Fatalf("non-finite at offset %d accepted", off)
					}
					checkUnchanged()
				}
			}
			fields := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
			if pos != 1 {
				fields = nil
			}
			for _, field := range fields {
				bad := append([]byte(nil), wire.Bytes()...)
				binary.LittleEndian.PutUint32(bad[field*4:], 0xffffffff)
				if err = next.RestorePayload(bytes.NewReader(bad), int64(len(bad)), s.model.AssetSHA256()); err == nil {
					t.Fatalf("field%d accepted", field)
				}
				checkUnchanged()
			}
			wrong := s.model.AssetSHA256()
			wrong[0] ^= 1
			if err = next.RestorePayload(bytes.NewReader(wire.Bytes()), int64(wire.Len()), wrong); err == nil {
				t.Fatal("foreign asset accepted")
			}
			if err = next.RestorePayload(bytes.NewReader(wire.Bytes()[:wire.Len()-1]), int64(wire.Len()), s.model.AssetSHA256()); err == nil {
				t.Fatal("short payload accepted")
			}
			checkUnchanged()
		})
	}
}

func ExampleBindVerified() {
	_, err := BindVerified(bytes.NewReader(nil), 0, nil, [32]byte{})
	fmt.Println(err != nil) // Output: true
}

func TestPayloadWriterPreflight(t *testing.T) {
	s := payloadFixture(t, 129)
	check := func() {
		t.Helper()
		var wire bytes.Buffer
		if err := s.WritePayload(&wire); err == nil || wire.Len() != 0 {
			t.Fatal("invalid state wrote payload bytes", err, wire.Len())
		}
	}
	families := [][]float32{s.logits}
	for _, raw := range s.kv.raw {
		families = append(families, raw)
	}
	for owner, o := range s.kv.owners {
		families = append(families, o.kv, o.index)
		if owner < 3 {
			families = append(families, s.pool[owner].kv, s.pool[owner].scores)
		}
	}
	for _, values := range families {
		old := values[0]
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			values[0] = bad
			check()
		}
		values[0] = old
	}
	for layer := range s.kv.raw {
		raw := s.kv.raw[layer]
		s.kv.raw[layer] = raw[:len(raw)-1]
		check()
		s.kv.raw[layer] = raw
	}
	var valid bytes.Buffer
	if err := s.WritePayload(&valid); err != nil || valid.Len() == 0 {
		t.Fatal("valid state rejected", err)
	}
}
func ExampleModel_AssetSHA256() {
	var model Model
	fmt.Println(model.AssetSHA256() == ([32]byte{})) // Output: true
}
func ExampleState_WritePayload() {
	var state State
	fmt.Println(state.WritePayload(io.Discard) != nil) // Output: true
}
func ExampleState_RestorePayload() {
	var state State
	fmt.Println(state.RestorePayload(bytes.NewReader(nil), 0, [32]byte{}) != nil) // Output: true
}
