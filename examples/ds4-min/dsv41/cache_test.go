package dsv41

import "testing"

func TestKVOwnersAndIsolation(t *testing.T) {
	s, err := NewKVState(3)
	if err != nil {
		t.Fatal(err)
	}
	other, err := NewKVState(3)
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]float32, HeadDim)
	kv := make([]float32, HeadDim)
	idx := make([]float32, IndexDim)
	for pos := 0; pos < 3; pos++ {
		for layer := 0; layer < 40; layer++ {
			l, _ := Schedule(layer)
			for i := range raw {
				raw[i] = float32(layer*100 + pos)
			}
			for i := range kv {
				kv[i] = float32(l.Owner*100 + pos)
			}
			var k, x []float32
			if l.PublishKV && (pos+1)%l.Ratio == 0 {
				k, x = kv, idx
			}
			if err := s.Commit(layer, pos, raw, k, x); err != nil {
				t.Fatal(layer, pos, err)
			}
			r, c, _, err := s.Rows(layer, pos)
			if err != nil {
				t.Fatal(err)
			}
			if r[len(r)-1] != raw[0] {
				t.Fatal("private raw window mixed")
			}
			if len(c) > 0 && c[len(c)-1] != float32(l.Owner*100+(pos+1)/l.Ratio*l.Ratio-1) {
				t.Fatal("wrong compressed owner", layer, pos, c[len(c)-1])
			}
		}
	}
	if _, _, _, err := other.Rows(20, 0); err == nil {
		t.Fatal("uncommitted state exposed")
	}
	s.Reset()
	if _, _, _, err := s.Rows(0, 0); err == nil {
		t.Fatal("reset exposed old rows")
	}
	if err := s.Commit(1, 0, raw, nil, nil); err == nil {
		t.Fatal("layer order ignored")
	}
}
