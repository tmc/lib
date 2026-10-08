package ncclinspector

import (
	"bytes"
	"io"
	"reflect"
	"testing"
)

func FuzzDecoder(f *testing.F) {
	f.Add(fixture(f, "stats"), uint16(13))
	f.Add([]byte("{}\n"), uint16(1))
	f.Fuzz(func(t *testing.T, b []byte, split uint16) {
		if len(b) > 4096 {
			return
		}
		limits := Limits{Bytes: 4096, Line: 2048, Records: 8}
		var input bytes.Buffer
		d, err := NewDecoder(&input, limits)
		if err != nil {
			t.Fatal(err)
		}
		cut := int(split) % (len(b) + 1)
		var got []Record
		var terminal error
		for _, chunk := range [][]byte{b[:cut], b[cut:]} {
			input.Write(chunk)
			for terminal == nil {
				r, err := d.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					terminal = err
					break
				}
				got = append(got, r)
			}
		}
		if terminal == nil {
			terminal = d.Finish()
		}
		want, err := Read(bytes.NewReader(b), limits)
		if (terminal == nil) != (err == nil) {
			t.Fatalf("finite/incremental acceptance differs: %v, %v", err, terminal)
		}
		if err == nil && !reflect.DeepEqual(got, want) {
			t.Fatal("incremental records differ from finite capture")
		}
		var committed int64
		for _, r := range got {
			committed += int64(len(r.Raw))
		}
		if d.Offset() != committed {
			t.Fatal("offset includes incomplete or refused bytes")
		}
	})
}

func FuzzFilePosition(f *testing.F) {
	p := FilePosition{device: 1, inode: 2, epoch: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", offset: 4, high: 8, records: 1, limits: Limits{Bytes: 100, Line: 10, Records: 3}}
	b, err := p.MarshalText()
	if err != nil {
		f.Fatal(err)
	}
	f.Add(b)
	f.Add([]byte("{}"))
	f.Fuzz(func(t *testing.T, b []byte) {
		var p FilePosition
		if err := p.UnmarshalText(b); err != nil {
			return
		}
		canonical, err := p.MarshalText()
		if err != nil {
			t.Fatal(err)
		}
		var q FilePosition
		if err := q.UnmarshalText(canonical); err != nil || !reflect.DeepEqual(p, q) {
			t.Fatal("cursor round trip changed resume facts")
		}
	})
}
