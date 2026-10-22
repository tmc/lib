package dsv4shape

import (
	"fmt"
	"testing"
)

func ExampleConfig_Layer() {
	c, _ := For(Pro)
	l, _ := c.Layer(0)
	fmt.Println(c.Layers, l.CompressRatio, l.Indexer)
	// Output: 61 128 false
}

func ExampleFor() {
	c, _ := For(Pro)
	fmt.Println(c.Embed, c.Experts, c.IndexerTopK)
	// Output: 7168 384 1024
}

func ExampleConfig_Validate() {
	c, _ := For(Flash)
	fmt.Println(c.Validate())
	// Output: <nil>
}

func ExampleSelect() {
	_, err := Select(nil)
	fmt.Println(err)
	// Output: shape: unsupported or incomplete deepseek4 metadata
}

func TestShapes(t *testing.T) {
	for _, v := range []Variant{Flash, Pro} {
		c, err := For(v)
		if err != nil {
			t.Fatal(err)
		}
		md := make(map[string]uint64)
		for k, n := range c.fields() {
			md["deepseek4."+k] = uint64(n)
		}
		got, err := Select(md)
		if err != nil || got != c {
			t.Fatalf("select: %v %v", got, err)
		}
		for k := range md {
			x := md[k]
			md[k]++
			if _, err := Select(md); err == nil {
				t.Fatalf("accepted mutation %s", k)
			}
			delete(md, k)
			if _, err := Select(md); err == nil {
				t.Fatalf("accepted missing %s", k)
			}
			md[k] = x
		}
		for i := 0; i < c.Layers; i++ {
			l, err := c.Layer(i)
			want := 128
			if i < 2 && v == Flash {
				want = 0
			} else if i >= 2 && i%2 == 0 {
				want = 4
			}
			if err != nil || l.CompressRatio != want || l.Indexer != (want == 4) || l.HashRouter != (i < 3) {
				t.Fatalf("variant %d layer %d: %+v %v", v, i, l, err)
			}
		}
		for _, i := range []int{-1, c.Layers} {
			if _, err := c.Layer(i); err == nil {
				t.Fatal("invalid layer accepted")
			}
		}
		c.Experts++
		if _, err := c.Layer(0); err == nil {
			t.Fatal("mutated config accepted")
		}
	}
	if _, err := For(2); err == nil {
		t.Fatal("unknown variant accepted")
	}
}
