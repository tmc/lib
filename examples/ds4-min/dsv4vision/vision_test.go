package dsv4vision

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func ExampleLayout() {
	types, perm, err := Layout(1, 1, 0)
	fmt.Println(types, perm, err)
	// Output: [1 1 1 0 2 1 3 1 4] [0] <nil>
}

func ExampleNewImage() {
	i, err := NewImage(1, 1, []byte{255, 0, 0})
	fmt.Println(i.Width, i.Height, i.RGB, err)
	// Output: 1 1 [255 0 0] <nil>
}

func ExamplePreprocess() {
	i, _ := NewImage(1, 1, []byte{127, 127, 127})
	p, err := Preprocess(i)
	fmt.Println(p.GridWidth, p.GridHeight, len(p.Values), err)
	// Output: 28 28 460992 <nil>
}

func ExampleDecode() {
	i := image.NewNRGBA(image.Rect(0, 0, 1, 1))
	i.SetNRGBA(0, 0, color.NRGBA{255, 0, 0, 255})
	var b bytes.Buffer
	_ = png.Encode(&b, i)
	got, err := Decode(&b)
	fmt.Println(got.Width, got.Height, got.RGB, err)
	// Output: 1 1 [255 0 0] <nil>
}

// TestMutationGate applies mutations to the implementation, not a duplicate
// reference. Each child must fail the ordinary pinned C gate.
func TestMutationGate(t *testing.T) {
	if os.Getenv("DS4_VISION_MUTATIONS") != "1" {
		t.Skip("set DS4_VISION_MUTATIONS=1")
	}
	if os.Getenv("DS4_VISION_ORACLE_SOURCE") == "" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("mutation gate requires pinned source and gpu-lock")
	}
	abs, err := filepath.Abs("vision.go")
	if err != nil {
		t.Fatal(err)
	}
	src, err := os.ReadFile(abs)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []struct{ name, old, new string }{
		{"centering", "ox, oy = int(math.RoundToEven(float64(bw-cw)*0.5)), int(math.RoundToEven(float64(bh-ch)*0.5))", "ox, oy = int(math.Floor(float64(bw-cw)*0.5)), int(math.Floor(float64(bh-ch)*0.5))"},
		{"compress-padding", "pad := 3 - startPos%4", "pad := 0"},
	} {
		t.Run(m.name, func(t *testing.T) {
			if strings.Count(string(src), m.old) != 1 {
				t.Fatalf("mutation site count != 1: %q", m.old)
			}
			dir := t.TempDir()
			mut := filepath.Join(dir, "vision.go")
			if err := os.WriteFile(mut, []byte(strings.Replace(string(src), m.old, m.new, 1)), 0600); err != nil {
				t.Fatal(err)
			}
			overlay, _ := json.Marshal(map[string]any{"Replace": map[string]string{abs: mut}})
			path := filepath.Join(dir, "overlay.json")
			if err := os.WriteFile(path, overlay, 0600); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("go", "test", "-mod=readonly", "-overlay", path, "-count=1", "-run", "^TestPinnedC$", ".")
			out, err := cmd.CombinedOutput()
			if err == nil {
				t.Fatalf("actual implementation mutant passed: %s", out)
			}
			if !bytes.Contains(out, []byte("--- FAIL: TestPinnedC")) {
				t.Fatalf("mutant failed outside comparison gate: %v %s", err, out)
			}
			if m.name == "centering" && !bytes.Contains(out, []byte("--- FAIL: TestPinnedC/rgb-20-4x2")) {
				t.Fatalf("centering mutant did not fail the discriminating fixture: %s", out)
			}
			t.Logf("ordinary comparison output:\n%s", out)
			t.Logf("actual %s mutant rejected by pinned comparison", m.name)
		})
	}
}

func TestDecode(t *testing.T) {
	im := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	for y := 0; y < 2; y++ {
		for x := 0; x < 3; x++ {
			im.SetNRGBA(x, y, color.NRGBA{uint8(x * 71), uint8(y * 93), 29, 255})
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, im); err != nil {
		t.Fatal(err)
	}
	i, err := Decode(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if i.Width != 3 || i.Height != 2 || i.RGB[3] != 71 {
		t.Fatal(i)
	}
	for o := 1; o <= 8; o++ {
		b.Reset()
		if err := jpeg.Encode(&b, im, &jpeg.Options{Quality: 100}); err != nil {
			t.Fatal(err)
		}
		base, err := Decode(bytes.NewReader(b.Bytes()))
		if err != nil {
			t.Fatal(err)
		}
		tiff := []byte{'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0, 18, 1, 3, 0, 1, 0, 0, 0, byte(o), 0, 0, 0, 0, 0, 0, 0}
		exif := append([]byte("Exif\x00\x00"), tiff...)
		encoded := append([]byte{255, 216, 255, 225, byte((len(exif) + 2) >> 8), byte(len(exif) + 2)}, exif...)
		encoded = append(encoded, b.Bytes()[2:]...)
		got, err := Decode(bytes.NewReader(encoded))
		if err != nil {
			t.Fatal(err)
		}
		w, h := 3, 2
		if o >= 5 {
			w, h = h, w
		}
		if got.Width != w || got.Height != h {
			t.Fatalf("orientation %d: %dx%d", o, got.Width, got.Height)
		}
		// Independently index the output-to-input permutation for every pixel.
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				sx, sy := x, y
				switch o {
				case 2:
					sx = 2 - x
				case 3:
					sx, sy = 2-x, 1-y
				case 4:
					sy = 1 - y
				case 5:
					sx, sy = y, x
				case 6:
					sx, sy = y, 1-x
				case 7:
					sx, sy = 2-y, 1-x
				case 8:
					sx, sy = 2-y, x
				}
				if !bytes.Equal(got.RGB[(y*w+x)*3:(y*w+x+1)*3], base.RGB[(sy*3+sx)*3:(sy*3+sx+1)*3]) {
					t.Fatalf("orientation %d pixel %d,%d", o, x, y)
				}
			}
		}
	}
	for _, b := range [][]byte{nil, []byte("gif"), {255, 216, 255, 225, 0, 20}} {
		if _, err := Decode(bytes.NewReader(b)); err == nil {
			t.Fatal("invalid encoded image admitted")
		}
	}
}

// TestPinnedC is a host-only oracle gate. It deliberately requires an explicit
// source path rather than accepting an arbitrary checkout from PATH.
func TestPinnedC(t *testing.T) {
	src := os.Getenv("DS4_VISION_ORACLE_SOURCE")
	if src == "" {
		t.Skip("set DS4_VISION_ORACLE_SOURCE to pinned 11e8cad checkout")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("oracle gate requires gpu-lock")
	}
	rev, err := exec.Command("git", "-C", src, "rev-parse", "HEAD").CombinedOutput()
	if err != nil || string(bytes.TrimSpace(rev)) != "11e8cad4473716a41baf4d62e92ed9764a00fce6" {
		t.Fatalf("oracle revision: %s %v", rev, err)
	}
	diff, err := exec.Command("git", "-C", src, "diff", "HEAD", "--", "ds4_image.c", "ds4_image.h", "third_party/iris").CombinedOutput()
	if err != nil || len(diff) != 0 {
		t.Fatalf("dirty oracle source: %s %v", diff, err)
	}
	// Compile immutable git objects, not the checkout paths checked above.
	frozen := t.TempDir()
	for _, name := range []string{"ds4_image.c", "ds4_image.h", "third_party/iris/jpeg.h", "third_party/iris/png.h"} {
		b, err := exec.Command("git", "-C", src, "show", "11e8cad4473716a41baf4d62e92ed9764a00fce6:"+name).Output()
		if err != nil {
			t.Fatalf("read frozen %s: %v", name, err)
		}
		path := filepath.Join(frozen, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, b, 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("oracle source %s sha256=%x", name, sha256.Sum256(b))
	}
	src = frozen
	bin := filepath.Join(t.TempDir(), "image-oracle")
	cmd := exec.Command("cc", "-O2", "-ffp-contract=off", "-I", src, "testdata/oracle.c", filepath.Join(src, "ds4_image.c"), "-lm", "-o", bin)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile oracle: %v %s", err, out)
	}
	run := func(mode string, h, w, pos int, rgb []byte) ([]byte, error) {
		cmd := exec.Command(bin, mode, strconv.Itoa(h), strconv.Itoa(w), strconv.Itoa(pos))
		cmd.Stdin = bytes.NewReader(rgb)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		b, err := cmd.Output()
		if err != nil {
			return nil, fmt.Errorf("%w: %s", err, stderr.Bytes())
		}
		return b, nil
	}
	sizes := [][2]int{{1, 1}, {1, 19}, {19, 1}, {2, 3}, {3, 2}, {7, 9}, {9, 7}, {13, 15}, {15, 13}, {27, 29}, {29, 27}, {63, 65}, {65, 63}, {97, 101}, {101, 97}, {64, 512}, {64, 513}, {512, 64}, {513, 64}, {385, 383}, {2, 4}}
	for j, size := range sizes {
		t.Run(fmt.Sprintf("rgb-%02d-%dx%d", j, size[1], size[0]), func(t *testing.T) {
			h, w := size[0], size[1]
			rgb := make([]byte, h*w*3)
			for k := range rgb {
				rgb[k] = byte((k*37 + k/11 + j*53) % 256)
			}
			i, err := NewImage(w, h, rgb)
			if err != nil {
				t.Fatal(err)
			}
			p, err := Preprocess(i)
			if err != nil {
				t.Fatal(err)
			}
			want, err := run("rgb", h, w, 0, rgb)
			if err != nil {
				t.Fatal(err)
			}
			meta := []int{p.ContentWidth, p.ContentHeight, p.PaddedWidth, p.PaddedHeight, p.GridWidth, p.GridHeight, p.LLMGridWidth, p.LLMGridHeight, p.GridWidth * p.GridHeight}
			if len(want) != 36+len(p.Values)*4 {
				t.Fatalf("length: %d want %d", len(want), 36+len(p.Values)*4)
			}
			for k, v := range meta {
				if uint32(v) != binary.LittleEndian.Uint32(want[k*4:]) {
					t.Fatalf("metadata %d: %d vs %d", k, v, binary.LittleEndian.Uint32(want[k*4:]))
				}
			}
			for k, v := range p.Values {
				if math.Float32bits(v) != binary.LittleEndian.Uint32(want[36+k*4:]) {
					t.Fatalf("patch %d: bits %08x vs %08x", k, math.Float32bits(v), binary.LittleEndian.Uint32(want[36+k*4:]))
				}
			}
		})
	}
	accepted, rejected := 0, 0
	for h := 1; h <= 27; h++ {
		for w := 1; w <= 27; w++ {
			for pos := 0; pos < 4; pos++ {
				types, perm, err := Layout(h, w, pos)
				want, cerr := run("layout", h, w, pos, nil)
				if (err == nil) != (cerr == nil) {
					t.Fatalf("layout admission %d/%d/%d: %v vs %v", h, w, pos, err, cerr)
				}
				if err != nil {
					rejected++
					continue
				}
				accepted++
				if len(want) != 8+len(types)+4*len(perm) || int(binary.LittleEndian.Uint32(want)) != len(types) || int(binary.LittleEndian.Uint32(want[4:])) != len(perm) {
					t.Fatal("layout shape")
				}
				for j, v := range types {
					if byte(v) != want[8+j] {
						t.Fatalf("layout types %d/%d/%d index %d", h, w, pos, j)
					}
				}
				for j, v := range perm {
					if uint32(v) != binary.LittleEndian.Uint32(want[8+len(types)+j*4:]) {
						t.Fatalf("layout permutation %d/%d/%d index %d", h, w, pos, j)
					}
				}
			}
		}
	}
	t.Logf("%d decoded-RGB preprocess fixtures; layout accepted=%d rejected=%d", len(sizes), accepted, rejected)
}
