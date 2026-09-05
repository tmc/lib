package nvjpeg2k

import (
	"strings"
	"testing"
)

// The checks in decode.go run entirely on the host: they compare what a caller
// says about its device buffers against what the bitstream metadata requires.
// These tests exercise them with hand-built layouts, so they need no library
// and no GPU. What they cannot show is that a layout read from a real
// bitstream is correct; that needs a decode. See physical_test.go.

// layout builds a Layout the way layoutOf would, without calling the library.
func layout(pixelType ImageType, format ImageFormat, comps ...ComponentLayout) *Layout {
	return &Layout{
		Info:       ImageInfo{ImageWidth: comps[0].Width, ImageHeight: comps[0].Height, NumTilesX: 1, NumTilesY: 1, NumComponents: uint32(len(comps))},
		PixelType:  pixelType,
		Format:     format,
		Components: comps,
		NLTKnown:   true,
		src:        Bitstream(1),
	}
}

func TestBytesPerSample(t *testing.T) {
	cases := []struct {
		typ  ImageType
		want int
	}{
		{Uint8, 1}, {Uint16, 2}, {Int16, 2}, {FP16, 2}, {FP32, 4}, {ImageType(99), 0},
	}
	for _, c := range cases {
		if got := c.typ.BytesPerSample(); got != c.want {
			t.Errorf("%v.BytesPerSample() = %d, want %d", c.typ, got, c.want)
		}
	}
}

// TestBuffersPerComponent is the point of the whole layer: a 4:2:0 image has a
// chroma plane half the width and half the height of the luma plane, and
// sizing all three from the image dimensions is the mistake this package
// exists to prevent.
func TestBuffersPerComponent(t *testing.T) {
	l := layout(Uint8, FormatPlanar,
		ComponentLayout{Width: 64, Height: 32, Precision: 8},
		ComponentLayout{Width: 32, Height: 16, Precision: 8},
		ComponentLayout{Width: 32, Height: 16, Precision: 8},
	)
	got := l.Buffers()
	want := []BufferRequirement{
		{MinPitchInBytes: 64, Rows: 32, MinSizeInBytes: 2048},
		{MinPitchInBytes: 32, Rows: 16, MinSizeInBytes: 512},
		{MinPitchInBytes: 32, Rows: 16, MinSizeInBytes: 512},
	}
	if len(got) != len(want) {
		t.Fatalf("Buffers() returned %d requirements, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Buffers()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestBuffersUint16Pitch(t *testing.T) {
	l := layout(Uint16, FormatPlanar, ComponentLayout{Width: 10, Height: 4, Precision: 12})
	got := l.Buffers()
	want := BufferRequirement{MinPitchInBytes: 20, Rows: 4, MinSizeInBytes: 80}
	if len(got) != 1 || got[0] != want {
		t.Errorf("Buffers() = %+v, want one %+v", got, want)
	}
}

func TestBuffersInterleaved(t *testing.T) {
	c := ComponentLayout{Width: 16, Height: 8, Precision: 8}
	l := layout(Uint8, FormatInterleaved, c, c, c)
	got := l.Buffers()
	want := BufferRequirement{MinPitchInBytes: 48, Rows: 8, MinSizeInBytes: 384}
	if len(got) != 1 {
		t.Fatalf("interleaved layout wants %d buffers, want 1", len(got))
	}
	if got[0] != want {
		t.Errorf("Buffers()[0] = %+v, want %+v", got[0], want)
	}
}

func TestBindAccepts(t *testing.T) {
	l := layout(Uint8, FormatPlanar,
		ComponentLayout{Width: 64, Height: 32, Precision: 8},
		ComponentLayout{Width: 32, Height: 16, Precision: 8},
	)
	dst, err := l.Bind(
		DeviceBuffer{Ptr: 0x1000, PitchInBytes: 128, SizeInBytes: 128 * 32},
		DeviceBuffer{Ptr: 0x9000, PitchInBytes: 32, SizeInBytes: 512},
	)
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	if dst.image.NumComponents != 2 {
		t.Errorf("Image.NumComponents = %d, want 2", dst.image.NumComponents)
	}
	if dst.image.PixelType != Uint8 {
		t.Errorf("Image.PixelType = %v, want Uint8", dst.image.PixelType)
	}
	if dst.image.PixelData == nil || *dst.image.PixelData != 0x1000 {
		t.Error("Image.PixelData does not point at the first device address")
	}
	if dst.image.PitchInBytes == nil || *dst.image.PitchInBytes != 128 {
		t.Error("Image.PitchInBytes does not point at the first pitch")
	}
	if dst.ptrs[1] != 0x9000 || dst.pitches[1] != 32 {
		t.Errorf("second component bound as ptr %#x pitch %d", dst.ptrs[1], dst.pitches[1])
	}
	if dst.Layout() != l {
		t.Error("Layout() does not return the layout the destination was bound against")
	}
	if bufs := dst.Buffers(); len(bufs) != 2 || bufs[0].Ptr != 0x1000 {
		t.Errorf("Buffers() = %+v", bufs)
	}
}

func TestBindRejects(t *testing.T) {
	l := layout(Uint8, FormatPlanar,
		ComponentLayout{Width: 64, Height: 32, Precision: 8},
		ComponentLayout{Width: 32, Height: 16, Precision: 8},
	)
	ok0 := DeviceBuffer{Ptr: 0x1000, PitchInBytes: 64, SizeInBytes: 2048}
	ok1 := DeviceBuffer{Ptr: 0x9000, PitchInBytes: 32, SizeInBytes: 512}
	cases := []struct {
		name string
		bufs []DeviceBuffer
		want string
	}{
		{"too few buffers", []DeviceBuffer{ok0}, "layout needs 2"},
		{"too many buffers", []DeviceBuffer{ok0, ok1, ok1}, "layout needs 2"},
		{"nil device pointer", []DeviceBuffer{ok0, {PitchInBytes: 32, SizeInBytes: 512}}, "nil device pointer"},
		{
			// The image is 64 wide, the chroma plane 32; a caller that sized
			// every plane from the image dimensions gets caught here.
			name: "pitch below the component width",
			bufs: []DeviceBuffer{{Ptr: 0x1000, PitchInBytes: 32, SizeInBytes: 2048}, ok1},
			want: "pitch 32 is below the required 64",
		},
		{
			name: "buffer too small for its own pitch",
			bufs: []DeviceBuffer{{Ptr: 0x1000, PitchInBytes: 128, SizeInBytes: 2048}, ok1},
			want: "needs 4096",
		},
		{
			name: "buffer too small for the component height",
			bufs: []DeviceBuffer{ok0, {Ptr: 0x9000, PitchInBytes: 32, SizeInBytes: 256}},
			want: "needs 512",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := l.Bind(c.bufs...)
			if err == nil {
				t.Fatal("Bind accepted a destination it should have rejected")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("Bind error = %q, want it to mention %q", err, c.want)
			}
		})
	}
}

func TestCheckPixelType(t *testing.T) {
	cases := []struct {
		name   string
		layout *Layout
		want   string // "" means accepted
	}{
		{
			name:   "8-bit into Uint8",
			layout: layout(Uint8, FormatPlanar, ComponentLayout{Width: 4, Height: 4, Precision: 8}),
		},
		{
			name:   "12-bit into Uint8",
			layout: layout(Uint8, FormatPlanar, ComponentLayout{Width: 4, Height: 4, Precision: 12}),
			want:   "precision 12, too wide for Uint8",
		},
		{
			name:   "12-bit into Uint16",
			layout: layout(Uint16, FormatPlanar, ComponentLayout{Width: 4, Height: 4, Precision: 12}),
		},
		{
			name: "second component too wide",
			layout: layout(Uint8, FormatPlanar,
				ComponentLayout{Width: 4, Height: 4, Precision: 8},
				ComponentLayout{Width: 4, Height: 4, Precision: 10}),
			want: "component 1 has precision 10",
		},
		{
			name: "FP16 without NLT",
			layout: layout(FP16, FormatPlanar,
				ComponentLayout{Width: 4, Height: 4, Precision: 16, NLT: NltNone}),
			want: "requires NLT type 3 on every component",
		},
		{
			name: "FP16 with BDnlt 16",
			layout: layout(FP16, FormatPlanar,
				ComponentLayout{Width: 4, Height: 4, Precision: 16, NLT: NltType3, NLTBitDepth: 16}),
		},
		{
			name: "FP16 with BDnlt 32",
			layout: layout(FP16, FormatPlanar,
				ComponentLayout{Width: 4, Height: 4, Precision: 32, NLT: NltType3, NLTBitDepth: 32}),
			want: "FP16 requires BDnlt 16",
		},
		{
			name: "FP32 with BDnlt 32",
			layout: layout(FP32, FormatPlanar,
				ComponentLayout{Width: 4, Height: 4, Precision: 32, NLT: NltType3, NLTBitDepth: 32}),
		},
		{
			name: "FP32 with BDnlt 8",
			layout: layout(FP32, FormatPlanar,
				ComponentLayout{Width: 4, Height: 4, Precision: 8, NLT: NltType3, NLTBitDepth: 8}),
			want: "FP32 requires BDnlt 16 or 32",
		},
		{
			name:   "unknown pixel type",
			layout: layout(ImageType(99), FormatPlanar, ComponentLayout{Width: 4, Height: 4, Precision: 8}),
			want:   "unsupported pixel type",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.layout.checkPixelType()
			switch {
			case c.want == "" && err != nil:
				t.Errorf("checkPixelType rejected a valid layout: %v", err)
			case c.want != "" && err == nil:
				t.Error("checkPixelType accepted an invalid layout")
			case c.want != "" && !strings.Contains(err.Error(), c.want):
				t.Errorf("checkPixelType error = %q, want it to mention %q", err, c.want)
			}
		})
	}
}

// TestCheckPixelTypeFloatNeedsNLTSymbol checks that a floating-point
// destination is refused, rather than quietly allowed, when the loaded library
// is too old to report NLT information.
func TestCheckPixelTypeFloatNeedsNLTSymbol(t *testing.T) {
	l := layout(FP32, FormatPlanar, ComponentLayout{Width: 4, Height: 4, Precision: 32})
	l.NLTKnown = false
	err := l.checkPixelType()
	if err == nil {
		t.Fatal("checkPixelType accepted FP32 without NLT information")
	}
	if !strings.Contains(err.Error(), "nvjpeg2kStreamGetComponentNLTInfo") {
		t.Errorf("error = %q, want it to name the missing symbol", err)
	}
}

func TestParamsCheckAgainst(t *testing.T) {
	planar := layout(Uint8, FormatPlanar, ComponentLayout{Width: 4, Height: 4, Precision: 8})
	interleaved := layout(Uint8, FormatInterleaved, ComponentLayout{Width: 4, Height: 4, Precision: 8})

	if err := (*Params)(nil).checkAgainst(planar); err != nil {
		t.Errorf("nil params rejected a planar layout: %v", err)
	}
	if err := (*Params)(nil).checkAgainst(interleaved); err == nil {
		t.Error("nil params accepted an interleaved layout; the library default is planar")
	}

	p := &Params{handle: DecodeParams(1), format: FormatPlanar}
	if err := p.checkAgainst(planar); err != nil {
		t.Errorf("planar params rejected a planar layout: %v", err)
	}
	if err := p.checkAgainst(interleaved); err == nil {
		t.Error("planar params accepted an interleaved layout")
	}

	p.format = FormatInterleaved
	if err := p.checkAgainst(interleaved); err != nil {
		t.Errorf("interleaved params rejected an interleaved layout: %v", err)
	}

	p.format = FormatPlanar
	p.area = true
	err := p.checkAgainst(planar)
	if err == nil {
		t.Fatal("params with a decode area were accepted")
	}
	if !strings.Contains(err.Error(), "decode area") {
		t.Errorf("error = %q, want it to mention the decode area", err)
	}

	destroyed := &Params{}
	if err := destroyed.checkAgainst(planar); err == nil {
		t.Error("a destroyed Params was accepted")
	}
}

// TestDecodeIntoGuards checks the guards that run before any library call.
// They are reached with a zero Handle because none of them touches the
// library, which is what lets them be tested here.
func TestDecodeIntoGuards(t *testing.T) {
	l := layout(Uint8, FormatPlanar, ComponentLayout{Width: 4, Height: 4, Precision: 8})
	dst, err := l.Bind(DeviceBuffer{Ptr: 0x1000, PitchInBytes: 4, SizeInBytes: 16})
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	tileLayout := layout(Uint8, FormatPlanar, ComponentLayout{Width: 2, Height: 2, Precision: 8})
	tileLayout.Tile, tileLayout.TileID = true, 1
	tileDst, err := tileLayout.Bind(DeviceBuffer{Ptr: 0x2000, PitchInBytes: 2, SizeInBytes: 4})
	if err != nil {
		t.Fatalf("Bind tile: %v", err)
	}

	var h Handle
	cases := []struct {
		name string
		call func() error
		want string
	}{
		{"nil destination", func() error {
			return h.DecodeInto(0, l.src, nil, nil, 0)
		}, "nil destination"},
		{"wrong bitstream", func() error {
			return h.DecodeInto(0, Bitstream(2), nil, dst, 0)
		}, "different bitstream"},
		{"tile destination", func() error {
			return h.DecodeInto(0, tileLayout.src, nil, tileDst, 0)
		}, "use DecodeTileInto"},
		{"whole-image destination for a tile", func() error {
			return h.DecodeTileInto(0, l.src, nil, 0, 0, dst, 0)
		}, "use DecodeInto"},
		{"wrong tile", func() error {
			return h.DecodeTileInto(0, tileLayout.src, nil, 0, 0, tileDst, 0)
		}, "bound for tile 1, not tile 0"},
		{"reduced resolution", func() error {
			return h.DecodeTileInto(0, tileLayout.src, nil, 1, 2, tileDst, 0)
		}, "ResolutionComponentDim"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.call()
			if err == nil {
				t.Fatal("the guard did not fire")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error = %q, want it to mention %q", err, c.want)
			}
		})
	}
}

// TestParseRejectsEmptyBuffer checks that Parse fails on an empty slice before
// it can dereference data[0]. It runs with no library present because the
// check precedes the symbol lookup.
func TestParseRejectsEmptyBuffer(t *testing.T) {
	err := Handle(0).Parse(nil, Bitstream(1))
	if err == nil {
		t.Fatal("Parse accepted an empty buffer")
	}
	if !strings.Contains(err.Error(), "empty bitstream buffer") {
		t.Errorf("error = %q, want it to mention the empty buffer", err)
	}
	if err := Handle(0).Parse([]byte{0xff, 0x4f}, 0); err == nil {
		t.Error("Parse accepted a zero Bitstream handle")
	}
}
