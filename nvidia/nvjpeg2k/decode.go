package nvjpeg2k

import (
	"errors"
	"fmt"
	"runtime"
)

// BytesPerSample returns the size in bytes of one sample of type t, or 0 if t
// is not one of the types this package knows.
func (t ImageType) BytesPerSample() int {
	switch t {
	case Uint8:
		return 1
	case Uint16, Int16, FP16:
		return 2
	case FP32:
		return 4
	}
	return 0
}

// bitsPerSample returns the number of value bits a destination of type t can
// hold. The header's rule for the integer types is that the destination width
// must be at least the SIZ-marker per-component precision.
func (t ImageType) bitsPerSample() int {
	switch t {
	case Uint8:
		return 8
	case Uint16, Int16:
		return 16
	}
	return 0
}

// ComponentLayout is what the bitstream says about one image component. Every
// field is read from the codestream, not derived from the image dimensions:
// components of one image differ in extent (a subsampled chroma plane is
// smaller than the luma plane), in precision, and in signedness.
type ComponentLayout struct {
	// Width and Height are the component's extent in samples. For a tile
	// layout they are the tile's extent for this component, which is smaller
	// than the image extent whenever the image is tiled.
	Width, Height uint32

	// Precision is the component bit depth from the SIZ marker, and Signed
	// reports whether its samples are signed.
	Precision uint8
	Signed    bool

	// NLT is the nonlinear transformation the component's NLT marker selects,
	// NltNone when there is none, and NLTBitDepth is the effective output bit
	// depth (BDnlt, falling back to Precision when no marker is present).
	// Both are zero when the loaded library does not export
	// nvjpeg2kStreamGetComponentNLTInfo; see [Layout.NLTKnown].
	NLT         NLTType
	NLTBitDepth uint8
}

// BufferRequirement is the minimum shape of one destination allocation.
//
// MinSizeInBytes is computed from MinPitchInBytes. A buffer whose pitch is
// larger than the minimum needs proportionally more bytes, which is why
// [Layout.Bind] recomputes the size from the pitch the caller actually
// supplies rather than comparing against this value alone.
type BufferRequirement struct {
	MinPitchInBytes int
	Rows            int
	MinSizeInBytes  int
}

// Layout is the destination description for one parsed bitstream, read from
// that bitstream's metadata. It is the only thing in this package that knows
// how large a decode output must be.
//
// A Layout is a snapshot. Parsing another bitstream into the same [Bitstream]
// handle invalidates it, and nothing in this package can detect that.
type Layout struct {
	// Info is the image header: extent, tile grid, and component count.
	Info ImageInfo

	// PixelType is the sample type the destination buffers hold, and Format
	// is how the components are arranged across them.
	PixelType ImageType
	Format    ImageFormat

	// Components has one entry per image component, in codestream order.
	Components []ComponentLayout

	// Tile reports whether this layout describes a single tile rather than
	// the whole image, and TileID names that tile when it does.
	Tile   bool
	TileID uint32

	// NLTKnown reports whether the per-component NLT fields were read. They
	// are not when the loaded library predates
	// nvjpeg2kStreamGetComponentNLTInfo.
	NLTKnown bool

	src Bitstream
}

// LayoutOf reads the metadata a whole-image decode destination must match:
// the image header and then, for every component, that component's own
// dimensions, precision, and signedness.
//
// pixelType and format describe the destination the caller intends to
// allocate. LayoutOf rejects combinations the header documents as invalid for
// this bitstream, so the error arrives before any device memory is allocated
// rather than as a corrupt image afterwards.
func LayoutOf(bs Bitstream, pixelType ImageType, format ImageFormat) (*Layout, error) {
	return layoutOf(bs, pixelType, format, false, 0)
}

// LayoutOfTile reads the metadata a single-tile decode destination must
// match. A tile's components are smaller than the image's; sizing a tile
// decode from [LayoutOf] overruns nothing the host can see and corrupts the
// output.
func LayoutOfTile(bs Bitstream, tileID uint32, pixelType ImageType, format ImageFormat) (*Layout, error) {
	return layoutOf(bs, pixelType, format, true, tileID)
}

func layoutOf(bs Bitstream, pixelType ImageType, format ImageFormat, tile bool, tileID uint32) (*Layout, error) {
	if bs == 0 {
		return nil, errors.New("nvjpeg2k: layout: zero Bitstream handle")
	}
	bps := pixelType.BytesPerSample()
	if bps == 0 {
		return nil, fmt.Errorf("nvjpeg2k: layout: unsupported pixel type %v", pixelType)
	}
	switch format {
	case FormatPlanar, FormatInterleaved:
	default:
		return nil, fmt.Errorf("nvjpeg2k: layout: unsupported image format %v", format)
	}
	info, err := bs.ImageInfo()
	if err != nil {
		return nil, fmt.Errorf("nvjpeg2k: layout: image info: %w", err)
	}
	if info.NumComponents == 0 {
		return nil, errors.New("nvjpeg2k: layout: bitstream reports zero components")
	}
	if tile {
		tiles := uint64(info.NumTilesX) * uint64(info.NumTilesY)
		if uint64(tileID) >= tiles {
			return nil, fmt.Errorf("nvjpeg2k: layout: tile %d out of range, image has %d tiles", tileID, tiles)
		}
	}
	l := &Layout{
		Info:       info,
		PixelType:  pixelType,
		Format:     format,
		Components: make([]ComponentLayout, info.NumComponents),
		Tile:       tile,
		TileID:     tileID,
		NLTKnown:   Has("nvjpeg2kStreamGetComponentNLTInfo"),
		src:        bs,
	}
	for i := range l.Components {
		ci, err := bs.ComponentInfo(uint32(i))
		if err != nil {
			return nil, fmt.Errorf("nvjpeg2k: layout: component %d info: %w", i, err)
		}
		c := ComponentLayout{
			Width:     ci.ComponentWidth,
			Height:    ci.ComponentHeight,
			Precision: ci.Precision,
			Signed:    ci.Sgn != 0,
		}
		if tile {
			w, h, err := bs.TileComponentDim(uint32(i), tileID)
			if err != nil {
				return nil, fmt.Errorf("nvjpeg2k: layout: tile %d component %d dimensions: %w", tileID, i, err)
			}
			c.Width, c.Height = w, h
		}
		if c.Width == 0 || c.Height == 0 {
			return nil, fmt.Errorf("nvjpeg2k: layout: component %d has zero extent %dx%d", i, c.Width, c.Height)
		}
		if l.NLTKnown {
			nlt, err := bs.ComponentNLTInfo(uint32(i))
			if err != nil {
				return nil, fmt.Errorf("nvjpeg2k: layout: component %d NLT info: %w", i, err)
			}
			c.NLT, c.NLTBitDepth = nlt.Type, nlt.BitDepth
		}
		l.Components[i] = c
	}
	if err := l.checkPixelType(); err != nil {
		return nil, err
	}
	if format == FormatInterleaved {
		first := l.Components[0]
		for i, c := range l.Components {
			if c.Width != first.Width || c.Height != first.Height {
				return nil, fmt.Errorf("nvjpeg2k: layout: interleaved output needs equal component dimensions, "+
					"component 0 is %dx%d and component %d is %dx%d",
					first.Width, first.Height, i, c.Width, c.Height)
			}
		}
	}
	return l, nil
}

// checkPixelType applies the destination-type rules the header states for
// this bitstream. The floating-point rules are checkable only when the
// library reports NLT information.
func (l *Layout) checkPixelType() error {
	switch l.PixelType {
	case Uint8, Uint16, Int16:
		bits := l.PixelType.bitsPerSample()
		for i, c := range l.Components {
			if int(c.Precision) > bits {
				return fmt.Errorf("nvjpeg2k: layout: component %d has precision %d, too wide for %v",
					i, c.Precision, l.PixelType)
			}
		}
		return nil
	case FP16, FP32:
		if !l.NLTKnown {
			return fmt.Errorf("nvjpeg2k: layout: %v requires an NLT type 3 bitstream, and the loaded "+
				"library does not export nvjpeg2kStreamGetComponentNLTInfo, so that cannot be checked", l.PixelType)
		}
		for i, c := range l.Components {
			if c.NLT != NltType3 {
				return fmt.Errorf("nvjpeg2k: layout: %v requires NLT type 3 on every component, "+
					"component %d has %v", l.PixelType, i, c.NLT)
			}
			switch {
			case l.PixelType == FP16 && c.NLTBitDepth != 16:
				return fmt.Errorf("nvjpeg2k: layout: FP16 requires BDnlt 16 on every component, "+
					"component %d has %d", i, c.NLTBitDepth)
			case l.PixelType == FP32 && c.NLTBitDepth != 16 && c.NLTBitDepth != 32:
				return fmt.Errorf("nvjpeg2k: layout: FP32 requires BDnlt 16 or 32 on every component, "+
					"component %d has %d", i, c.NLTBitDepth)
			}
		}
		return nil
	}
	return fmt.Errorf("nvjpeg2k: layout: unsupported pixel type %v", l.PixelType)
}

// Buffers returns the minimum shape of every destination allocation, in the
// order [Layout.Bind] expects them: one per component for [FormatPlanar], and
// exactly one for [FormatInterleaved].
func (l *Layout) Buffers() []BufferRequirement {
	bps := l.PixelType.BytesPerSample()
	if l.Format == FormatInterleaved {
		c := l.Components[0]
		pitch := int(c.Width) * len(l.Components) * bps
		return []BufferRequirement{{
			MinPitchInBytes: pitch,
			Rows:            int(c.Height),
			MinSizeInBytes:  pitch * int(c.Height),
		}}
	}
	reqs := make([]BufferRequirement, len(l.Components))
	for i, c := range l.Components {
		pitch := int(c.Width) * bps
		reqs[i] = BufferRequirement{
			MinPitchInBytes: pitch,
			Rows:            int(c.Height),
			MinSizeInBytes:  pitch * int(c.Height),
		}
	}
	return reqs
}

// DeviceBuffer is one caller-owned device allocation that a decode writes
// into. This package never allocates or frees device memory: Ptr is a device
// address obtained from the CUDA runtime or driver, and the caller keeps it
// alive until the decode's CUDA stream has completed.
type DeviceBuffer struct {
	// Ptr is the device address of the first sample.
	Ptr uintptr

	// PitchInBytes is the distance between the starts of two consecutive
	// rows, at least the layout's minimum pitch.
	PitchInBytes int

	// SizeInBytes is the size of the allocation at Ptr. [Layout.Bind] uses it
	// to reject a buffer that cannot hold PitchInBytes times the row count.
	SizeInBytes int
}

// Destination is a set of device buffers checked against a [Layout] and ready
// to be decoded into. It holds the host-side pointer and pitch arrays the
// library reads during a decode call, and keeps them reachable for as long as
// the Destination is.
//
// A Destination does not own the device memory it names.
type Destination struct {
	layout  *Layout
	buffers []DeviceBuffer
	ptrs    []uintptr
	pitches []uint
	image   Image
}

// Bind checks caller-owned device buffers against the layout and returns a
// [Destination] that [Handle.DecodeInto] will accept. The buffers must be in
// the order [Layout.Buffers] describes.
//
// Bind cannot tell whether Ptr is really a device address, whether it is still
// allocated, or whether SizeInBytes is truthful; it checks only what the
// caller reports against what the bitstream requires.
func (l *Layout) Bind(buffers ...DeviceBuffer) (*Destination, error) {
	reqs := l.Buffers()
	if len(buffers) != len(reqs) {
		return nil, fmt.Errorf("nvjpeg2k: bind: got %d buffers, layout needs %d", len(buffers), len(reqs))
	}
	d := &Destination{
		layout:  l,
		buffers: append([]DeviceBuffer(nil), buffers...),
		ptrs:    make([]uintptr, len(buffers)),
		pitches: make([]uint, len(buffers)),
	}
	for i, b := range buffers {
		req := reqs[i]
		if b.Ptr == 0 {
			return nil, fmt.Errorf("nvjpeg2k: bind: buffer %d has a nil device pointer", i)
		}
		if b.PitchInBytes < req.MinPitchInBytes {
			return nil, fmt.Errorf("nvjpeg2k: bind: buffer %d pitch %d is below the required %d bytes",
				i, b.PitchInBytes, req.MinPitchInBytes)
		}
		need := b.PitchInBytes * req.Rows
		if b.SizeInBytes < need {
			return nil, fmt.Errorf("nvjpeg2k: bind: buffer %d is %d bytes, needs %d for %d rows of pitch %d",
				i, b.SizeInBytes, need, req.Rows, b.PitchInBytes)
		}
		d.ptrs[i] = b.Ptr
		d.pitches[i] = uint(b.PitchInBytes)
	}
	d.image = Image{
		PixelData:     &d.ptrs[0],
		PitchInBytes:  &d.pitches[0],
		PixelType:     l.PixelType,
		NumComponents: uint32(len(l.Components)),
	}
	return d, nil
}

// Layout returns the layout the destination was bound against.
func (d *Destination) Layout() *Layout { return d.layout }

// Buffers returns a copy of the device buffers the destination names.
func (d *Destination) Buffers() []DeviceBuffer {
	return append([]DeviceBuffer(nil), d.buffers...)
}

// Params is a [DecodeParams] handle together with the settings applied
// through it. [Handle.DecodeInto] needs to know the output format to check a
// destination against, and the C API has no way to read a setting back, so
// this type records what was set.
//
// The zero Params is not usable; create one with [NewParams] and release it
// with [Params.Destroy].
type Params struct {
	handle DecodeParams
	format ImageFormat
	area   bool
	rgb    bool
}

// NewParams creates a decode parameters handle. The caller destroys it.
func NewParams() (*Params, error) {
	h, err := NewDecodeParams()
	if err != nil {
		return nil, err
	}
	return &Params{handle: h, format: FormatPlanar}, nil
}

// Handle returns the underlying [DecodeParams] for use with the raw decode
// calls. Settings applied through it are invisible to [Handle.DecodeInto].
func (p *Params) Handle() DecodeParams { return p.handle }

// Destroy releases the decode parameters handle. It is not idempotent and the
// handle is not finalized automatically.
func (p *Params) Destroy() error {
	if p == nil || p.handle == 0 {
		return nil
	}
	err := p.handle.Destroy()
	p.handle = 0
	return err
}

// SetOutputFormat selects planar or interleaved output.
func (p *Params) SetOutputFormat(format ImageFormat) error {
	if err := p.handle.SetOutputFormat(format); err != nil {
		return err
	}
	p.format = format
	return nil
}

// SetRGBOutput requests RGB output for chroma-subsampled sYCC streams. The
// header documents it as ignored for other color spaces.
func (p *Params) SetRGBOutput(rgb bool) error {
	v := int32(0)
	if rgb {
		v = 1
	}
	if err := p.handle.SetRGBOutput(v); err != nil {
		return err
	}
	p.rgb = rgb
	return nil
}

// SetDecodeArea restricts decoding to a rectangle within the tile of
// interest. A [Layout] describes the full component extent, so once an area
// is set [Handle.DecodeInto] refuses the parameters and the caller must size
// the buffers itself and use [Handle.DecodeImage] directly.
func (p *Params) SetDecodeArea(startX, endX, startY, endY uint32) error {
	if err := p.handle.SetDecodeArea(startX, endX, startY, endY); err != nil {
		return err
	}
	p.area = true
	return nil
}

// RGBOutput reports whether RGB output was requested.
func (p *Params) RGBOutput() bool { return p.rgb }

// OutputFormat reports the output format that was set, [FormatPlanar] if none
// was.
func (p *Params) OutputFormat() ImageFormat { return p.format }

// checkAgainst reports whether a destination built from l may be decoded
// under these parameters.
func (p *Params) checkAgainst(l *Layout) error {
	if p == nil {
		if l.Format != FormatPlanar {
			return fmt.Errorf("nvjpeg2k: decode: layout format is %v but no decode parameters were "+
				"supplied, and the library's default is %v", l.Format, FormatPlanar)
		}
		return nil
	}
	if p.handle == 0 {
		return errors.New("nvjpeg2k: decode: decode parameters have been destroyed")
	}
	if p.area {
		return errors.New("nvjpeg2k: decode: a decode area was set, so the layout no longer describes " +
			"the output extent; size the buffers yourself and call DecodeImage")
	}
	if p.format != l.Format {
		return fmt.Errorf("nvjpeg2k: decode: layout format is %v but the decode parameters ask for %v",
			l.Format, p.format)
	}
	return nil
}

// DecodeInto decodes a whole image into buffers that were checked against the
// bitstream's own metadata. It is the checked form of [Handle.Decode] and
// [Handle.DecodeImage]: it refuses a destination built from a different
// bitstream, a tile destination, and decode parameters that disagree with the
// destination's layout.
//
// params may be nil, which decodes with the library's defaults.
//
// The call is asynchronous with respect to the host: it returns once the work
// is submitted to stream. Every device buffer the destination names must stay
// allocated until stream has completed, and the decode state must not be
// reused for another image before then.
func (h Handle) DecodeInto(state DecodeState, bs Bitstream, params *Params, dst *Destination, stream Stream) error {
	if dst == nil {
		return errors.New("nvjpeg2k: DecodeInto: nil destination")
	}
	if dst.layout.src != bs {
		return errors.New("nvjpeg2k: DecodeInto: destination was bound to a different bitstream")
	}
	if dst.layout.Tile {
		return errors.New("nvjpeg2k: DecodeInto: destination describes one tile; use DecodeTileInto")
	}
	if err := params.checkAgainst(dst.layout); err != nil {
		return err
	}
	var err error
	if params == nil {
		err = Decode(h, state, bs, &dst.image, stream)
	} else {
		err = DecodeImage(h, state, bs, params.handle, &dst.image, stream)
	}
	runtime.KeepAlive(dst)
	return err
}

// DecodeTileInto decodes one tile into buffers that were checked against that
// tile's own component dimensions. It is the checked form of
// [Handle.DecodeTile].
//
// numResLevels is the number of resolution levels to decode; 0 decodes the
// tile at full resolution. A layout describes the full-resolution extent, so
// any other value is refused here: use [Handle.DecodeTile] and size the
// buffers from [Bitstream.ResolutionComponentDim].
//
// params may be nil. The asynchrony and buffer-lifetime rules of
// [Handle.DecodeInto] apply.
func (h Handle) DecodeTileInto(state DecodeState, bs Bitstream, params *Params, tileID, numResLevels uint32, dst *Destination, stream Stream) error {
	if dst == nil {
		return errors.New("nvjpeg2k: DecodeTileInto: nil destination")
	}
	if dst.layout.src != bs {
		return errors.New("nvjpeg2k: DecodeTileInto: destination was bound to a different bitstream")
	}
	if !dst.layout.Tile {
		return errors.New("nvjpeg2k: DecodeTileInto: destination describes the whole image; use DecodeInto")
	}
	if dst.layout.TileID != tileID {
		return fmt.Errorf("nvjpeg2k: DecodeTileInto: destination was bound for tile %d, not tile %d",
			dst.layout.TileID, tileID)
	}
	if numResLevels != 0 {
		return errors.New("nvjpeg2k: DecodeTileInto: a layout describes the full-resolution tile; " +
			"for a reduced-resolution decode size the buffers from ResolutionComponentDim and call DecodeTile")
	}
	if err := params.checkAgainst(dst.layout); err != nil {
		return err
	}
	var ph DecodeParams
	if params != nil {
		ph = params.handle
	}
	err := DecodeTile(h, state, bs, ph, tileID, numResLevels, &dst.image, stream)
	runtime.KeepAlive(dst)
	return err
}
