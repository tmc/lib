//go:build nvdec_physical

package nvdec_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvdec"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"unsafe"
)

// TestPhysicalDecode compares three 256x256 frames with CPU-decoded NV12 references.
func TestPhysicalDecode(t *testing.T) {
	for _, tc := range []struct {
		name  string
		codec nvdec.Codec
	}{
		{"h264", nvdec.CodecH264}, {"hevc", nvdec.CodecHevc}, {"ivf", nvdec.CodecAV1},
	} {
		t.Run(tc.name, func(t *testing.T) { testPhysicalDecode(t, tc.name, tc.codec) })
	}
}

func testPhysicalDecode(t *testing.T, name string, codec nvdec.Codec) {
	dir := os.Getenv("NVDEC_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("NVDEC_FIXTURE_DIR is required with nvdec_physical")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	check(driver.Init(0))
	dev, e := driver.NewDeviceGet(0)
	check(e)
	ctx, e := driver.NewContext(nil, 0, dev)
	check(e)
	defer func() {
		if err := ctx.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	data, e := os.ReadFile(filepath.Join(dir, "three."+name))
	check(e)
	want, e := os.ReadFile(filepath.Join(dir, "three."+name+".nv12"))
	check(e)
	if len(data) == 0 || len(want) != 3*256*256*3/2 {
		t.Fatal("invalid fixture or reference size")
	}
	caps := nvdec.DecodeCaps{ECodecType: codec, EChromaFormat: nvdec.ChromaFormat420}
	cr, ce := nvdec.GetDecoderCaps(&caps)
	t.Logf("caps result=%v error=%v min=%dx%d max=%dx%d", cr, ce, caps.NMinWidth, caps.NMinHeight, caps.NMaxWidth, caps.NMaxHeight)
	var decoder nvdec.Decoder
	var failure error
	var got []byte
	frames := 0
	operation := ""
	result := func(r nvdec.Result, e error) bool {
		if e != nil {
			if failure == nil {
				failure = fmt.Errorf("%s: %w", operation, e)
			}
			return false
		}
		if r != 0 {
			if failure == nil {
				failure = fmt.Errorf("%s: NVDEC result %v", operation, r)
			}
			return false
		}
		return true
	}
	seq := purego.NewCallback(func(_ uintptr, f *nvdec.VideoFormat) int32 {
		if f.CodedWidth != 256 || f.CodedHeight != 256 || f.BitDepthLumaMinus8 != 0 {
			failure = fmt.Errorf("unexpected format %+v", f)
			return 0
		}
		t.Logf("sequence: coded=%dx%d surfaces=%d chroma=%v", f.CodedWidth, f.CodedHeight, f.MinNumDecodeSurfaces, f.ChromaFormat)
		ci := nvdec.DecodeCreateInfo{UlWidth: 256, UlHeight: 256, UlNumDecodeSurfaces: uint64(f.MinNumDecodeSurfaces), CodecType: f.Codec, ChromaFormat: f.ChromaFormat, OutputFormat: nvdec.SurfaceFormatNV12, UlTargetWidth: 256, UlTargetHeight: 256, UlNumOutputSurfaces: 2}
		operation = "CreateDecoder"
		if !result(nvdec.CreateDecoder(&decoder, &ci)) {
			return 0
		}
		return int32(f.MinNumDecodeSurfaces)
	})
	decode := purego.NewCallback(func(_ uintptr, p *nvdec.PicParams) int32 {
		operation = "DecodePicture"
		if !result(nvdec.DecodePicture(decoder, p)) {
			return 0
		}
		return 1
	})
	display := purego.NewCallback(func(_ uintptr, p *nvdec.ParserDispInfo) int32 {
		if p == nil {
			return 1
		}
		var ptr uint64
		var pitch uint32
		pp := nvdec.ProcParams{ProgressiveFrame: p.ProgressiveFrame, TopFieldFirst: p.TopFieldFirst, UnpairedField: 1}
		operation = "MapVideoFrame64"
		if !result(nvdec.MapVideoFrame64(decoder, p.PictureIndex, &ptr, &pitch, &pp)) {
			return 0
		}
		defer func() { result(nvdec.UnmapVideoFrame64(decoder, ptr)) }()
		for y := 0; y < 256*3/2; y++ {
			row := make([]byte, 256)
			if e := driver.DevicePtr(ptr+uint64(y)*uint64(pitch)).MemcpyDtoH(unsafe.Pointer(&row[0]), 256); e != nil {
				failure = e
				return 0
			}
			got = append(got, row...)
		}
		frames++
		return 1
	})
	params := nvdec.ParserParams{CodecType: codec, UlMaxNumDecodeSurfaces: 1, PfnSequenceCallback: seq, PfnDecodePicture: decode, PfnDisplayPicture: display}
	var parser nvdec.Parser
	operation = "CreateVideoParser"
	if !result(nvdec.CreateVideoParser(&parser, &params)) {
		t.Fatal(failure)
	}
	defer func() {
		if !result(nvdec.DestroyVideoParser(parser)) {
			t.Error(failure)
		}
		if decoder != 0 {
			if !result(nvdec.DestroyDecoder(decoder)) {
				t.Error(failure)
			}
		}
	}()
	packets := [][]byte{data}
	if codec == nvdec.CodecAV1 {
		if len(data) < 32 || string(data[:4]) != "DKIF" || string(data[8:12]) != "AV01" {
			t.Fatal("invalid AV1 IVF header")
		}
		packets = nil
		for rest := data[32:]; len(rest) > 0; {
			if len(rest) < 12 {
				t.Fatal("truncated IVF frame header")
			}
			n := uint64(binary.LittleEndian.Uint32(rest[:4]))
			rest = rest[12:]
			if n == 0 || n > uint64(len(rest)) {
				t.Fatal("invalid IVF frame size")
			}
			packets = append(packets, rest[:int(n)])
			rest = rest[int(n):]
		}
	}
	for _, payload := range packets {
		packet := nvdec.SourceDataPacket{PayloadSize: uint64(len(payload)), Payload: uintptr(unsafe.Pointer(&payload[0]))}
		operation = "ParseVideoData"
		if !result(nvdec.ParseVideoData(parser, &packet)) {
			t.Fatal(failure)
		}
	}
	runtime.KeepAlive(data)
	operation = "ParseVideoData"
	if !result(nvdec.ParseVideoData(parser, &nvdec.SourceDataPacket{Flags: 1})) {
		t.Fatal(failure)
	}
	if failure != nil {
		t.Fatal(failure)
	}
	if frames != 3 || !bytes.Equal(got, want) {
		t.Fatal(fmt.Sprintf("frames=%d bytes=%d reference=%d equal=%v", frames, len(got), len(want), bytes.Equal(got, want)))
	}
	t.Logf("%d NVDEC frames, %d NV12 bytes exactly matched CPU FFmpeg reference\n", frames, len(got))
}
