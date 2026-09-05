//go:build nvdec_physical

package nvdec_test

import (
	"bytes"
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

// TestPhysicalH264 compares three 128x96 frames with a CPU-decoded NV12 reference.
func TestPhysicalH264(t *testing.T) {
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
	data, e := os.ReadFile(filepath.Join(dir, "three.h264"))
	check(e)
	want, e := os.ReadFile(filepath.Join(dir, "three.nv12"))
	check(e)
	if len(data) == 0 || len(want) != 3*128*96*3/2 {
		t.Fatal("invalid fixture or reference size")
	}
	var decoder nvdec.Decoder
	var failure error
	var got []byte
	frames := 0
	result := func(r nvdec.Result, e error) bool {
		if e != nil {
			failure = e
			return false
		}
		if r != 0 {
			failure = fmt.Errorf("NVDEC result %v", r)
			return false
		}
		return true
	}
	seq := purego.NewCallback(func(_ uintptr, f *nvdec.VideoFormat) int32 {
		if f.CodedWidth != 128 || f.CodedHeight != 96 || f.BitDepthLumaMinus8 != 0 {
			failure = fmt.Errorf("unexpected format %+v", f)
			return 0
		}
		ci := nvdec.DecodeCreateInfo{UlWidth: 128, UlHeight: 96, UlNumDecodeSurfaces: uint64(f.MinNumDecodeSurfaces), CodecType: f.Codec, ChromaFormat: f.ChromaFormat, OutputFormat: nvdec.SurfaceFormatNV12, UlTargetWidth: 128, UlTargetHeight: 96, UlNumOutputSurfaces: 2}
		if !result(nvdec.CreateDecoder(&decoder, &ci)) {
			return 0
		}
		return int32(f.MinNumDecodeSurfaces)
	})
	decode := purego.NewCallback(func(_ uintptr, p *nvdec.PicParams) int32 {
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
		if !result(nvdec.MapVideoFrame64(decoder, p.PictureIndex, &ptr, &pitch, &pp)) {
			return 0
		}
		defer func() { result(nvdec.UnmapVideoFrame64(decoder, ptr)) }()
		for y := 0; y < 96*3/2; y++ {
			row := make([]byte, 128)
			if e := driver.DevicePtr(ptr+uint64(y)*uint64(pitch)).MemcpyDtoH(unsafe.Pointer(&row[0]), 128); e != nil {
				failure = e
				return 0
			}
			got = append(got, row...)
		}
		frames++
		return 1
	})
	params := nvdec.ParserParams{CodecType: nvdec.CodecH264, UlMaxNumDecodeSurfaces: 1, PfnSequenceCallback: seq, PfnDecodePicture: decode, PfnDisplayPicture: display}
	var parser nvdec.Parser
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
	packet := nvdec.SourceDataPacket{PayloadSize: uint64(len(data)), Payload: uintptr(unsafe.Pointer(&data[0]))}
	if !result(nvdec.ParseVideoData(parser, &packet)) {
		t.Fatal(failure)
	}
	runtime.KeepAlive(data)
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
