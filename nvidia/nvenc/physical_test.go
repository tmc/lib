//go:build nvenc_physical

package nvenc_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvenc"
)

// TestPhysicalEncode compares lossless H.264 output with an independent CPU decode.
// Selecting nvenc_physical requires a compatible GPU, driver, and FFmpeg.
func TestPhysicalEncode(t *testing.T) {
	if err := physicalEncode(t); err != nil {
		t.Fatal(err)
	}
}

func physicalEncode(t *testing.T) (err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	var version uint32
	status, err := nvenc.GetMaxSupportedVersion(&version)
	if err != nil {
		return err
	}
	if status != nvenc.Success {
		return fmt.Errorf("version query: %v", status)
	}
	t.Logf("driver max API=%#x pinned header=%d.%d\n", version, nvenc.APIMajorVersion, nvenc.APIMinorVersion)
	required := uint32(nvenc.APIMajorVersion<<4 | nvenc.APIMinorVersion)
	if version < required {
		return fmt.Errorf("driver supports API %#x, bindings require %#x", version, required)
	}
	list := nvenc.FunctionList{Version: nvenc.FunctionListVer}
	status, err = nvenc.CreateInstance(&list)
	if err != nil {
		return err
	}
	if status != nvenc.Success {
		return fmt.Errorf("create function table: %v", status)
	}
	if err := driver.Init(0); err != nil {
		return err
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		return err
	}
	ctx, err := driver.NewContext(nil, 0, dev)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, ctx.Destroy()) }()
	api, err := nvenc.Bind(&list)
	if err != nil {
		return err
	}
	params := nvenc.OpenEncodeSessionExParams{Version: nvenc.OpenEncodeSessionExParamsVer, DeviceType: nvenc.DeviceTypeCuda, Device: uintptr(ctx), ApiVersion: nvenc.APIVersion}
	var encoder unsafe.Pointer
	status = api.NvEncOpenEncodeSessionEx(&params, &encoder)
	t.Logf("session API=%#x status=%v nonzero=%v\n", params.ApiVersion, status, encoder != nil)
	if status != nvenc.Success {
		return fmt.Errorf("open session: %v", status)
	}
	if encoder == nil {
		return fmt.Errorf("open session returned nil")
	}
	defer func() { err = errors.Join(err, checkNVENC("destroy encoder", api.NvEncDestroyEncoder(encoder))) }()
	var count uint32
	status = api.NvEncGetEncodeGUIDCount(encoder, &count)
	t.Logf("codec count=%d status=%v\n", count, status)
	if status != nvenc.Success {
		return fmt.Errorf("codec count: %v", status)
	}
	if count == 0 || count > 1024 {
		return fmt.Errorf("no codecs")
	}
	guids := make([]nvenc.GUID, count)
	if st := api.NvEncGetEncodeGUIDs(encoder, &guids[0], count, &count); st != nvenc.Success {
		return fmt.Errorf("codec GUIDs: %v", st)
	}
	found := false
	for _, g := range guids {
		if g == nvenc.CodecH264GUID {
			found = true
		}
	}
	if !found {
		return fmt.Errorf("H264 GUID not supported")
	}
	var presets uint32
	if st := api.NvEncGetEncodePresetCount(encoder, nvenc.CodecH264GUID, &presets); st != nvenc.Success {
		return fmt.Errorf("preset count: %v", st)
	}
	t.Logf("H264 preset count=%d\n", presets)
	if presets == 0 || presets > 1024 {
		return fmt.Errorf("no H264 presets")
	}
	presetGUIDs := make([]nvenc.GUID, presets)
	if st := api.NvEncGetEncodePresetGUIDs(encoder, nvenc.CodecH264GUID, &presetGUIDs[0], presets, &presets); st != nvenc.Success {
		return fmt.Errorf("preset GUIDs: %v", st)
	}
	found = false
	for _, g := range presetGUIDs {
		if g == nvenc.PresetP1GUID {
			found = true
		}
	}
	if !found {
		return fmt.Errorf("P1 preset unsupported")
	}
	preset := nvenc.PresetConfig{Version: nvenc.PresetConfigVer, PresetCfg: nvenc.Config{Version: nvenc.ConfigVer}}
	if st := api.NvEncGetEncodePresetConfigEx(encoder, nvenc.CodecH264GUID, nvenc.PresetP1GUID, nvenc.TuningInfoLossless, &preset); st != nvenc.Success {
		return fmt.Errorf("lossless preset: %v", st)
	}
	t.Logf("lossless preset received, config version=%#x\n", preset.PresetCfg.Version)
	return encodeFrames(t, api, encoder, &preset.PresetCfg)
}
func checkNVENC(action string, status nvenc.Status) error {
	if status != nvenc.Success {
		return fmt.Errorf("%s: %v", action, status)
	}
	return nil
}

func encodeFrames(t *testing.T, api *nvenc.API, encoder unsafe.Pointer, config *nvenc.Config) (err error) {
	const width, height, frames = 256, 256, 3
	caps := nvenc.CapsParam{Version: nvenc.CapsParamVer, CapsToQuery: nvenc.CapsSupportLosslessEncode}
	var lossless int32
	if err := checkNVENC("lossless capability", api.NvEncGetEncodeCaps(encoder, nvenc.CodecH264GUID, &caps, &lossless)); err != nil {
		return err
	}
	if lossless != 1 {
		return fmt.Errorf("lossless capability = %d", lossless)
	}
	var formatCount uint32
	if err := checkNVENC("input format count", api.NvEncGetInputFormatCount(encoder, nvenc.CodecH264GUID, &formatCount)); err != nil {
		return err
	}
	if formatCount == 0 || formatCount > 1024 {
		return fmt.Errorf("invalid format count %d", formatCount)
	}
	formats := make([]nvenc.BufferFormat, formatCount)
	if err := checkNVENC("input formats", api.NvEncGetInputFormats(encoder, nvenc.CodecH264GUID, &formats[0], formatCount, &formatCount)); err != nil {
		return err
	}
	foundNV12 := false
	for _, format := range formats {
		if format == nvenc.BufferFormatNV12 {
			foundNV12 = true
		}
	}
	if !foundNV12 {
		return fmt.Errorf("NV12 input unsupported")
	}
	config.GopLength = 1
	config.FrameIntervalP = 1
	var pin runtime.Pinner
	pin.Pin(config)
	defer pin.Unpin()
	params := nvenc.InitializeParams{Version: nvenc.InitializeParamsVer, EncodeGUID: nvenc.CodecH264GUID, PresetGUID: nvenc.PresetP1GUID,
		EncodeWidth: width, EncodeHeight: height, DarWidth: width, DarHeight: height, FrameRateNum: 30, FrameRateDen: 1, EnablePTD: 1,
		EncodeConfig: uintptr(unsafe.Pointer(config)), MaxEncodeWidth: width, MaxEncodeHeight: height, TuningInfo: nvenc.TuningInfoLossless, BufferFormat: nvenc.BufferFormatNV12}
	invalid := params
	invalid.Version = 0
	if status := api.NvEncInitializeEncoder(encoder, &invalid); status != nvenc.ErrInvalidVersion {
		return fmt.Errorf("invalid initialize version: got %v, want %v", status, nvenc.ErrInvalidVersion)
	}
	t.Log("invalid initialize version rejected; retrying valid parameters")
	if err := checkNVENC("initialize encoder", api.NvEncInitializeEncoder(encoder, &params)); err != nil {
		return err
	}
	var inputs [4]nvenc.CreateInputBuffer
	for i := range inputs {
		input := &inputs[i]
		*input = nvenc.CreateInputBuffer{Version: nvenc.CreateInputBufferVer, Width: width, Height: height, BufferFmt: nvenc.BufferFormatNV12}
		if err := checkNVENC("create input", api.NvEncCreateInputBuffer(encoder, input)); err != nil {
			return err
		}
		if input.InputBuffer == 0 {
			return fmt.Errorf("created nil input")
		}
		defer func() {
			err = errors.Join(err, checkNVENC("destroy input", api.NvEncDestroyInputBuffer(encoder, input.InputBuffer)))
		}()
	}
	output := nvenc.CreateBitstreamBuffer{Version: nvenc.CreateBitstreamBufferVer}
	if err := checkNVENC("create bitstream", api.NvEncCreateBitstreamBuffer(encoder, &output)); err != nil {
		return err
	}
	if output.BitstreamBuffer == 0 {
		return fmt.Errorf("created nil output")
	}
	defer func() {
		err = errors.Join(err, checkNVENC("destroy bitstream", api.NvEncDestroyBitstreamBuffer(encoder, output.BitstreamBuffer)))
	}()
	var expected, stream []byte
	for frame := range frames {
		data := make([]byte, width*height*3/2)
		for y := range height {
			for x := range width {
				data[y*width+x] = byte(16 + (x*3+y*5+frame*19)%220)
			}
		}
		for i := width * height; i < len(data); i += 2 {
			data[i] = byte(40 + (i/2+frame*11)%170)
			data[i+1] = byte(45 + (i/7+frame*23)%160)
		}
		expected = append(expected, data...)
		lock := nvenc.LockInputBuffer{Version: nvenc.LockInputBufferVer, InputBuffer: inputs[frame].InputBuffer}
		if err := checkNVENC("lock input", api.NvEncLockInputBuffer(encoder, &lock)); err != nil {
			return err
		}
		if lock.BufferDataPtr == 0 || lock.Pitch < width || lock.Pitch > 1<<20 {
			return errors.Join(fmt.Errorf("invalid input layout ptr=%#x pitch=%d", lock.BufferDataPtr, lock.Pitch), checkNVENC("unlock input", api.NvEncUnlockInputBuffer(encoder, lock.InputBuffer)))
		}
		dst := nativeBytes(lock.BufferDataPtr, int(lock.Pitch)*height*3/2)
		for row := range height * 3 / 2 {
			copy(dst[row*int(lock.Pitch):row*int(lock.Pitch)+width], data[row*width:(row+1)*width])
		}
		if err := checkNVENC("unlock input", api.NvEncUnlockInputBuffer(encoder, lock.InputBuffer)); err != nil {
			return err
		}
		pic := nvenc.PicParams{Version: nvenc.PicParamsVer, InputWidth: width, InputHeight: height, InputPitch: lock.Pitch,
			EncodePicFlags: uint32(nvenc.PicFlagForceidr | nvenc.PicFlagOutputSpspps), FrameIdx: uint32(frame), InputTimeStamp: uint64(frame), InputDuration: 1,
			InputBuffer: lock.InputBuffer, OutputBitstream: output.BitstreamBuffer, BufferFmt: nvenc.BufferFormatNV12, PictureStruct: nvenc.PicStructFrame}
		if err := checkNVENC("encode picture", api.NvEncEncodePicture(encoder, &pic)); err != nil {
			return err
		}
		bits := nvenc.LockBitstream{Version: nvenc.LockBitstreamVer, OutputBitstream: output.BitstreamBuffer}
		if err := checkNVENC("lock bitstream", api.NvEncLockBitstream(encoder, &bits)); err != nil {
			return err
		}
		if bits.BitstreamBufferPtr == 0 || bits.BitstreamSizeInBytes == 0 || bits.BitstreamSizeInBytes > 16<<20 {
			return errors.Join(fmt.Errorf("invalid bitstream size %d", bits.BitstreamSizeInBytes), checkNVENC("unlock bitstream", api.NvEncUnlockBitstream(encoder, bits.OutputBitstream)))
		}
		stream = append(stream, nativeBytes(bits.BitstreamBufferPtr, int(bits.BitstreamSizeInBytes))...)
		t.Logf("frame=%d input pitch=%d bitstream bytes=%d\n", frame, lock.Pitch, bits.BitstreamSizeInBytes)
		if err := checkNVENC("unlock bitstream", api.NvEncUnlockBitstream(encoder, bits.OutputBitstream)); err != nil {
			return err
		}
	}
	eos := nvenc.PicParams{Version: nvenc.PicParamsVer, EncodePicFlags: uint32(nvenc.PicFlagEos)}
	if err := checkNVENC("flush EOS", api.NvEncEncodePicture(encoder, &eos)); err != nil {
		return err
	}
	dir := os.Getenv("NVENC_ARTIFACT_DIR")
	if dir == "" {
		dir = t.TempDir()
	}
	outputPath := filepath.Join(dir, "output.h264")
	if err := os.WriteFile(outputPath, stream, 0644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "input.nv12"), expected, 0644); err != nil {
		return err
	}
	decodeContext, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(decodeContext, "ffmpeg", "-v", "error", "-hwaccel", "none", "-threads", "1", "-i", outputPath, "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1")
	cmd.WaitDelay = time.Second
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	decoded, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("CPU decode: %w: %s", err, stderr.String())
	}
	if !bytes.Equal(decoded, expected) {
		return fmt.Errorf("lossless CPU decode differs: got %d bytes sha256 %x, want %d bytes sha256 %x", len(decoded), sha256.Sum256(decoded), len(expected), sha256.Sum256(expected))
	}
	t.Logf("CPU decode matched %d frames, %d NV12 bytes sha256 %x\n", frames, len(expected), sha256.Sum256(expected))
	return nil
}

// nativeBytes views memory owned by NVENC while its buffer is locked. The ABI
// stores these native pointers in uintptr fields; the address never names a Go
// allocation. Callers validate the pointer and extent before making this view.
func nativeBytes(address uintptr, size int) []byte {
	p := *(*unsafe.Pointer)(unsafe.Pointer(&address))
	return unsafe.Slice((*byte)(p), size)
}
