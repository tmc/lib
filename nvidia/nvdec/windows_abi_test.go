//go:build windows && amd64

package nvdec

import (
	"os"
	"runtime"
	"testing"
	"unsafe"
)

// TestWindowsABIFixture calls a C DLL compiled from the pinned codec headers.
// It checks the loader and field marshaling, without a CUDA driver or GPU.
func TestWindowsABIFixture(t *testing.T) {
	path := os.Getenv("NVDEC_ABI_FIXTURE_DLL")
	if path == "" {
		t.Skip("set NVDEC_ABI_FIXTURE_DLL to the compiled codegen windows_fixture.c DLL")
	}
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Close(); err != nil {
			t.Error(err)
		}
	}()
	caps := DecodeCaps{ECodecType: CodecH264}
	if result, err := GetDecoderCaps(&caps); result != 0 || err != nil {
		t.Fatalf("caps: %v, %v", result, err)
	}
	if caps.BIsSupported != 1 || caps.NMaxWidth != 4096 {
		t.Fatalf("caps fields: %+v", caps)
	}
	ci := DecodeCreateInfo{UlWidth: 321, UlHeight: 123, UlNumDecodeSurfaces: 5, UlTargetWidth: 320, UlTargetHeight: 120, UlNumOutputSurfaces: 2}
	var decoder Decoder
	if result, err := CreateDecoder(&decoder, &ci); result != 0 || err != nil {
		t.Fatalf("create: %v, %v", result, err)
	}
	if decoder != Decoder(0x1234) {
		t.Fatalf("decoder = %#x", decoder)
	}
	// A wrong field must reach the C fixture and produce its failure result.
	ci.UlTargetHeight++
	if result, err := CreateDecoder(&decoder, &ci); result != 1 || err != nil {
		t.Fatalf("invalid create: %v, %v", result, err)
	}
	data := []byte("xyz")
	packet := SourceDataPacket{Flags: 2, PayloadSize: 3, Payload: uintptr(unsafe.Pointer(&data[0])), Timestamp: 0x123456789}
	result, err := ParseVideoData(Parser(0x5678), &packet)
	runtime.KeepAlive(data)
	if result != 0 || err != nil {
		t.Fatalf("packet: %v, %v", result, err)
	}
	t.Log("Windows DLL loader, output fields, LLP64 decoder/packet fields and invalid-field control passed; fixture only")
}
