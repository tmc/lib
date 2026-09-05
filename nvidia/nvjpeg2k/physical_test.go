//go:build nvjpeg2k_physical

// This file is the physical consumer test: it decodes real bitstreams on a
// real GPU and compares the result against an independent decoder. It is
// behind a build tag because it cannot be made to pass by skipping, and a
// skipped hardware test is not evidence that the hardware path works.
//
// It has never been run. It was written on a machine with no GPU, no CUDA
// runtime, and no libnvjpeg2k, and everything below is unverified.
//
// # What it needs
//
//   - An NVIDIA GPU. The vendor documents nvJPEG2000 as requiring a GPU of
//     compute capability 6.0 or newer; that requirement is not verified here.
//   - An NVIDIA driver and CUDA runtime new enough for the loaded
//     libnvjpeg2k. The pinned build is the CUDA 13 variant of 0.11.0.51.
//   - libnvjpeg2k.so.0 on the loader path, or NVJPEG2K_LIB_PATH pointing at
//     it. This package does not vendor it.
//   - A CUDA memory allocator. This package allocates no device memory, so
//     the test needs one; see the cudaAlloc stub at the bottom of this file,
//     which must be replaced with a real binding before the test can run.
//   - The fixtures named below, which are not committed. See
//     testdata/README.md for what each one has to be and where to get it.
//   - An independent JPEG 2000 decoder to compare against: OpenJPEG's
//     opj_decompress, or Pillow with openjpeg support. Point
//     NVJPEG2K_REFERENCE_DIR at a directory holding the reference decode of
//     each fixture as a raw planar dump.
//
// # How to run it
//
//	NVJPEG2K_FIXTURE_DIR=/path/to/fixtures \
//	NVJPEG2K_REFERENCE_DIR=/path/to/references \
//	go test -tags nvjpeg2k_physical ./nvjpeg2k/...
//
// Every test below fails, loudly, when a prerequisite is missing. None of them
// skips.

package nvjpeg2k

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// fixture names the bitstreams the physical tests decode. They are described
// in testdata/README.md and are not committed: binary fixtures do not belong
// in this repository, and several of the useful ones are third-party images
// with their own licenses.
type fixture struct {
	// file is the fixture's name inside NVJPEG2K_FIXTURE_DIR.
	file string

	// lossless says the codestream is reversible, so the decode must match
	// the reference decoder bit for bit. Only a lossless fixture supports an
	// exact comparison; a lossy one only supports a tolerance, and a
	// tolerance hides the class of bug this test exists to find.
	lossless bool

	pixelType ImageType
	format    ImageFormat
}

var fixtures = []fixture{
	// The exact-comparison fixture. Reversible 5/3, no subsampling, so every
	// decoded sample must equal the reference decoder's sample.
	{file: "lossless_rgb_8bit.j2k", lossless: true, pixelType: Uint8, format: FormatPlanar},

	// The per-component trap: 4:2:0 chroma planes are half the width and half
	// the height of the luma plane. A destination sized from the image
	// dimensions decodes without an error and produces garbage, which is why
	// this fixture is here and why the sizes come from Layout.Buffers.
	{file: "lossless_ycc420_8bit.jp2", lossless: true, pixelType: Uint8, format: FormatPlanar},

	// Precision above 8 bits, to catch a destination type chosen by image
	// dimensions rather than by component precision.
	{file: "lossless_gray_12bit.j2k", lossless: true, pixelType: Uint16, format: FormatPlanar},

	// A tiled image, decoded a tile at a time through DecodeTileInto.
	{file: "lossless_rgb_8bit_tiled.j2k", lossless: true, pixelType: Uint8, format: FormatPlanar},
}

func fixtureDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("NVJPEG2K_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("NVJPEG2K_FIXTURE_DIR is not set; see testdata/README.md for the fixtures this test needs")
	}
	return dir
}

func referenceDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("NVJPEG2K_REFERENCE_DIR")
	if dir == "" {
		t.Fatal("NVJPEG2K_REFERENCE_DIR is not set; the comparison decoder's output is what makes this test evidence")
	}
	return dir
}

// requireLibrary fails, rather than skipping, when libnvjpeg2k cannot be
// loaded. Running this file at all is a statement that the library is
// expected to be there.
func requireLibrary(t *testing.T) {
	t.Helper()
	if err := Load(""); err != nil {
		var libErr *LibraryError
		if errors.As(err, &libErr) {
			t.Fatalf("libnvjpeg2k did not load: %v", libErr)
		}
		t.Fatalf("libnvjpeg2k did not load: %v", err)
	}
}

// TestPhysicalDecodeMatchesReference decodes each fixture and compares the
// result against an independent decoder's output. For the lossless fixtures
// the comparison is exact.
func TestPhysicalDecodeMatchesReference(t *testing.T) {
	requireLibrary(t)
	fdir, rdir := fixtureDir(t), referenceDir(t)

	handle, err := NewHandle()
	if err != nil {
		t.Fatalf("NewHandle: %v", err)
	}
	defer handle.Destroy()

	for _, f := range fixtures {
		t.Run(f.file, func(t *testing.T) {
			if !f.lossless {
				t.Fatalf("%s is lossy; this test only makes an exact claim, so it needs a lossless fixture", f.file)
			}
			data, err := os.ReadFile(filepath.Join(fdir, f.file))
			if err != nil {
				t.Fatalf("read fixture: %v", err)
			}
			want, err := os.ReadFile(filepath.Join(rdir, f.file+".raw"))
			if err != nil {
				t.Fatalf("read reference decode: %v", err)
			}

			state, err := NewDecodeState(handle)
			if err != nil {
				t.Fatalf("NewDecodeState: %v", err)
			}
			defer state.Destroy()

			bs, err := NewBitstream()
			if err != nil {
				t.Fatalf("NewBitstream: %v", err)
			}
			defer bs.Destroy()

			if err := handle.Parse(data, bs); err != nil {
				t.Fatalf("Parse: %v", err)
			}
			layout, err := LayoutOf(bs, f.pixelType, f.format)
			if err != nil {
				t.Fatalf("LayoutOf: %v", err)
			}

			var buffers []DeviceBuffer
			for _, req := range layout.Buffers() {
				b, free, err := cudaAlloc(req.MinSizeInBytes)
				if err != nil {
					t.Fatalf("device allocation: %v", err)
				}
				defer free()
				buffers = append(buffers, DeviceBuffer{
					Ptr:          b,
					PitchInBytes: req.MinPitchInBytes,
					SizeInBytes:  req.MinSizeInBytes,
				})
			}
			dst, err := layout.Bind(buffers...)
			if err != nil {
				t.Fatalf("Bind: %v", err)
			}
			if err := handle.DecodeInto(state, bs, nil, dst, 0); err != nil {
				t.Fatalf("DecodeInto: %v", err)
			}
			if err := cudaSynchronize(); err != nil {
				t.Fatalf("synchronize: %v", err)
			}

			var got []byte
			for i, b := range buffers {
				plane, err := cudaCopyToHost(b.Ptr, b.SizeInBytes)
				if err != nil {
					t.Fatalf("copy component %d back: %v", i, err)
				}
				got = append(got, plane...)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("decode of %s differs from the reference decoder: %s", f.file, firstDifference(got, want))
			}
		})
	}
}

// TestPhysicalMalformedInputCleansUp feeds truncated and corrupted bitstreams
// to the parser and the decoder and checks that each one produces an error and
// leaves the handles usable, rather than aborting the process or leaking the
// decode state. A crash here is the failure mode that matters: this is a
// parser fed untrusted input.
func TestPhysicalMalformedInputCleansUp(t *testing.T) {
	requireLibrary(t)
	fdir := fixtureDir(t)

	good, err := os.ReadFile(filepath.Join(fdir, fixtures[0].file))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	handle, err := NewHandle()
	if err != nil {
		t.Fatalf("NewHandle: %v", err)
	}
	defer handle.Destroy()

	inputs := map[string][]byte{
		"truncated header":     good[:min(len(good), 16)],
		"truncated codestream": good[:len(good)/2],
		"empty after marker":   {0xff, 0x4f, 0xff, 0x51},
		"not jpeg2000":         []byte("this is not a codestream"),
		"corrupted body":       flipByte(good, len(good)*3/4),
	}
	for name, data := range inputs {
		t.Run(name, func(t *testing.T) {
			bs, err := NewBitstream()
			if err != nil {
				t.Fatalf("NewBitstream: %v", err)
			}
			defer bs.Destroy()

			// A parse error is the expected outcome for most of these, but
			// not for all: a corrupted body may parse and fail at decode.
			// Either is acceptable; a success on both, or a crash, is not.
			perr := handle.Parse(data, bs)
			if perr != nil {
				return
			}
			layout, lerr := LayoutOf(bs, Uint8, FormatPlanar)
			if lerr != nil {
				return
			}
			state, err := NewDecodeState(handle)
			if err != nil {
				t.Fatalf("NewDecodeState: %v", err)
			}
			defer state.Destroy()

			var buffers []DeviceBuffer
			for _, req := range layout.Buffers() {
				b, free, err := cudaAlloc(req.MinSizeInBytes)
				if err != nil {
					t.Fatalf("device allocation: %v", err)
				}
				defer free()
				buffers = append(buffers, DeviceBuffer{Ptr: b, PitchInBytes: req.MinPitchInBytes, SizeInBytes: req.MinSizeInBytes})
			}
			dst, err := layout.Bind(buffers...)
			if err != nil {
				t.Fatalf("Bind: %v", err)
			}
			derr := handle.DecodeInto(state, bs, nil, dst, 0)
			serr := cudaSynchronize()
			if derr == nil && serr == nil {
				t.Errorf("malformed input %q parsed and decoded without error", name)
			}
		})
	}

	// The handle must still work after every failure above.
	bs, err := NewBitstream()
	if err != nil {
		t.Fatalf("NewBitstream after malformed input: %v", err)
	}
	defer bs.Destroy()
	if err := handle.Parse(good, bs); err != nil {
		t.Fatalf("the library handle did not survive malformed input: %v", err)
	}
}

// TestPhysicalRuntimeVersionMatchesHeader compares the loaded library's
// version against the pinned header's. A mismatch does not fail the test on
// its own -- the ABI is stable within a major version -- but a major-version
// difference means these bindings describe a different library.
func TestPhysicalRuntimeVersionMatchesHeader(t *testing.T) {
	requireLibrary(t)
	major, err := GetProperty(MajorVersion)
	if err != nil {
		t.Fatalf("GetProperty(MajorVersion): %v", err)
	}
	minor, err := GetProperty(MinorVersion)
	if err != nil {
		t.Fatalf("GetProperty(MinorVersion): %v", err)
	}
	patch, err := GetProperty(PatchLevel)
	if err != nil {
		t.Fatalf("GetProperty(PatchLevel): %v", err)
	}
	t.Logf("loaded libnvjpeg2k %d.%d.%d from %s; bindings generated from header %d.%d.%d.%d",
		major, minor, patch, Path(),
		HeaderVersionMajor, HeaderVersionMinor, HeaderVersionPatch, HeaderVersionBuild)
	if int(major) != HeaderVersionMajor {
		t.Errorf("loaded major version %d, bindings were generated from %d", major, HeaderVersionMajor)
	}
}

func firstDifference(got, want []byte) string {
	if len(got) != len(want) {
		return fmt.Sprintf("length %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Sprintf("first difference at byte %d: %#02x, want %#02x", i, got[i], want[i])
		}
	}
	return "no difference"
}

func flipByte(b []byte, i int) []byte {
	out := append([]byte(nil), b...)
	if i < len(out) {
		out[i] ^= 0xff
	}
	return out
}

// cudaAlloc, cudaCopyToHost and cudaSynchronize are the CUDA operations this
// test needs and this package deliberately does not provide. They are stubs:
// the test cannot run until they are replaced with a real binding, and they
// panic rather than returning a zero value so that a half-wired test cannot
// report a pass.
func cudaAlloc(size int) (ptr uintptr, free func(), err error) {
	panic("nvjpeg2k physical test: cudaAlloc is a stub; wire it to a CUDA allocator before running")
}

func cudaCopyToHost(ptr uintptr, size int) ([]byte, error) {
	panic("nvjpeg2k physical test: cudaCopyToHost is a stub; wire it to cudaMemcpy before running")
}

func cudaSynchronize() error {
	panic("nvjpeg2k physical test: cudaSynchronize is a stub; wire it to cudaDeviceSynchronize before running")
}
