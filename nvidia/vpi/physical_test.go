//go:build linux

package vpi

import (
	"os"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

// This file holds the tests that need a real VPI runtime. Nothing in it has
// ever been executed: the author had no machine with libnvvpi installed. See
// the package documentation for what that leaves unverified.
//
// The tests are opt-in through VPI_PHYSICAL_BACKENDS, and inside that opt-in
// they never skip. An operator who sets
//
//	VPI_PHYSICAL_BACKENDS=cpu,cuda go test ./vpi/
//
// is asserting that this machine has those backends; if the library disagrees,
// or the operation fails, or the output is wrong, the test fails. A skipped run
// is not a passing run, and the skip message below names exactly what is
// missing rather than reporting success.
//
// Loading the library at all requires more than the VPI package. The aarch64
// libnvvpi.so.3 links against the L4T Tegra driver stack — libnvrm_sync.so,
// libnvrm_host1x.so, libnvrm_surface.so, libnvrm_mem.so, libnvbufsurface.so.1.0.0,
// libnvvic.so, libnvmedia.so, libnvsocsys.so, libnvpvaintf.so, libnvos.so and
// libEGL_nvidia.so.0 — so a container or a non-Jetson machine cannot dlopen it
// even with the VPI package unpacked.

const physicalEnv = "VPI_PHYSICAL_BACKENDS"

const physicalPrereq = physicalEnv + ` is unset. These tests need a machine with the VPI 3.2.4 runtime installed:
  - a Jetson module running JetPack 6.x / L4T r36.4, with nvidia-vpi (libnvvpi.so.3)
    and the L4T driver stack it links against (nvidia-l4t-core, nvidia-l4t-multimedia:
    libnvrm_*.so, libnvbufsurface.so.1.0.0, libnvvic.so, libnvmedia.so, libnvpvaintf.so,
    libnvos.so, libEGL_nvidia.so.0);
  - for the cuda backend additionally a CUDA-capable GPU with a driver the installed
    VPI accepts;
  - for the pva, vic, nvenc and ofa backends, a Jetson module that actually has that
    hardware block; they do not exist on x86_64 or on a discrete-GPU aarch64 server.
Set ` + physicalEnv + ` to a comma-separated list of the backends this machine has,
for example "cpu" or "cpu,cuda". Each named backend is then required, not optional.`

var physicalNames = map[string]Backend{
	"cpu":   BackendCPU,
	"cuda":  BackendCUDA,
	"pva":   BackendPVA,
	"vic":   BackendVIC,
	"nvenc": BackendNVENC,
	"ofa":   BackendOFA,
}

// requestedBackends parses VPI_PHYSICAL_BACKENDS, skipping the test when it is
// unset and failing when it names something that is not a backend.
func requestedBackends(t *testing.T) []Backend {
	t.Helper()
	spec := strings.TrimSpace(os.Getenv(physicalEnv))
	if spec == "" {
		t.Skip(physicalPrereq)
	}
	var want []Backend
	for _, name := range strings.Split(spec, ",") {
		name = strings.ToLower(strings.TrimSpace(name))
		if name == "" {
			continue
		}
		b, ok := physicalNames[name]
		if !ok {
			t.Fatalf("%s names %q, which is not a VPI backend", physicalEnv, name)
		}
		want = append(want, b)
	}
	if len(want) == 0 {
		t.Fatalf("%s is set but names no backend", physicalEnv)
	}
	return want
}

// requirePhysical loads the library and checks that it reports every requested
// backend. It fails rather than skipping: the operator asserted these backends
// exist.
func requirePhysical(t *testing.T) []Backend {
	t.Helper()
	want := requestedBackends(t)
	if err := Load(""); err != nil {
		t.Fatalf("%s is set, so a VPI runtime is expected here, but the library did not load: %v", physicalEnv, err)
	}
	have, err := Backends()
	if err != nil {
		t.Fatalf("querying backends: %v", err)
	}
	for _, b := range want {
		if !have.Has(b) {
			t.Fatalf("%s asks for %v; VPI reports %v", physicalEnv, b, have)
		}
	}
	// The bindings are generated from the VPI 3.2.4 headers. A different major
	// version is a different ABI.
	if v := GetVersion(); v/1000000 != 3 {
		t.Fatalf("vpiGetVersion = %d (major %d); these bindings are generated from 3.2.4 headers", v, v/1000000)
	}
	t.Logf("libnvvpi at %s reports version %d and backends %v", Path(), GetVersion(), have)
	return want
}

// TestPhysicalRescale runs one real image operation end to end on every
// requested backend and checks the result, not merely the status code.
//
// The input is a horizontal ramp: every row is 0,1,2,... A linear rescale of a
// monotone ramp is monotone, so the output must be non-decreasing along each
// row and must not be uniform. Checking only that SubmitRescale returned
// Success would pass on an implementation that wrote nothing.
func TestPhysicalRescale(t *testing.T) {
	for _, backend := range requirePhysical(t) {
		t.Run(backend.String(), func(t *testing.T) {
			const (
				inW, inH   = 64, 32
				outW, outH = 32, 16
			)
			// Every image is also given the CPU backend, because the test
			// reads it back through a host lock.
			flags := uint64(backend|BackendCPU) | RequireBackends

			stream, err := StreamCreate(flags)
			if err != nil {
				t.Fatalf("StreamCreate(%v): %v", backend, err)
			}
			defer StreamDestroy(stream)

			input, err := ImageCreate(inW, inH, ImageFormatU8, flags)
			if err != nil {
				t.Fatalf("ImageCreate input: %v", err)
			}
			defer ImageDestroy(input)

			output, err := ImageCreate(outW, outH, ImageFormatU8, flags)
			if err != nil {
				t.Fatalf("ImageCreate output: %v", err)
			}
			defer ImageDestroy(output)

			if w, h, err := ImageGetSize(output); err != nil || w != outW || h != outH {
				t.Fatalf("ImageGetSize = %d,%d,%v; want %d,%d,nil", w, h, err, outW, outH)
			}
			if f, err := ImageGetFormat(input); err != nil || f != ImageFormatU8 {
				t.Fatalf("ImageGetFormat = %v,%v; want U8,nil", f, err)
			}

			writeRamp(t, input, inW, inH)

			if err := SubmitRescale(stream, backend, input, output, InterpLinear, BorderClamp, 0); err != nil {
				t.Fatalf("SubmitRescale on %v: %v", backend, err)
			}
			if err := StreamSync(stream); err != nil {
				t.Fatalf("StreamSync: %v", err)
			}
			checkRamp(t, output, outW, outH)
		})
	}
}

// writeRamp fills img with a horizontal ramp through a host lock.
func writeRamp(t *testing.T, img Image, w, h int32) {
	t.Helper()
	data, err := ImageLockData(img, LockWrite, ImageBufferHostPitchLinear)
	if err != nil {
		t.Fatalf("ImageLockData for write: %v", err)
	}
	defer func() {
		if err := ImageUnlock(img); err != nil {
			t.Fatalf("ImageUnlock: %v", err)
		}
	}()
	buf, err := data.PitchLinear()
	if err != nil {
		t.Fatalf("locked image is not pitch-linear: %v", err)
	}
	if buf.NumPlanes != 1 {
		t.Fatalf("U8 image has %d planes, want 1", buf.NumPlanes)
	}
	plane := &buf.Planes[0]
	if plane.Width != w || plane.Height != h {
		t.Fatalf("locked plane is %dx%d, want %dx%d", plane.Width, plane.Height, w, h)
	}
	bytes, err := data.PlaneBytes(0)
	if err != nil {
		t.Fatalf("PlaneBytes: %v", err)
	}
	for y := int32(0); y < plane.Height; y++ {
		row := bytes[y*plane.PitchBytes:]
		for x := int32(0); x < plane.Width; x++ {
			row[x] = byte(x * 4)
		}
	}
}

// checkRamp reads img back and checks that it still looks like a ramp.
func checkRamp(t *testing.T, img Image, w, h int32) {
	t.Helper()
	data, err := ImageLockData(img, LockRead, ImageBufferHostPitchLinear)
	if err != nil {
		t.Fatalf("ImageLockData for read: %v", err)
	}
	defer func() {
		if err := ImageUnlock(img); err != nil {
			t.Fatalf("ImageUnlock: %v", err)
		}
	}()
	buf, err := data.PitchLinear()
	if err != nil {
		t.Fatalf("locked image is not pitch-linear: %v", err)
	}
	plane := &buf.Planes[0]
	if plane.Width != w || plane.Height != h {
		t.Fatalf("output plane is %dx%d, want %dx%d", plane.Width, plane.Height, w, h)
	}
	bytes, err := data.PlaneBytes(0)
	if err != nil {
		t.Fatalf("PlaneBytes: %v", err)
	}
	for y := int32(0); y < plane.Height; y++ {
		row := bytes[y*plane.PitchBytes:]
		uniform := true
		for x := int32(1); x < plane.Width; x++ {
			if row[x] < row[x-1] {
				t.Fatalf("row %d is not monotone at x=%d: %d then %d", y, x, row[x-1], row[x])
			}
			if row[x] != row[0] {
				uniform = false
			}
		}
		if uniform {
			t.Fatalf("row %d is uniformly %d; the rescale wrote nothing recognizable", y, row[0])
		}
	}
}

// TestPhysicalWrapExternalMemory wraps memory this process owns, runs an
// operation on it, and checks that VPI read the caller's bytes.
//
// The memory is an anonymous mapping, not a Go allocation: ImageCreateWrapper's
// contract requires an address that stays put and that the caller frees, and a
// Go heap object satisfies neither. munmap happens after ImageDestroy.
func TestPhysicalWrapExternalMemory(t *testing.T) {
	backends := requirePhysical(t)
	const (
		inW, inH   = 64, 32
		outW, outH = 32, 16
	)
	backend := backends[0]
	flags := uint64(backend|BackendCPU) | RequireBackends

	mem, err := syscall.Mmap(-1, 0, inW*inH, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	unmapped := false
	defer func() {
		if !unmapped {
			syscall.Munmap(mem)
		}
	}()
	for y := 0; y < inH; y++ {
		for x := 0; x < inW; x++ {
			mem[y*inW+x] = byte(x * 4)
		}
	}

	var buf ImageBufferPitchLinear
	buf.Format = ImageFormatU8
	buf.NumPlanes = 1
	buf.Planes[0] = ImagePlanePitchLinear{
		PixelType:  PixelTypeInvalid,
		Width:      inW,
		Height:     inH,
		PitchBytes: inW,
		Data:       uintptr(unsafe.Pointer(&mem[0])),
	}
	var data ImageData
	if err := data.SetPitchLinear(ImageBufferHostPitchLinear, buf); err != nil {
		t.Fatal(err)
	}
	params, err := InitImageWrapperParams()
	if err != nil {
		t.Fatalf("InitImageWrapperParams: %v", err)
	}
	if params.ColorSpec != ColorSpecDefault {
		t.Errorf("InitImageWrapperParams left ColorSpec = %d, want ColorSpecDefault = %d", params.ColorSpec, ColorSpecDefault)
	}

	input, err := ImageCreateWrapper(&data, &params, flags)
	if err != nil {
		t.Fatalf("ImageCreateWrapper: %v", err)
	}
	// The wrapper must die before the memory does.
	defer func() {
		ImageDestroy(input)
		if err := syscall.Munmap(mem); err != nil {
			t.Errorf("munmap: %v", err)
		}
		unmapped = true
	}()

	if w, h, err := ImageGetSize(input); err != nil || w != inW || h != inH {
		t.Fatalf("wrapped image is %d,%d,%v; want %d,%d,nil", w, h, err, inW, inH)
	}

	stream, err := StreamCreate(flags)
	if err != nil {
		t.Fatalf("StreamCreate: %v", err)
	}
	defer StreamDestroy(stream)

	output, err := ImageCreate(outW, outH, ImageFormatU8, flags)
	if err != nil {
		t.Fatalf("ImageCreate output: %v", err)
	}
	defer ImageDestroy(output)

	if err := SubmitRescale(stream, backend, input, output, InterpLinear, BorderClamp, 0); err != nil {
		t.Fatalf("SubmitRescale on wrapped memory: %v", err)
	}
	if err := StreamSync(stream); err != nil {
		t.Fatalf("StreamSync: %v", err)
	}
	checkRamp(t, output, outW, outH)
}

// TestPhysicalBoundSymbolsResolve checks that every symbol these bindings name
// is present in the library that actually loaded, which is what
// cmd/gen-vpi's TestBoundSymbolsExistInPinnedLibrary cannot check: it reads the
// symbol table of the pinned 3.2.4 package, not of the installed runtime.
func TestPhysicalBoundSymbolsResolve(t *testing.T) {
	requirePhysical(t)
	var missing []string
	for _, symbol := range boundSymbols {
		if !Has(symbol) {
			missing = append(missing, symbol)
		}
	}
	if len(missing) > 0 {
		t.Errorf("the loaded library (%s, version %d) is missing %d bound symbols: %s",
			Path(), GetVersion(), len(missing), strings.Join(missing, " "))
	}
}

// boundSymbols lists every C symbol registerFuncs binds. TestBoundSymbolsMatch
// in cmd/gen-vpi keeps it in step with the generated loader, so a symbol added
// or removed by a regeneration cannot go unnoticed here.
var boundSymbols = []string{
	"vpiStatusGetName",
	"vpiContextCreate",
	"vpiContextDestroy",
	"vpiContextGetCurrent",
	"vpiContextGetFlags",
	"vpiContextPop",
	"vpiContextPush",
	"vpiContextSetCurrent",
	"vpiGetLastStatus",
	"vpiGetVersion",
	"vpiImageCreate",
	"vpiImageCreateView",
	"vpiImageCreateWrapper",
	"vpiImageDestroy",
	"vpiImageGetFlags",
	"vpiImageGetFormat",
	"vpiImageGetSize",
	"vpiImageLock",
	"vpiImageLockData",
	"vpiImageSetView",
	"vpiImageSetWrapper",
	"vpiImageUnlock",
	"vpiInitImageWrapperParams",
	"vpiPayloadDestroy",
	"vpiPayloadGetFlags",
	"vpiPeekAtLastStatus",
	"vpiStreamCreate",
	"vpiStreamDestroy",
	"vpiStreamFlush",
	"vpiStreamGetFlags",
	"vpiStreamSync",
	"vpiStreamWaitEvent",
	"vpiSubmitRescale",
}
