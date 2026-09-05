//go:build linux

package vpi

import "strings"

// Backend is a bit mask selecting VPI compute backends. It is the C type
// VPIBackend, whose values are single bits so that a set of backends is their
// bitwise OR.
//
// Backend availability is a property of the machine, not of this package.
// Nothing here reports that a backend exists; [Backends] asks the library.
type Backend uint64

// Backends. Each is one bit of VPIBackend.
//
// Only [BackendCPU] is present on every platform VPI supports. [BackendCUDA]
// needs a CUDA-capable GPU and a driver the installed VPI accepts. [BackendPVA],
// [BackendVIC], [BackendNVENC] and [BackendOFA] are Tegra (Jetson) hardware
// blocks and do not exist on a discrete-GPU x86_64 or aarch64 server; see
// [BackendTegra]. Even on Jetson, a given module may lack a given block.
const (
	BackendCPU     Backend = 1 << 0  // host CPU
	BackendCUDA    Backend = 1 << 1  // CUDA-capable GPU
	BackendPVA     Backend = 1 << 2  // Programmable Vision Accelerator (Tegra only)
	BackendVIC     Backend = 1 << 3  // Video Image Compositor (Tegra only)
	BackendNVENC   Backend = 1 << 4  // video encoder engine (Tegra only)
	BackendOFA     Backend = 1 << 5  // Optical Flow Accelerator (Tegra only)
	BackendInvalid Backend = 1 << 15 // not a backend; VPI_BACKEND_INVALID
)

const (
	// BackendTegra is the set of backends that exist only on Tegra (Jetson)
	// hardware. It is VPI_BACKEND_TEGRA.
	BackendTegra = BackendVIC | BackendNVENC | BackendOFA

	// BackendAll is every defined backend. It is VPI_BACKEND_ALL. Passing it
	// is a request, not an assertion: without [RequireBackends] VPI silently
	// drops the backends the platform does not have.
	BackendAll = BackendCPU | BackendCUDA | BackendPVA | BackendVIC | BackendNVENC | BackendOFA

	// BackendMask covers the bits of a flags word that select backends. It is
	// VPI_BACKEND_MASK. The remaining bits carry the object flags in types.go.
	BackendMask Backend = (BackendInvalid << 1) - 1
)

var backendNames = []struct {
	bit  Backend
	name string
}{
	{BackendCPU, "CPU"},
	{BackendCUDA, "CUDA"},
	{BackendPVA, "PVA"},
	{BackendVIC, "VIC"},
	{BackendNVENC, "NVENC"},
	{BackendOFA, "OFA"},
	{BackendInvalid, "INVALID"},
}

// String returns the backend bits as a "|"-joined list, or "none" when no bit
// is set. Bits outside the known set are reported numerically.
func (b Backend) String() string {
	if b == 0 {
		return "none"
	}
	var parts []string
	rest := b
	for _, n := range backendNames {
		if rest&n.bit != 0 {
			parts = append(parts, n.name)
			rest &^= n.bit
		}
	}
	if rest != 0 {
		parts = append(parts, "0x"+strings.TrimLeft(hex(uint64(rest)), "0"))
	}
	return strings.Join(parts, "|")
}

func hex(v uint64) string {
	const digits = "0123456789abcdef"
	var buf [16]byte
	for i := 15; i >= 0; i-- {
		buf[i] = digits[v&0xf]
		v >>= 4
	}
	return string(buf[:])
}

// Has reports whether every backend in want is set in b.
func (b Backend) Has(want Backend) bool { return b&want == want }

// Backends reports the backends the running platform provides.
//
// It asks the library rather than inferring anything: a context created with no
// backend flags enables every backend VPI finds on the machine, and
// [ContextGetFlags] then reports that set. The context is created and destroyed
// by this call and does not become current.
//
// A returned set is the library's answer at this moment for this process. It is
// not a claim that an operation will succeed on a listed backend: an individual
// algorithm supports its own subset of backends and image formats, and a
// backend can still fail at submission. Callers that need a specific backend
// should also pass [RequireBackends] when creating objects, so an unavailable
// backend fails at creation instead of at submission.
//
// Backends returns an error if the library is unavailable, which is the same
// error every other call in this package returns in that case.
func Backends() (Backend, error) {
	ctx, err := ContextCreate(0)
	if err != nil {
		return 0, err
	}
	defer ContextDestroy(ctx)
	flags, err := ContextGetFlags(ctx)
	if err != nil {
		return 0, err
	}
	return Backend(flags) & BackendMask, nil
}
