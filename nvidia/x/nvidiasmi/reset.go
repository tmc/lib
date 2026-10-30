package nvidiasmi

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Reset resets the hardware and software state of the GPUs named by
// ids, by running nvidia-smi --gpu-reset -i with ids joined by commas.
// It is meant to clear a GPU that would otherwise need a reboot, such
// as after a double-bit ECC error.
//
// Each id must be a GPU's index ("0"; at most three digits), UUID
// ("GPU-" and 32 hexadecimal digits with dashes, as in [GPU.UUID]),
// or PCI bus ID (domain:bus:device.function in hexadecimal, as in
// [GPU.ID]). Reset returns an error without running nvidia-smi if ids
// is empty or any id has another form: nvidia-smi resets every GPU
// when given no -i, and also accepts board serial numbers, which
// several GPUs on one board can share. Indexes follow the driver's
// enumeration, which can change across reboots; prefer UUIDs or PCI
// bus IDs.
//
// nvidia-smi(1) documents these preconditions
// (https://docs.nvidia.com/deploy/nvidia-smi/index.html):
//
//   - It requires root (administrator privileges on Windows).
//   - No application may be using the GPUs, including CUDA programs,
//     X servers, and monitoring tools such as another nvidia-smi.
//     If the GPUs cannot be reset individually, no compute
//     application may be running on any GPU in the system.
//   - On pre-Ampere GPUs, and on Ampere NVSwitch systems without a
//     running Fabric Manager, every GPU connected to a reset GPU by
//     NVLink must be reset in the same call; nvidia-smi reports an
//     error naming the missing GPUs otherwise.
//   - Reset is not supported on MIG-enabled vGPU guests. On Windows
//     it restarts the driver of the named GPUs.
//
// NVIDIA also documents that a reset is not guaranteed to work, is
// not recommended for production environments, and that each reset
// GPU's health should be checked before further use, power cycling
// the node if it is unhealthy. Reset issues nvidia-smi's default
// function-level reset.
//
// If nvidia-smi exits with a nonzero status, the error is an *[Error]
// holding its exit status and output. A GPU in use has no documented
// exit status of its own; nvidia-smi names the reason in its output.
// Canceling ctx kills nvidia-smi, which can leave a reset incomplete.
func Reset(ctx context.Context, ids ...string) error {
	if len(ids) == 0 {
		return errors.New("reset gpu: no GPU named")
	}
	for _, id := range ids {
		if !validID(id) {
			return fmt.Errorf("reset gpu: %q is not a GPU index, UUID, or PCI bus ID", id)
		}
	}
	_, err := run(ctx, "--gpu-reset", "-i", strings.Join(ids, ","))
	return err
}

// validID reports whether id is a GPU index, UUID, or PCI bus ID.
func validID(id string) bool {
	if len(id) >= 1 && len(id) <= 3 && isDigits(id) {
		return true
	}
	if u, ok := strings.CutPrefix(id, "GPU-"); ok {
		return hexGroups(u, "-", 8, 4, 4, 4, 12)
	}
	// domain:bus:device.function; the domain is 4 or 8 digits.
	dom, rest, ok := strings.Cut(id, ":")
	if !ok || (len(dom) != 4 && len(dom) != 8) || !isHex(dom) {
		return false
	}
	bus, rest, ok := strings.Cut(rest, ":")
	if !ok || !hexGroups(bus, "", 2) {
		return false
	}
	return hexGroups(rest, ".", 2, 1)
}

// hexGroups reports whether s is sep-separated groups of hexadecimal
// digits with the given lengths.
func hexGroups(s, sep string, lens ...int) bool {
	var groups []string
	if sep == "" {
		groups = []string{s}
	} else {
		groups = strings.Split(s, sep)
	}
	if len(groups) != len(lens) {
		return false
	}
	for i, g := range groups {
		if len(g) != lens[i] || !isHex(g) {
			return false
		}
	}
	return true
}

func isDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func isHex(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i] | 0x20
		if !('0' <= s[i] && s[i] <= '9') && !('a' <= c && c <= 'f') {
			return false
		}
	}
	return true
}
