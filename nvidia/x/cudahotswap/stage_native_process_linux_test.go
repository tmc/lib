//go:build linux

package cudahotswap

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"syscall"
	"testing"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
)

const stageNativeExecutableLimit = 16 << 20

// stageNativeObservedProcess records this running process's executable body.
// /proc/self/exe is the kernel-owned link to the actual executing image, not
// an expected filename supplied by the operation controller.
func stageNativeObservedProcess() (nativereceipt.Process, error) {
	process, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		return nativereceipt.Process{}, err
	}
	f, err := os.Open("/proc/self/exe")
	if err != nil {
		return nativereceipt.Process{}, fmt.Errorf("open actual executable: %w", err)
	}
	defer f.Close()
	before, err := f.Stat()
	if err != nil {
		return nativereceipt.Process{}, fmt.Errorf("stat actual executable: %w", err)
	}
	if !before.Mode().IsRegular() || before.Size() <= 0 || before.Size() > stageNativeExecutableLimit {
		return nativereceipt.Process{}, fmt.Errorf("actual executable exceeds size/type limit")
	}
	hash := sha256.New()
	n, err := io.Copy(hash, io.LimitReader(f, stageNativeExecutableLimit+1))
	if err != nil {
		return nativereceipt.Process{}, fmt.Errorf("hash actual executable: %w", err)
	}
	after, err := f.Stat()
	if err != nil {
		return nativereceipt.Process{}, fmt.Errorf("stat hashed executable: %w", err)
	}
	a, ok := before.Sys().(*syscall.Stat_t)
	b, other := after.Sys().(*syscall.Stat_t)
	if !ok || !other || !os.SameFile(before, after) || before.Size() != after.Size() || before.Mode() != after.Mode() || !before.ModTime().Equal(after.ModTime()) || a.Ctim != b.Ctim || n != before.Size() || n > stageNativeExecutableLimit {
		return nativereceipt.Process{}, fmt.Errorf("actual executable changed during hashing")
	}
	if err := f.Close(); err != nil {
		return nativereceipt.Process{}, fmt.Errorf("close actual executable: %w", err)
	}
	current, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		return nativereceipt.Process{}, err
	}
	if current != process {
		return nativereceipt.Process{}, fmt.Errorf("actual process identity changed during hashing")
	}
	process.ExecutableSHA256 = hex.EncodeToString(hash.Sum(nil))
	return process, nil
}

func TestStageNativeObservedExecutableCPU(t *testing.T) {
	process, err := stageNativeObservedProcess()
	if err != nil {
		t.Fatal(err)
	}
	identity, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(identity.Executable)
	if err != nil {
		t.Fatal(err)
	}
	want := sha256.Sum256(raw)
	if process.ExecutableSHA256 != hex.EncodeToString(want[:]) {
		t.Fatal("producer does not record actual executing body digest")
	}
	process.ExecutableSHA256 = ""
	if process != identity {
		t.Fatal("producer changes observed process identity")
	}
}
