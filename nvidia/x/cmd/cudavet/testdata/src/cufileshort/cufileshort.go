package cufileshort

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cufile"
	"github.com/tmc/lib/nvidia/driver"
)

func blankCount(h cufile.Handle, buf uintptr, n uint) error {
	_, err := h.Read(buf, n, 0, 0) // want `the byte count of Handle.Read is discarded; a short transfer is reported only by the count`
	return err
}

func blankWrite(h cufile.Handle, buf uintptr, n uint) error {
	var err error
	_, err = h.Write(buf, n, 0, 0) // want `the byte count of Handle.Write is discarded`
	return err
}

func bare(h cufile.Handle, buf uintptr, n uint) {
	h.Write(buf, n, 0, 0) // want `the byte count of Handle.Write is discarded`
}

func allBlank(h cufile.Handle, buf uintptr, n uint) {
	_, _ = h.Read(buf, n, 0, 0) // want `the byte count of Handle.Read is discarded`
}

func checked(h cufile.Handle, buf uintptr, n uint) error {
	got, err := h.Read(buf, n, 0, 0)
	if err != nil {
		return err
	}
	if got != int64(n) {
		return fmt.Errorf("short read: %d of %d bytes", got, n)
	}
	return nil
}

// Only the error is discarded; the count is kept.
func countKept(h cufile.Handle, buf uintptr, n uint) int64 {
	got, _ := h.Write(buf, n, 0, 0)
	return got
}

func returned(h cufile.Handle, buf uintptr, n uint) (int64, error) {
	return h.Read(buf, n, 0, 0)
}

// ReadAsync delivers its count through bytesRead when the stream runs.
func async(h cufile.Handle, buf uintptr, size *uint, off, bufOff, bytesRead *int64, s driver.Stream) error {
	return h.ReadAsync(buf, size, off, bufOff, bytesRead, s)
}

type file struct{}

func (file) Read(p []byte, off int64) (int, error) { return len(p), nil }

// Read of another type is not cuFile's.
func other(f file, p []byte) {
	f.Read(p, 1)
}

// A zero-byte transfer tests the error path; it has no count to check.
func zeroBytes(h cufile.Handle) error {
	_, err := h.Read(0, 0, 0, 0)
	return err
}
