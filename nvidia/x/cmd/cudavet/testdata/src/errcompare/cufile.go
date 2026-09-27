package errcompare

import (
	"errors"
	"io/fs"
	"os"
	"syscall"

	"github.com/tmc/lib/nvidia/cufile"
)

func missing(h cufile.Handle, buf uintptr) bool {
	_, err := h.Read(buf, 4096, 0, 0)
	return err == syscall.ENOENT // want `a failed cuFile call returns an \*ErrnoError, which never equals syscall.ENOENT; use errors.Is`
}

func notExist(h cufile.Handle, buf uintptr) bool {
	_, err := h.Read(buf, 4096, 0, 0)
	return fs.ErrNotExist == err // want `a failed cuFile call returns an \*ErrnoError, which never equals fs.ErrNotExist; use errors.Is`
}

func errno(h cufile.Handle, buf uintptr) string {
	_, err := h.Read(buf, 4096, 0, 0)
	switch err {
	case syscall.EIO: // want `a failed cuFile call returns an \*ErrnoError, which never equals syscall.EIO; use errors.Is`
		return "io"
	case os.ErrPermission: // want `a failed cuFile call returns an \*ErrnoError, which never equals os.ErrPermission; use errors.Is`
		return "permission"
	case nil:
		return ""
	}
	return "other"
}

func matched(h cufile.Handle, buf uintptr) bool {
	_, err := h.Read(buf, 4096, 0, 0)
	return errors.Is(err, syscall.ENOENT) || errors.Is(err, fs.ErrNotExist)
}

func carried(h cufile.Handle, buf uintptr) bool {
	_, err := h.Read(buf, 4096, 0, 0)
	var e *cufile.ErrnoError
	return errors.As(err, &e) && e.Errno == syscall.ENOENT
}
