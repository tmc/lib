//go:build windows || !(unix || darwin || linux || freebsd || openbsd || netbsd)

package nvtelemetry

import (
	"errors"
	"fmt"
	"os"
	"runtime"
)

// ErrPlatformUnsupported is returned when durable delivery writer leases
// are used on an operating system without supported file locking primitives.
var ErrPlatformUnsupported = errors.New("telemetry delivery: platform unsupported")

func lockFileExclusive(f *os.File) error {
	return fmt.Errorf("%w: writer lease flock is unsupported on %s", ErrPlatformUnsupported, runtime.GOOS)
}

func unlockFile(f *os.File) error {
	return nil
}
