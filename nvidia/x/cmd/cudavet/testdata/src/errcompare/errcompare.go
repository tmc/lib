package errcompare

import (
	"errors"

	"github.com/tmc/lib/nvidia/driver"
)

func invalid(err error) bool {
	return err == driver.ErrInvalidValue // want `a failed call returns a \*CallError, which never equals driver.ErrInvalidValue; use errors.Is`
}

func kind(err error) string {
	switch err {
	case driver.ErrNotReady: // want `a failed call returns a \*CallError, which never equals driver.ErrNotReady; use errors.Is`
		return "not ready"
	}
	return ""
}

func is(err error) bool {
	return errors.Is(err, driver.ErrInvalidValue)
}

func result(r driver.Result) bool {
	return r == driver.ErrInvalidValue
}

func unwrapped(err error) bool {
	var ce *driver.CallError
	return errors.As(err, &ce) && ce.Result == driver.ErrNotReady
}
