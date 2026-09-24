package cudaerrors

import "errors"

// ErrUnsupported reports that an instrument examined the target and refused.
// It is a measurement.
var ErrUnsupported = errors.New("target does not support the requested capability")

// ErrHardwarePending reports that no instrument has examined the target, so
// support is unknown. It is the absence of a measurement and must never be
// reported as a negative one.
//
// The distinction from [ErrUnsupported] is the whole point of this package.
// "No instrument has looked" and "the instrument said no" are different
// answers, and code that collapses them reports an untested target as a
// failing one -- which reads as a hardware fact and is not one.
var ErrHardwarePending = errors.New("capability not established for target")
