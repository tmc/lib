// Package driver is a declaration-only stand-in for the real driver package,
// carrying the handle types and asynchronous copies the checks look for.
package driver

import "unsafe"

type DevicePtr uintptr

func (d DevicePtr) CopyFromHostAsync[T any](srcHost []T, s Stream) error                { return nil }
func (d DevicePtr) CopyToHostAsync[T any](dstHost []T, s Stream) error                  { return nil }
func (d DevicePtr) CopyFromHostAsyncPtr(srcHost unsafe.Pointer, n uint, s Stream) error { return nil }
func (d DevicePtr) CopyToHostAsyncPtr(dstHost unsafe.Pointer, n uint, s Stream) error   { return nil }

type Array uintptr

func (a Array) CopyFromHostAsyncPtr(dstOffset uint, srcHost unsafe.Pointer, n uint, s Stream) error {
	return nil
}

type Stream uintptr

func (s Stream) Destroy() error { return nil }

type Event uintptr

func (e Event) Destroy() error { return nil }

func NewStream(flags uint32) (Stream, error) { return 0, nil }
func NewEvent(flags uint32) (Event, error)   { return 0, nil }
