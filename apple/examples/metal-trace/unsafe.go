package main

import "unsafe"

type unsafePointer = unsafe.Pointer

func unsafeSliceCast[T any](ptr unsafe.Pointer, length int) []T {
	return unsafe.Slice((*T)(ptr), length)
}
