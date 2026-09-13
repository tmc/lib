//go:build !linux

package cuptisass

func nativeFree(ptr uintptr) {}
