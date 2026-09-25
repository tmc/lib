//go:build !linux

package main

import (
	"errors"
	"syscall"
)

var signals = []struct {
	name string
	sig  syscall.Signal
}{
	{"Cancel", 0},
}

func sendSignal(pid int, sig syscall.Signal) error {
	return errors.New("sending signals is not supported")
}
