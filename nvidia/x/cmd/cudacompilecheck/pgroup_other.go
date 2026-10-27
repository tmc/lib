//go:build !unix

package main

import "os/exec"

func setProcessGroup(cmd *exec.Cmd) {}

func isGroupError(err error) bool { return false }
