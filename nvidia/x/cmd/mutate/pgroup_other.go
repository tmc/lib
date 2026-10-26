//go:build !unix

package main

import "os/exec"

func setProcessGroup(cmd *exec.Cmd) {}
