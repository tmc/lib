//go:build linux

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// validateVisionLease is a cooperative live-lease check, not a security
// boundary. It corroborates the sealed owner line, holder PID/token/host and
// process ancestry against the actual lock and /proc rather than env presence.
func validateVisionLease() error {
	// The reviewed DGX lock is the shared holder, not a caller-supplied owner
	// file which could corroborate an unrelated process.
	const ownerPath = "/home/tmc/ml-explore/.gpu-lock/lock/owner"
	if os.Getenv("DS4_VISION_LOCK_OWNER_FILE") != ownerPath {
		return fmt.Errorf("vision lock owner path is not the shared DGX lock")
	}
	owner, err := os.ReadFile(ownerPath)
	if err != nil {
		return err
	}
	if strings.TrimSuffix(string(owner), "\n") != os.Getenv("DS4_VISION_LOCK_OWNER") {
		return fmt.Errorf("vision lock owner changed")
	}
	fields := strings.Fields(string(owner))
	if len(fields) < 5 || !strings.HasPrefix(fields[0], "pid=") || !strings.HasPrefix(fields[1], "host=") {
		return fmt.Errorf("invalid vision lock owner")
	}
	pid, err := strconv.Atoi(strings.TrimPrefix(fields[0], "pid="))
	if err != nil || pid < 2 {
		return fmt.Errorf("invalid vision lock PID")
	}
	if strconv.Itoa(pid) != os.Getenv("GPU_LOCK_HELD") {
		return fmt.Errorf("vision holder PID mismatch")
	}
	host, err := os.Hostname()
	if err != nil {
		return err
	}
	host = strings.SplitN(host, ".", 2)[0]
	if fields[1] != "host="+host {
		return fmt.Errorf("vision lock host mismatch")
	}
	token := ""
	for _, field := range fields {
		if strings.HasPrefix(field, "token=") {
			token = strings.TrimPrefix(field, "token=")
		}
	}
	if token == "" || !strings.HasPrefix(token, host+"-"+strconv.Itoa(pid)+"-") {
		return fmt.Errorf("invalid vision lock token")
	}
	cmd, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return err
	}
	lock, err := filepath.EvalSymlinks(os.Getenv("DS4_VISION_LOCK_EXEC"))
	if err != nil {
		return err
	}
	matched := false
	for _, arg := range strings.Split(string(cmd), "\x00") {
		if arg == lock {
			matched = true
		}
	}
	if !matched {
		return fmt.Errorf("holder is not the sealed gpu-lock executable")
	}
	ancestor := os.Getpid()
	for n := 0; n < 128 && ancestor > 1; n++ {
		if ancestor == pid {
			return nil
		}
		status, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", ancestor))
		if err != nil {
			return err
		}
		parent := 0
		for _, line := range strings.Split(string(status), "\n") {
			if strings.HasPrefix(line, "PPid:") {
				parent, err = strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "PPid:")))
				if err != nil {
					return err
				}
			}
		}
		if parent <= 0 || parent == ancestor {
			return fmt.Errorf("invalid holder ancestry")
		}
		ancestor = parent
	}
	return fmt.Errorf("vision test is not descended from lock holder")
}
