package cudalock

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Owner describes the process currently holding the lock.
type Owner struct {
	PID   int
	Host  string
	Label string
	Epoch int64
	At    time.Time
}

// String formats the owner line byte-compatibly with bash gpu-lock:
// "pid=<pid> host=<host> label=<label> epoch=<epoch> at=<iso8601>\n"
func (o Owner) String() string {
	atStr := o.At.Format(time.RFC3339)
	if atStr == "" || o.At.IsZero() {
		atStr = time.Now().Format(time.RFC3339)
	}
	return fmt.Sprintf("pid=%d host=%s label=%s epoch=%d at=%s\n",
		o.PID, o.Host, o.Label, o.Epoch, atStr)
}

// ParseOwner parses an owner file line.
func ParseOwner(data []byte) (Owner, error) {
	line := strings.TrimSpace(string(data))
	if line == "" {
		return Owner{}, fmt.Errorf("cudalock: empty owner data")
	}

	var o Owner
	fields := strings.Fields(line)
	for _, f := range fields {
		k, v, ok := strings.Cut(f, "=")
		if !ok {
			continue
		}
		switch k {
		case "pid":
			p, err := strconv.Atoi(v)
			if err != nil {
				return Owner{}, fmt.Errorf("cudalock: invalid pid %q: %w", v, err)
			}
			o.PID = p
		case "host":
			o.Host = v
		case "label":
			o.Label = v
		case "epoch":
			ep, err := strconv.ParseInt(v, 10, 64)
			if err == nil {
				o.Epoch = ep
			}
		case "at":
			t, err := time.Parse(time.RFC3339, v)
			if err == nil {
				o.At = t
			}
		}
	}

	if o.PID <= 0 {
		return Owner{}, fmt.Errorf("cudalock: missing or invalid pid in owner record %q", line)
	}
	return o, nil
}

// isAlive checks whether a process with the given PID is currently alive.
// Returns false only when the process is confirmed dead (ESRCH).
// If we lack permission (EPERM), the process is still running.
func isAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	err = process.Signal(syscall.Signal(0))
	if err == nil {
		return true
	}
	if err == syscall.EPERM {
		return true // Process exists but owned by another user
	}
	if err == syscall.ESRCH {
		return false // No such process
	}
	return false
}

// shortHostname returns the host's short name (up to first dot).
func shortHostname() string {
	h, err := os.Hostname()
	if err != nil || h == "" {
		return "localhost"
	}
	if dot := strings.IndexByte(h, '.'); dot != -1 {
		return h[:dot]
	}
	return h
}
