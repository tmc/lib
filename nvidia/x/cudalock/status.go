package cudalock

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ComputeApp represents an active GPU compute application running on the hardware.
type ComputeApp struct {
	PID        int
	Name       string
	UsedMemory string
}

// StatusReport details both the filesystem mutex status and physical GPU device activity.
type StatusReport struct {
	LockDir      string
	MutexHeld    bool
	Owner        *Owner
	OwnerAlive   bool
	OwnerAge     time.Duration
	ComputeApps  []ComputeApp
	DeviceError  string
}

// String formats the status report clearly dividing the Mutex and Device sections.
func (s *StatusReport) String() string {
	var sb strings.Builder
	sb.WriteString("=== FILESYSTEM MUTEX ===\n")
	sb.WriteString(fmt.Sprintf("Directory: %s\n", s.LockDir))
	if s.MutexHeld && s.Owner != nil {
		aliveStr := "ALIVE"
		if !s.OwnerAlive {
			aliveStr = "DEAD"
		}
		sb.WriteString(fmt.Sprintf("Status:    HELD (pid=%d, host=%s, label=%s, %s)\n",
			s.Owner.PID, s.Owner.Host, s.Owner.Label, aliveStr))
		sb.WriteString(fmt.Sprintf("Acquired:  %s (%s ago)\n",
			s.Owner.At.Format(time.RFC3339), s.OwnerAge.Round(time.Second)))
	} else if s.MutexHeld {
		sb.WriteString("Status:    HELD (directory exists without valid owner file)\n")
	} else {
		sb.WriteString("Status:    FREE\n")
	}

	sb.WriteString("\n=== PHYSICAL DEVICE ACTIVITY ===\n")
	if s.DeviceError != "" {
		sb.WriteString(fmt.Sprintf("Status:    UNAVAILABLE (%s)\n", s.DeviceError))
	} else if len(s.ComputeApps) == 0 {
		sb.WriteString("Status:    IDLE (0 compute applications active)\n")
	} else {
		sb.WriteString(fmt.Sprintf("Status:    ACTIVE (%d compute application(s) running)\n", len(s.ComputeApps)))
		sb.WriteString(fmt.Sprintf("%-8s %-32s %s\n", "PID", "NAME", "MEMORY"))
		for _, app := range s.ComputeApps {
			sb.WriteString(fmt.Sprintf("%-8d %-32s %s\n", app.PID, app.Name, app.UsedMemory))
		}
	}
	return sb.String()
}

// Status inspects the specified lock directory (or default if empty) and device state.
func Status(dir string) (*StatusReport, error) {
	if dir == "" {
		dir = DefaultDir()
	}

	rep := &StatusReport{
		LockDir: dir,
	}

	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		rep.MutexHeld = true
		ownerFile := filepath.Join(dir, "owner")
		if data, err := os.ReadFile(ownerFile); err == nil {
			if o, err := ParseOwner(data); err == nil {
				rep.Owner = &o
				rep.OwnerAlive = isAlive(o.PID)
				if !o.At.IsZero() {
					rep.OwnerAge = time.Since(o.At)
				} else if o.Epoch > 0 {
					rep.OwnerAge = time.Since(time.Unix(o.Epoch, 0))
				}
			}
		}
	}

	// Query physical device compute applications via nvidia-smi
	apps, err := QueryComputeApps()
	if err != nil {
		rep.DeviceError = err.Error()
	} else {
		rep.ComputeApps = apps
	}

	return rep, nil
}

// QueryComputeApps queries active compute applications across GPUs via nvidia-smi.
func QueryComputeApps() ([]ComputeApp, error) {
	smiPath, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi not found: %w", err)
	}

	cmd := exec.Command(smiPath,
		"--query-compute-apps=pid,process_name,used_memory",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("query compute apps: %w", err)
	}

	var apps []ComputeApp
	lines := strings.Split(string(bytes.TrimSpace(out)), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		name := strings.TrimSpace(parts[1])
		mem := strings.TrimSpace(parts[2]) + " MiB"
		apps = append(apps, ComputeApp{
			PID:        pid,
			Name:       name,
			UsedMemory: mem,
		})
	}
	return apps, nil
}
