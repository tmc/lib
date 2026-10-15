package nvtelemetry

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// ErrProcEntityNotLive means the supplied proc stat snapshot identifies a
// zombie or dead process. No process identity is returned for that snapshot.
var ErrProcEntityNotLive = errors.New("proc entity: process is not live")

// ProcEntityFacts preserves Linux proc observations. Cgroups are raw membership
// paths, not proof of container or pod UID. StartTimeTicks is proc stat field 22
// in clock ticks since boot; it is not a nanosecond or an exec generation.
type ProcEntityFacts struct {
	Process        EntityProcess
	StartTimeTicks uint64
	Cgroups        []string
}

// ReadProcEntity reads supplied /proc/PID/stat, status and cgroup snapshots.
// bootID, nodeID and pidNamespace must be supplied from the same host snapshot.
// It performs no filesystem access and cannot prove these reads were atomic.
// The returned process generation is the boot-scoped start time tick count.
// A zombie or dead process returns [ErrProcEntityNotLive]. Other states describe
// only the supplied snapshot, not whether the process remains alive.
func ReadProcEntity(stat, status, cgroup io.Reader, bootID, nodeID, pidNamespace string) (ProcEntityFacts, error) {
	var f ProcEntityFacts
	if bootID == "" || nodeID == "" {
		return f, fmt.Errorf("proc entity: missing boot or node identity")
	}
	b, err := io.ReadAll(io.LimitReader(stat, 65537))
	if err != nil {
		return f, err
	}
	if len(b) > 65536 {
		return f, fmt.Errorf("proc entity: stat too large")
	}
	s := strings.TrimSpace(string(b))
	left, right := strings.IndexByte(s, '('), strings.LastIndexByte(s, ')')
	if left <= 0 || right <= left {
		return f, fmt.Errorf("proc entity: invalid stat command")
	}
	pid, err := strconv.ParseUint(strings.TrimSpace(s[:left]), 10, 32)
	if err != nil || pid == 0 {
		return f, fmt.Errorf("proc entity: invalid pid")
	}
	fields := strings.Fields(s[right+1:])
	if len(fields) < 20 {
		return f, fmt.Errorf("proc entity: missing start time")
	}
	start, err := strconv.ParseUint(fields[19], 10, 64)
	if err != nil || start == 0 {
		return f, fmt.Errorf("proc entity: invalid start time")
	}
	if len(fields[0]) != 1 {
		return f, fmt.Errorf("proc entity: invalid process state")
	}
	if fields[0] == "Z" || fields[0] == "X" || fields[0] == "x" {
		return f, fmt.Errorf("%w: pid %d state %s", ErrProcEntityNotLive, pid, fields[0])
	}
	f.Process = EntityProcess{Process: cudaworkload.ProcessGenID{PID: uint32(pid), Generation: start}, BootID: bootID, NodeID: nodeID, PIDNamespace: pidNamespace}
	f.StartTimeTicks = start
	err = scanEntityLines(status, func(line string) error {
		if strings.HasPrefix(line, "NSpid:") {
			if len(f.Process.NamespacePIDs) > 0 {
				return fmt.Errorf("proc entity: repeated namespace pid")
			}
			for _, word := range strings.Fields(strings.TrimPrefix(line, "NSpid:")) {
				n, e := strconv.ParseUint(word, 10, 32)
				if e != nil || n == 0 {
					return fmt.Errorf("proc entity: invalid namespace pid")
				}
				f.Process.NamespacePIDs = append(f.Process.NamespacePIDs, uint32(n))
			}
		}
		return nil
	})
	if err != nil {
		return ProcEntityFacts{}, err
	}
	if len(f.Process.NamespacePIDs) > 0 && f.Process.NamespacePIDs[0] != uint32(pid) {
		return ProcEntityFacts{}, fmt.Errorf("proc entity: inconsistent namespace pid")
	}
	err = scanEntityLines(cgroup, func(line string) error {
		parts := strings.SplitN(line, ":", 3)
		if len(parts) != 3 || !strings.HasPrefix(parts[2], "/") {
			return fmt.Errorf("proc entity: invalid cgroup")
		}
		if _, e := strconv.ParseUint(parts[0], 10, 32); e != nil {
			return fmt.Errorf("proc entity: invalid cgroup hierarchy")
		}
		f.Cgroups = append(f.Cgroups, line)
		return nil
	})
	if err != nil {
		return ProcEntityFacts{}, err
	}
	return f, nil
}

func scanEntityLines(r io.Reader, fn func(string) error) error {
	s := bufio.NewScanner(io.LimitReader(r, entityMaxBytes+1))
	s.Buffer(make([]byte, 1024), 65536)
	n := 0
	bytes := 0
	for s.Scan() {
		n++
		bytes += len(s.Bytes()) + 1
		if n > entityMaxRecords || bytes > entityMaxBytes {
			return fmt.Errorf("proc entity: document too large")
		}
		if err := fn(s.Text()); err != nil {
			return err
		}
	}
	if err := s.Err(); err != nil {
		return fmt.Errorf("proc entity: %w", err)
	}
	return nil
}
