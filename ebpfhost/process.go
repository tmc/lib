package ebpfhost

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ProcessStartTime returns /proc stat starttime ticks, a PID generation token.
// This is not a clock mapping or proof of workload identity.
func ProcessStartTime(pid int) (uint64, error) {
	data, e := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if e != nil {
		return 0, e
	}
	return parseStartTime(string(data))
}
func parseStartTime(stat string) (uint64, error) {
	end := strings.LastIndex(stat, ")")
	if end < 0 {
		return 0, fmt.Errorf("invalid process stat")
	}
	fields := strings.Fields(stat[end+1:])
	if len(fields) < 20 {
		return 0, fmt.Errorf("short process stat")
	}
	n, e := strconv.ParseUint(fields[19], 10, 64)
	if e != nil {
		return 0, fmt.Errorf("process starttime: %w", e)
	}
	return n, nil
}
