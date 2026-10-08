package ncclras

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

// UnsupportedError identifies a format or version outside the reader's profile.
// Version is empty when the producer did not report a product version. Protocol
// version 2 alone is not a product version or evidence of JSON support.
type UnsupportedError struct {
	Format  string
	Version string
}

func (e *UnsupportedError) Error() string {
	return fmt.Sprintf("ras: unsupported %s format or NCCL version %q", e.Format, e.Version)
}

// TextStatus is a bounded NCCL 2.24.3 uniform-job text observation. The profile
// has constructed-fixture coverage; authentic capture validation is pending.
// Individual rank identities, timestamps and collection timeout
// counts are unavailable. Raw retains the exact response. Errors and Warnings
// retain uninterpreted section lines; empty sections are not a health verdict.
type TextStatus struct {
	Version                          string
	CompiledCUDAMajor                int
	CompiledCUDAMinor                int
	CUDARuntime                      int
	CUDADriver                       int
	Nodes, Processes, GPUs           int
	ProcessesPerNode, GPUsPerProcess int
	CollectionSeconds                float64
	Groups                           []TextGroup
	Errors, Warnings                 []string
	Raw                              []byte
}

// TextGroup is one source-reported aggregate row, not individual communicators
// or ranks. RanksInGroup counts unique GPUs and need not equal Comms*RanksPerComm.
type TextGroup struct {
	Index, Comms, NodesPerComm, RanksPerNodeMin, RanksPerNodeMax int
	RanksPerComm, RanksInGroup                                   int
	Status, Errors                                               string
}

var textVersion = regexp.MustCompile(`^NCCL version ([0-9]+\.[0-9]+\.[0-9]+(?:[+_-][A-Za-z0-9.+_-]+)?) compiled with CUDA ([0-9]+)\.([0-9]+)$`)
var textCUDA = regexp.MustCompile(`^CUDA runtime version ([0-9]+), driver version ([0-9]+)$`)
var textDuration = regexp.MustCompile(`^Communicators\.\.\. \(([0-9]+\.[0-9]{2})s\)$`)

// ReadText reads one complete legacy text STATUS response. The experimental
// profile is NCCL 2.24.3's uniform job and aggregate communicator tables;
// authentic capture validation remains pending.
// Nonuniform jobs and other versions/formats return UnsupportedError. Detailed
// error/warning sections are retained as raw lines, not decoded into rank data.
// Records bounds groups and retained diagnostic lines together; section headings
// and separators do not count as diagnostic records.
// The caller must arrange cancellation for an io.Reader that can block.
func ReadText(r io.Reader, limits Limits) (*TextStatus, error) {
	l, err := limits.checked()
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, errors.New("ras text: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(l.Bytes)+1))
	if err != nil {
		return nil, fmt.Errorf("ras text read: %w", err)
	}
	if len(b) > l.Bytes {
		return nil, errors.New("ras text: byte limit exceeded")
	}
	if len(b) == 0 || b[len(b)-1] != '\n' {
		return nil, errors.New("ras text: incomplete response")
	}
	lines := strings.Split(string(b[:len(b)-1]), "\n")
	for _, line := range lines {
		if len(line) > l.Line {
			return nil, errors.New("ras text: line limit exceeded")
		}
		if strings.ContainsRune(line, '\r') || strings.ContainsRune(line, '\x00') {
			return nil, errors.New("ras text: invalid line framing")
		}
	}
	s := &TextStatus{Raw: bytes.Clone(b)}
	unsupported := func() (*TextStatus, error) { return nil, &UnsupportedError{Format: "text", Version: s.Version} }
	match := textVersion.FindStringSubmatch(lines[0])
	if match == nil {
		return unsupported()
	}
	s.Version = match[1]
	if strings.SplitN(s.Version, "+", 2)[0] != "2.24.3" {
		return unsupported()
	}
	number := func(value string) (int, error) {
		n, err := strconv.ParseUint(value, 10, 31)
		if err != nil || n == 0 {
			return 0, errors.New("ras text: positive bounded count required")
		}
		return int(n), nil
	}
	if s.CompiledCUDAMajor, err = number(match[2]); err != nil {
		return nil, err
	}
	minor, e := strconv.ParseUint(match[3], 10, 31)
	if e != nil {
		return nil, e
	}
	s.CompiledCUDAMinor = int(minor)
	if len(lines) < 16 {
		return nil, errors.New("ras text: truncated tables")
	}
	match = textCUDA.FindStringSubmatch(lines[1])
	if match == nil {
		return nil, errors.New("ras text: CUDA version preamble required")
	}
	if s.CUDARuntime, err = number(match[1]); err != nil {
		return nil, err
	}
	if s.CUDADriver, err = number(match[2]); err != nil {
		return nil, err
	}
	if lines[2] != "" || lines[3] != "Job summary" || lines[4] != "===========" || lines[5] != "" {
		return unsupported()
	}
	if lines[6] != "  Nodes  Processes         GPUs  Processes     GPUs" || lines[7] != "(total)   per node  per process    (total)  (total)" {
		return unsupported()
	}
	fields := strings.Fields(lines[8])
	if len(fields) != 5 {
		return nil, errors.New("ras text: uniform job count row required")
	}
	counts := []*int{&s.Nodes, &s.ProcessesPerNode, &s.GPUsPerProcess, &s.Processes, &s.GPUs}
	for i, p := range counts {
		if *p, err = number(fields[i]); err != nil {
			return nil, err
		}
	}
	if int64(s.Nodes)*int64(s.ProcessesPerNode) != int64(s.Processes) || int64(s.Processes)*int64(s.GPUsPerProcess) != int64(s.GPUs) {
		return nil, errors.New("ras text: contradictory job counts")
	}
	if lines[9] != "" {
		return unsupported()
	}
	match = textDuration.FindStringSubmatch(lines[10])
	if match == nil {
		return unsupported()
	}
	s.CollectionSeconds, err = strconv.ParseFloat(match[1], 64)
	if err != nil {
		return nil, err
	}
	if lines[11] != "=============" || lines[12] != "" || lines[13] != "Group     Comms     Nodes     Ranks     Ranks     Ranks    Status  Errors" || lines[14] != "    #  in group  per comm  per node  per comm  in group" {
		return unsupported()
	}
	i := 15
	if lines[i] == "No communicator data collected!" {
		i++
	} else {
		for i < len(lines) && lines[i] != "" {
			if len(s.Groups) >= l.Records {
				return nil, errors.New("ras text: record limit exceeded")
			}
			f := strings.Fields(lines[i])
			if len(f) != 8 {
				return unsupported()
			}
			g := TextGroup{Status: f[6], Errors: f[7]}
			index, e := strconv.ParseUint(f[0], 10, 31)
			if e != nil || int(index) != len(s.Groups) {
				return nil, errors.New("ras text: group index sequence required")
			}
			g.Index = int(index)
			for j, p := range []*int{&g.Comms, &g.NodesPerComm, &g.RanksPerComm, &g.RanksInGroup} {
				k := []int{1, 2, 4, 5}[j]
				if *p, err = number(f[k]); err != nil {
					return nil, err
				}
			}
			rangeParts := strings.Split(f[3], "-")
			if len(rangeParts) > 2 {
				return unsupported()
			}
			if g.RanksPerNodeMin, err = number(rangeParts[0]); err != nil {
				return nil, err
			}
			g.RanksPerNodeMax = g.RanksPerNodeMin
			if len(rangeParts) == 2 {
				if g.RanksPerNodeMax, err = number(rangeParts[1]); err != nil {
					return nil, err
				}
			}
			if g.RanksPerNodeMin > g.RanksPerNodeMax || g.RanksPerNodeMax > g.RanksPerComm || g.NodesPerComm > s.Nodes || g.RanksInGroup > s.GPUs {
				return nil, errors.New("ras text: contradictory group counts")
			}
			if !textOneOf(g.Status, "RUNNING", "INIT", "FINALIZE", "ABORT", "UNKNOWN") || !textOneOf(g.Errors, "OK", "MISMATCH", "INCOMPLETE", "ERROR", "INCOMPLETE,ERROR", "ERROR,MISMATCH", "INCOMPLETE,MISMATCH", "INCOMPLETE,ERROR,MISMATCH") {
				return unsupported()
			}
			s.Groups = append(s.Groups, g)
			i++
		}
	}
	if i+3 >= len(lines) || lines[i] != "" || lines[i+1] != "Errors" || lines[i+2] != "======" || lines[i+3] != "" {
		return nil, errors.New("ras text: errors section missing")
	}
	i += 4
	start := i
	for i < len(lines) && lines[i] != "Warnings" {
		i++
	}
	if i+2 >= len(lines) || lines[i+1] != "========" || lines[i+2] != "" {
		return nil, errors.New("ras text: warnings section missing")
	}
	if len(s.Groups)+(i-start)+(len(lines)-(i+3)) > l.Records {
		return nil, errors.New("ras text: diagnostic line limit exceeded")
	}
	s.Errors = append([]string(nil), lines[start:i]...)
	s.Warnings = append([]string(nil), lines[i+3:]...)
	return s, nil
}

func textOneOf(value string, values ...string) bool {
	for _, v := range values {
		if value == v {
			return true
		}
	}
	return false
}
