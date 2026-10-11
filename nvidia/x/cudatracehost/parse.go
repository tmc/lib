package cudatracehost

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var traceLine = regexp.MustCompile(`^\s*.*-(\d+)\s+\[\d+\]\s+\S+\s+(\d+)\.(\d+):\s+(\S+):\s*(.*)$`)

// Observation preserves the original kernel line and parsed clock/TID. Args
// use entry ABI slots; Return is only known on an observed return event.
type Observation struct {
	ThreadGenerationKnown bool
	ExecID                uint64
	ThreadGeneration      uint64
	Raw                   string
	TID                   uint32
	Timestamp             uint64
	Name                  string
	Args                  [11]uint64
	ArgsKnown             [11]bool
	Return                int32
	ReturnKnown           bool
}

// Parse parses the bounded text tracefs transport. It rejects truncated clocks,
// malformed numbers and unknown fields rather than inventing zero values.
func Parse(line string) (Observation, error) {
	o := Observation{Raw: line}
	if len(line) > 65536 {
		return o, fmt.Errorf("cudatracehost: trace line exceeds limit")
	}
	m := traceLine.FindStringSubmatch(line)
	if m == nil {
		return o, fmt.Errorf("cudatracehost: invalid trace line")
	}
	tid, e := strconv.ParseUint(m[1], 10, 32)
	if e != nil || tid == 0 {
		return o, fmt.Errorf("cudatracehost: invalid tid")
	}
	o.TID = uint32(tid)
	sec, e := strconv.ParseUint(m[2], 10, 64)
	if e != nil || sec > ^uint64(0)/1000000000 {
		return o, fmt.Errorf("cudatracehost: invalid timestamp")
	}
	frac := m[3]
	if len(frac) > 9 {
		return o, fmt.Errorf("cudatracehost: timestamp precision exceeds nanoseconds")
	}
	frac += strings.Repeat("0", 9-len(frac))
	ns, e := strconv.ParseUint(frac, 10, 64)
	if e != nil || sec*1000000000 > ^uint64(0)-ns {
		return o, fmt.Errorf("cudatracehost: timestamp overflow")
	}
	o.Timestamp = sec*1000000000 + ns
	o.Name = m[4]
	for _, field := range strings.Fields(m[5]) {
		k, v, ok := strings.Cut(field, "=")
		if !ok {
			continue
		} // tracefs address annotation
		if k == "ret" {
			if o.ReturnKnown {
				return o, fmt.Errorf("cudatracehost: duplicate return")
			}
			n, err := strconv.ParseUint(v, 0, 64)
			if err != nil {
				return o, fmt.Errorf("cudatracehost: invalid return: %w", err)
			}
			o.Return = int32(n)
			o.ReturnKnown = true
			continue
		}
		if strings.HasPrefix(k, "a") {
			i, err := strconv.Atoi(k[1:])
			if err != nil || i < 0 || i >= len(o.Args) {
				return o, fmt.Errorf("cudatracehost: invalid argument %s", k)
			}
			n, err := strconv.ParseUint(v, 0, 64)
			if err != nil {
				return o, fmt.Errorf("cudatracehost: invalid argument: %w", err)
			}
			if o.ArgsKnown[i] {
				return o, fmt.Errorf("cudatracehost: duplicate argument")
			}
			o.Args[i] = n
			o.ArgsKnown[i] = true
		}
	}
	return o, nil
}
