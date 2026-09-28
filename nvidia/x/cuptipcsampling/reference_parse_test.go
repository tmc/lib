package cuptipcsampling_test

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

type pcReferenceReport struct {
	Total   uint64
	Counts  []uint64
	Kernels []string
	Reasons []string
}

var (
	pcRangeStart = "----- PC sampling data for range defined by cuptiPCSamplingStart() and cuptiPCSamplingStop() -----"
	pcSummary    = regexp.MustCompile(`^Number of PCs remaining to be collected: [0-9]+, range id: [0-9]+, total samples: ([0-9]+), dropped samples: [0-9]+, non user kernels total samples: [0-9]+$`)
	pcKernel     = regexp.MustCompile(`functionName: ([^,]+)`)
	pcReason     = regexp.MustCompile(`stallReason: ([^,]+)`)
)

func parsePCReferenceOutput(output string) (pcReferenceReport, error) {
	var report pcReferenceReport
	kernels, reasons := map[string]bool{}, map[string]bool{}
	active, hasTotal := false, false
	for _, raw := range strings.Split(output, "\n") {
		line := strings.TrimSpace(raw)
		if line == pcRangeStart {
			if active && !hasTotal {
				return report, fmt.Errorf("truncated PC range: missing total samples line")
			}
			active, hasTotal = true, false
			continue
		}
		if !active {
			continue
		}
		if match := pcKernel.FindStringSubmatch(line); len(match) == 2 {
			kernels[strings.TrimSpace(match[1])] = true
		}
		if match := pcReason.FindStringSubmatch(line); len(match) == 2 {
			reasons[strings.TrimSpace(match[1])] = true
		}
		if strings.HasPrefix(line, "Number of PCs remaining to be collected:") {
			match := pcSummary.FindStringSubmatch(line)
			if len(match) != 2 {
				return report, fmt.Errorf("malformed PC sampling summary: %q", line)
			}
			count, err := strconv.ParseUint(match[1], 10, 64)
			if err != nil || ^uint64(0)-report.Total < count {
				return report, fmt.Errorf("invalid PC user sample total %q", match[1])
			}
			report.Total += count
			report.Counts = append(report.Counts, count)
			hasTotal = true
		}
	}
	if active && !hasTotal {
		return report, fmt.Errorf("truncated PC range: missing total samples line")
	}
	if len(report.Counts) == 0 {
		return report, fmt.Errorf("reference output contains no complete PC sampling ranges")
	}
	for name := range kernels {
		report.Kernels = append(report.Kernels, name)
	}
	for name := range reasons {
		report.Reasons = append(report.Reasons, name)
	}
	sort.Strings(report.Kernels)
	sort.Strings(report.Reasons)
	return report, nil
}

func TestParsePCReferenceOutputExactSampleFormat(t *testing.T) {
	const output = `----- PC sampling data for range defined by cuptiPCSamplingStart() and cuptiPCSamplingStop() -----
pcOffset : 0x10, stallReasonCount: 1, functionName: VectorAdd, correlationID: 1, stallReason: long_scoreboard, samples: 7
Number of PCs remaining to be collected: 0, range id: 3, total samples: 9, dropped samples: 0, non user kernels total samples: 90
----- PC sampling data for range defined by cuptiPCSamplingStart() and cuptiPCSamplingStop() -----
pcOffset : 0x20, stallReasonCount: 1, functionName: VectorSubtract, correlationID: 2, stallReason: short_scoreboard, samples: 2
Number of PCs remaining to be collected: 0, range id: 4, total samples: 11, dropped samples: 1, non user kernels total samples: 110
`
	report, err := parsePCReferenceOutput(output)
	if err != nil {
		t.Fatal(err)
	}
	if report.Total != 20 || len(report.Counts) != 2 {
		t.Fatalf("user totals = %d %v, want 20 from two ranges", report.Total, report.Counts)
	}
	if strings.Join(report.Kernels, ",") != "VectorAdd,VectorSubtract" || strings.Join(report.Reasons, ",") != "long_scoreboard,short_scoreboard" {
		t.Fatalf("inventory = kernels %v reasons %v", report.Kernels, report.Reasons)
	}
}

func TestParsePCReferenceOutputRejectsTruncatedRange(t *testing.T) {
	if _, err := parsePCReferenceOutput(pcRangeStart + "\npcOffset : 0x10, functionName: VectorAdd\n"); err == nil {
		t.Fatal("truncated PC range accepted")
	}
}
