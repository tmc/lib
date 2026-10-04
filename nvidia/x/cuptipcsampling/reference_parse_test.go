package cuptipcsampling_test

import (
	"fmt"
	"math/big"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

type pcReferenceReport struct {
	Total   uint64 // User samples; Offered includes non-user and dropped samples.
	Offered uint64
	Dropped uint64
	NonUser uint64
	Counts  []uint64
	Kernels []string
	Reasons []string
}

var (
	pcRangeStart = "----- PC sampling data for range defined by cuptiPCSamplingStart() and cuptiPCSamplingStop() -----"
	pcSummary    = regexp.MustCompile(`^Number of PCs remaining to be collected: [0-9]+, range id: [0-9]+, total samples: ([0-9]+), dropped samples: ([0-9]+), non user kernels total samples: ([0-9]+)$`)
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
		for _, match := range pcReason.FindAllStringSubmatch(line, -1) {
			if len(match) == 2 {
				reasons[strings.TrimSpace(match[1])] = true
			}
		}
		if strings.HasPrefix(line, "Number of PCs remaining to be collected:") {
			match := pcSummary.FindStringSubmatch(line)
			if len(match) != 4 {
				return report, fmt.Errorf("malformed PC sampling summary: %q", line)
			}
			var fields [3]uint64
			for i := range fields {
				n, err := strconv.ParseUint(match[i+1], 10, 64)
				if err != nil {
					return report, fmt.Errorf("invalid PC sampling count %q", match[i+1])
				}
				fields[i] = n
			}
			offered, dropped, nonUser := fields[0], fields[1], fields[2]
			user, err := pcReferenceUserSamples(offered, dropped, nonUser)
			if err != nil {
				return report, err
			}
			for _, pair := range [][2]uint64{{report.Total, user}, {report.Offered, offered}, {report.Dropped, dropped}, {report.NonUser, nonUser}} {
				if ^uint64(0)-pair[0] < pair[1] {
					return report, fmt.Errorf("PC sampling population sum overflow")
				}
			}
			report.Total += user
			report.Offered += offered
			report.Dropped += dropped
			report.NonUser += nonUser
			report.Counts = append(report.Counts, user)
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
, pcOffset : 0x80, stallReasonCount: 3, functionName: _Z9VectorAddPKiS0_Pii, correlationID: 2, stallReason: reason_a, samples: 5, stallReason: reason_b, samples: 1, stallReason: reason_c, samples: 2
, pcOffset : 0x90, stallReasonCount: 2, functionName: _Z14VectorSubtractPKiS0_Pii, correlationID: 3, stallReason: reason_d, samples: 4, stallReason: reason_e, samples: 1
Number of PCs remaining to be collected: 0, range id: 3, total samples: 99, dropped samples: 0, non user kernels total samples: 90
----- PC sampling data for range defined by cuptiPCSamplingStart() and cuptiPCSamplingStop() -----
, pcOffset : 0x20, stallReasonCount: 1, functionName: _Z14VectorMultiplyPKiS0_Pii, correlationID: 4, stallReason: reason_f, samples: 2
Number of PCs remaining to be collected: 0, range id: 4, total samples: 122, dropped samples: 1, non user kernels total samples: 110
`
	report, err := parsePCReferenceOutput(output)
	if err != nil {
		t.Fatal(err)
	}
	if report.Total != 20 || report.Offered != 221 || report.Dropped != 1 || report.NonUser != 200 || len(report.Counts) != 2 {
		t.Fatalf("user totals = %d %v, want 20 from two ranges", report.Total, report.Counts)
	}
	if strings.Join(report.Kernels, ",") != "_Z14VectorMultiplyPKiS0_Pii,_Z14VectorSubtractPKiS0_Pii,_Z9VectorAddPKiS0_Pii" || strings.Join(report.Reasons, ",") != "reason_a,reason_b,reason_c,reason_d,reason_e,reason_f" {
		t.Fatalf("inventory = kernels %v reasons %v", report.Kernels, report.Reasons)
	}
}

func TestParsePCReferenceOutputRejectsTruncatedRange(t *testing.T) {
	if _, err := parsePCReferenceOutput(pcRangeStart + "\npcOffset : 0x10, functionName: VectorAdd\n"); err == nil {
		t.Fatal("truncated PC range accepted")
	}
}

func TestParsePCReferenceOutputRejectsTruncatedRangeBeforeCompleteRange(t *testing.T) {
	output := pcRangeStart + `
, pcOffset : 0x10, stallReasonCount: 1, functionName: _Z9VectorAddPKiS0_Pii, correlationID: 1, stallReason: reason_a, samples: 1
` + pcRangeStart + `
, pcOffset : 0x20, stallReasonCount: 1, functionName: _Z9VectorAddPKiS0_Pii, correlationID: 1, stallReason: reason_b, samples: 1
Number of PCs remaining to be collected: 0, range id: 2, total samples: 1, dropped samples: 0, non user kernels total samples: 0
`
	if _, err := parsePCReferenceOutput(output); err == nil || !strings.Contains(err.Error(), "truncated PC range") {
		t.Fatalf("truncated first range error = %v, want truncation error", err)
	}
}

func TestParsePCReferenceOutputRejectsTruncatedFinalRangeAfterCompleteRange(t *testing.T) {
	output := pcRangeStart + `
, pcOffset : 0x10, stallReasonCount: 1, functionName: _Z9VectorAddPKiS0_Pii, correlationID: 1, stallReason: reason_a, samples: 1
` + pcSummaryLine("1", 1) + pcRangeStart + `
, pcOffset : 0x20, stallReasonCount: 1, functionName: _Z9VectorAddPKiS0_Pii, correlationID: 2, stallReason: reason_b, samples: 1
`
	if _, err := parsePCReferenceOutput(output); err == nil || !strings.Contains(err.Error(), "truncated PC range") {
		t.Fatalf("truncated final range error = %v, want truncation error", err)
	}
}

func TestParsePCReferenceOutputRejectsSampleTotalOverflow(t *testing.T) {
	for _, total := range []string{
		"18446744073709551616",
		"18446744073709551615", // first range; the next range overflows the running sum
	} {
		t.Run(total, func(t *testing.T) {
			output := pcRangeStart + "\n" + pcSummaryLine(total, 1)
			if total == "18446744073709551615" {
				output += pcRangeStart + "\n" + pcSummaryLine("1", 2)
			}
			if _, err := parsePCReferenceOutput(output); err == nil {
				t.Fatalf("overflowing total %s accepted", total)
			}
		})
	}
}

func pcSummaryLine(total string, rangeID int) string {
	return fmt.Sprintf("Number of PCs remaining to be collected: 0, range id: %d, total samples: %s, dropped samples: 0, non user kernels total samples: 0\n", rangeID, total)
}

func TestParsePCReferenceOutputPopulationAccounting(t *testing.T) {
	for _, tc := range []struct {
		name                      string
		offered, dropped, nonUser string
		wantUser                  uint64
		wantErr                   bool
	}{
		{"mixed", "1000", "900", "70", 30, false},
		{"all_non_user", "100", "0", "100", 0, false},
		{"all_dropped", "100", "100", "0", 0, false},
		{"dropped_exceeds_total", "10", "11", "0", 0, true},
		{"populations_exceed_total", "10", "6", "5", 0, true},
		{"dropped_overflow", "10", "18446744073709551616", "0", 0, true},
		{"non_user_overflow", "10", "0", "18446744073709551616", 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := pcRangeStart + "\n" + fmt.Sprintf("Number of PCs remaining to be collected: 0, range id: 1, total samples: %s, dropped samples: %s, non user kernels total samples: %s\n", tc.offered, tc.dropped, tc.nonUser)
			got, err := parsePCReferenceOutput(out)
			if (err != nil) != tc.wantErr {
				t.Fatalf("error = %v, want error %v", err, tc.wantErr)
			}
			if err == nil && (got.Total != tc.wantUser || got.Total+got.Dropped+got.NonUser != got.Offered) {
				t.Fatalf("population accounting = %+v, want user %d", got, tc.wantUser)
			}
		})
	}
}

// CUpti_PCSamplingData.totalSamples includes dropped and non-user samples.
// The remaining user population may still include unnamed PCs.
func pcReferenceUserSamples(offered, dropped, nonUser uint64) (uint64, error) {
	if dropped > offered || nonUser > offered-dropped {
		return 0, fmt.Errorf("PC sampling populations exceed offered total")
	}
	return offered - dropped - nonUser, nil
}

func FuzzPCReferencePopulationAccounting(f *testing.F) {
	f.Add(uint64(1000), uint64(900), uint64(70))
	f.Add(uint64(0), uint64(0), uint64(0))
	f.Add(^uint64(0), ^uint64(0), uint64(1))
	f.Fuzz(func(t *testing.T, offered, dropped, nonUser uint64) {
		out := pcRangeStart + "\n" + fmt.Sprintf("Number of PCs remaining to be collected: 0, range id: 1, total samples: %d, dropped samples: %d, non user kernels total samples: %d\n", offered, dropped, nonUser)
		got, err := parsePCReferenceOutput(out)
		// Arbitrary precision avoids sharing the parser's subtraction checks.
		sub := new(big.Int).Add(new(big.Int).SetUint64(dropped), new(big.Int).SetUint64(nonUser))
		want := new(big.Int).Sub(new(big.Int).SetUint64(offered), sub)
		if want.Sign() < 0 {
			if err == nil {
				t.Fatalf("contradictory populations accepted: %+v", got)
			}
			return
		}
		if err != nil || got.Total != want.Uint64() || got.Offered != offered || got.Dropped != dropped || got.NonUser != nonUser {
			t.Fatalf("populations = %+v, %v; want user %s", got, err, want)
		}
	})
}
