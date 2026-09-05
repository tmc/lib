package cudasanitizer

import (
	"math"
	"regexp"
	"strconv"
	"strings"
)

// logPrefix is the marker compute-sanitizer puts at the start of every line it
// writes, which is how its report is told apart from the application's own
// output when the two share a stream.
const logPrefix = "========="

// Finding is one report block from a compute-sanitizer log: a headline such as
// "Invalid __global__ write of size 4 bytes" followed by its indented detail
// lines.
type Finding struct {
	Headline string   // first line of the block, with the log prefix removed
	Detail   []string // remaining lines of the block, with the log prefix removed
}

// String returns the finding's headline and detail as the tool printed them,
// without the log prefix.
func (f Finding) String() string {
	var b strings.Builder
	b.WriteString(f.Headline)
	for _, d := range f.Detail {
		b.WriteString("\n")
		b.WriteString(d)
	}
	return b.String()
}

// Log is a parsed compute-sanitizer report.
//
// The zero Log means "no report was read", which is not the same as a report
// of zero errors: see [Log.SummaryFound].
type Log struct {
	// Text is the report as written by the tool.
	Text string

	// SummaryFound reports whether the report ended with a summary line.
	// Its absence means the tool did not reach the end of a run, so the
	// report carries no verdict at all.
	SummaryFound bool

	// Errors is the count from "ERROR SUMMARY: N errors", or from
	// racecheck's error count when only a RACECHECK SUMMARY was printed. It
	// is meaningful only when SummaryFound is true.
	//
	// A report can carry more than one summary: compute-sanitizer prints one
	// per process, so a run that forks, or two runs concatenated into one
	// file, ends with several. Every count in the text is added in. Taking
	// only the last would let a second process's clean summary erase the
	// first's errors, which is the one direction this type must never get
	// wrong.
	Errors int

	// Summaries is the number of summary lines the report contained. It is
	// 1 for an ordinary single-process run; more means Errors is a total
	// over several, and callers reporting a per-run figure should say so.
	Summaries int

	// RaceErrors preserves the error count across all RACECHECK SUMMARY lines,
	// including when ERROR SUMMARY lines are also present. The two families
	// may overlap, so callers must not add RaceErrors to Errors.
	RaceErrors int

	// Hazards and Warnings come from racecheck's
	// "RACECHECK SUMMARY: N hazards displayed (E errors, W warnings)".
	Hazards  int
	Warnings int

	// LeakedBytes and LeakedAllocations come from memcheck's
	// "LEAK SUMMARY: N bytes leaked in M allocations", printed under
	// --leak-check full. Leaks also count toward Errors.
	LeakedBytes       int64
	LeakedAllocations int

	// Findings are the individual report blocks, in the order printed.
	Findings []Finding
}

var (
	errorSummaryRE = regexp.MustCompile(`^ERROR SUMMARY: (\d+) error`)
	leakSummaryRE  = regexp.MustCompile(`^LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocation`)
	raceSummaryRE  = regexp.MustCompile(`^RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)`)
)

// ParseLog parses a compute-sanitizer report.
//
// It reads only lines carrying the tool's "=========" prefix, so text
// interleaved from the application under test is ignored and a combined stream
// can be passed in directly. Blocks are delimited by the bare prefix lines the
// tool prints between them.
//
// Counts saturate at the largest representable integer instead of wrapping.
// Counts accumulate across every summary in the text rather than being
// replaced by the last one; see [Log.Errors].
//
// ParseLog never fails: text it does not recognize simply does not populate a
// field. Callers must therefore check [Log.SummaryFound] before reading
// [Log.Errors], since an unparsed report and a clean report both leave Errors
// at zero.
func ParseLog(text string) *Log {
	l := &Log{Text: text}
	var block []string
	var errorSummaries, raceSummaries, raceErrorTotal int
	flush := func() {
		if len(block) > 0 {
			l.Findings = append(l.Findings, Finding{Headline: block[0], Detail: block[1:]})
			block = nil
		}
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRight(line, " \t\r")
		if !strings.HasPrefix(line, logPrefix) {
			continue
		}
		body := strings.TrimPrefix(line, logPrefix)
		if body != "" && !strings.HasPrefix(body, " ") {
			// A longer rule of equals signs, not a report line.
			continue
		}
		body = strings.TrimPrefix(body, " ")
		if strings.TrimSpace(body) == "" {
			flush()
			continue
		}
		switch {
		case body == "COMPUTE-SANITIZER":
			flush()
		case errorSummaryRE.MatchString(body):
			flush()
			m := errorSummaryRE.FindStringSubmatch(body)
			n, _ := strconv.Atoi(m[1])
			l.Errors = addCount(l.Errors, n)
			errorSummaries++
			l.SummaryFound = true
		case leakSummaryRE.MatchString(body):
			flush()
			m := leakSummaryRE.FindStringSubmatch(body)
			bytes, _ := strconv.ParseInt(m[1], 10, 64)
			allocations, _ := strconv.Atoi(m[2])
			if bytes > math.MaxInt64-l.LeakedBytes {
				l.LeakedBytes = math.MaxInt64
			} else {
				l.LeakedBytes += bytes
			}
			l.LeakedAllocations = addCount(l.LeakedAllocations, allocations)
		case raceSummaryRE.MatchString(body):
			flush()
			m := raceSummaryRE.FindStringSubmatch(body)
			hazards, _ := strconv.Atoi(m[1])
			raceErrors, _ := strconv.Atoi(m[2])
			warnings, _ := strconv.Atoi(m[3])
			l.Hazards = addCount(l.Hazards, hazards)
			l.Warnings = addCount(l.Warnings, warnings)
			raceErrorTotal = addCount(raceErrorTotal, raceErrors)
			raceSummaries++
			l.SummaryFound = true
		default:
			block = append(block, body)
		}
	}
	flush()
	// racecheck prints both an ERROR SUMMARY and a RACECHECK SUMMARY per
	// process, so count only one family of them.
	l.RaceErrors = raceErrorTotal
	l.Summaries = errorSummaries
	if errorSummaries == 0 {
		l.Summaries = raceSummaries
		// Older racecheck output ends with a RACECHECK SUMMARY instead of an
		// ERROR SUMMARY. Its error count is the verdict only when no
		// ERROR SUMMARY appeared anywhere, so that the two are never added
		// together for the same run.
		l.Errors = raceErrorTotal
	}
	return l
}

// Counts are nonnegative decimal integers. strconv clamps an out-of-range
// positive input; accumulation must preserve that bound too.
func addCount(a, b int) int {
	if b > math.MaxInt-a {
		return math.MaxInt
	}
	return a + b
}
