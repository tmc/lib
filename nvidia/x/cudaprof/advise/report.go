package advise

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/ianlancetaylor/demangle"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

// occupancyCaveat qualifies every recommendation derived from an occupancy or
// register-residency figure. Occupancy states what a kernel CAN keep resident,
// never what it is waiting on. Raising it pays only when the kernel is
// latency-bound, and it can run backwards when the transformation that frees the
// resource costs something else: deleting the shared-memory staging in
// ds4_f16_pair_ordered_chunks raised occupancy from 10.4% to 100% and measured
// 2.84% slower end to end, because the staging was buying coalescing. Advice
// built on a capacity figure is a hypothesis to A/B, not an instruction.
//
// It replaces three ad-hoc hedges that said less in four different spellings.
// Sites where the figure is NOT capacity-read-as-throughput are deliberately
// left bare and are pinned bare by tests: a hardware legality maximum (block
// size > 1024), grid starvation (SMs are literally idle, which is an occupancy
// figure read AS occupancy), and register spills (real work, not residency).
const occupancyCaveat = " Occupancy bounds residency, not throughput: A/B this. It pays only if the kernel is latency-bound (a bandwidth-bound streaming kernel will see no speedup), and it can regress when freeing the resource costs coalescing."

// Severity denotes what a reader should do with an advice finding.
//
// These are not defect levels. This report ranks where a captured span went
// and proposes leads; almost nothing in it is broken. "Critical" borrowed from
// linters and incident tooling asserted an urgency the analysis had not earned,
// and it read as alarm on findings that were merely large -- a kernel family
// holding most of the device time is what a well-tuned program looks like.
// How much to trust a finding is EvidenceLevel's job, and how much it is worth
// is the magnitude in the header; Severity answers only "in what order, and
// does this change how I read the rest".
type Severity string

const (
	// SeverityReadFirst marks a finding that changes how the rest of the
	// report should be read: the capture may not be usable, or the ordering
	// below it ranks a quantity that is not the one that matters. It is
	// earned by kind, never by cost, because the reader's first position is
	// worth more than the largest item in a list.
	SeverityReadFirst Severity = "Read first"

	// SeverityCost ranks where the captured time went. A finding here is a
	// measurement or a lead, not a problem: read the magnitude in the header
	// for how much it is worth and EvidenceLevel for how much it is known.
	SeverityCost Severity = "Cost"

	// SeverityContext carries findings too small to act on alone and
	// observations that bound the others.
	SeverityContext Severity = "Context"
)

// EvidenceLevel classifies the empirical basis of an advice finding.
type EvidenceLevel string

const (
	// EvidenceObserved indicates direct native hardware metrics, compiler diagnostics, or measured timings.
	EvidenceObserved EvidenceLevel = "observed"
	// EvidenceSuspected indicates static opcode heuristics, correlated PC samples without deciding transaction counters, or resource capacity bounds.
	EvidenceSuspected EvidenceLevel = "suspected"
	// EvidenceUnmeasured indicates missing required telemetry or skipped audits due to absent data.
	EvidenceUnmeasured EvidenceLevel = "unmeasured"
)

// Category classifies the optimization or architecture area of an advice finding.
type Category string

const (
	CategoryMemory               Category = "Memory"
	CategoryCompute              Category = "Compute"
	CategoryRegisterPressure     Category = "RegisterPressure"
	CategoryArchitecture         Category = "Architecture"
	CategoryArchitectureCompute  Category = "Architecture/Compute"
	CategoryArchitectureDispatch Category = "Architecture/Dispatch"
	CategoryMemoryCompute        Category = "Memory/Compute"
	CategoryDispatch             Category = "Dispatch"

	// CategoryKernel holds several patterns found in one kernel's code.
	CategoryKernel Category = "Kernel"
)

// Location specifies the source or machine code position associated with an advice item.
type Location struct {
	Kernel string `json:"kernel,omitempty"`

	// Module is the captured image the kernel's code was read from, set
	// when a kernel of this name ran from more than one module. Each is a
	// different kernel, charged the share of the name's time the journal
	// records from its module.
	Module string `json:"module,omitempty"`

	PC   uint64 `json:"pc,omitempty"`
	File string `json:"file,omitempty"`
	Line int    `json:"line,omitempty"`

	// CallSite is the host call site of a host API finding, or UNREAD with
	// the reason when the capture does not hold one.
	CallSite string `json:"call_site,omitempty"`

	// Source is UNREAD with the reason when the kernel's code carried a
	// source location that is not provably from the compilation the kernel
	// ran, and File and Line were cleared for it.
	Source string `json:"source,omitempty"`
}

func (l Location) String() string {
	var parts []string
	if l.Kernel != "" {
		parts = append(parts, fmt.Sprintf("kernel %s", l.Kernel))
	}
	if l.Module != "" {
		parts = append(parts, "module "+l.Module)
	}
	if l.File != "" {
		if l.Line > 0 {
			parts = append(parts, fmt.Sprintf("%s:%d", l.File, l.Line))
		} else {
			parts = append(parts, l.File)
		}
	}
	if l.Source != "" {
		parts = append(parts, "source "+l.Source)
	}
	if l.PC > 0 {
		parts = append(parts, fmt.Sprintf("PC 0x%04x", l.PC))
	}
	if l.CallSite != "" {
		parts = append(parts, "call site "+l.CallSite)
	}
	return strings.Join(parts, ", ")
}

// kernelKey returns the key of the kernel l names in kernel times: its name,
// qualified by its module when it has one.
func (l Location) kernelKey() string {
	if l.Module == "" {
		return l.Kernel
	}
	return l.Kernel + "\x00" + l.Module
}

// Item represents a single diagnosis or performance recommendation.
type Item struct {
	// ClassifiedShare preserves the evidence class and denominator basis when
	// WithShareEvidence admits a source-bound profile share. Analyze supplies
	// explicit Unread when identity or population evidence is missing. Nil
	// remains possible for Items constructed directly by callers.
	ClassifiedShare          *cudaevidence.Value `json:"classified_share,omitempty"`
	Severity                 Severity            `json:"severity"`
	Category                 Category            `json:"category"`
	EvidenceLevel            EvidenceLevel       `json:"evidence_level,omitempty"`
	Location                 Location            `json:"location"`
	Finding                  string              `json:"finding"`
	ActionableRecommendation string              `json:"actionable_recommendation"`
	NextMeasurement          string              `json:"next_measurement,omitempty"`
	DeviceTimeNanos          int64               `json:"device_time_nanos,omitempty"`

	// HostTimeNanos is host time a finding accounts for: time a host thread
	// spent inside API calls, which no kernel profile holds. It is kept apart
	// from DeviceTimeNanos so neither is read as the other, and the ranking
	// orders a finding by the larger of the two.
	HostTimeNanos  int64   `json:"host_time_nanos,omitempty"`
	ShareOfProfile float64 `json:"share_of_profile,omitempty"`

	// Patterns are the findings in one kernel's code that a CategoryKernel
	// item holds. They carry no time of their own: the kernel's time is the
	// item's, counted once.
	Patterns []Item `json:"patterns,omitempty"`

	// Gap is what the finding could explain of the gap to a base capture,
	// or nil when there is no base or no term of the comparison it moves.
	Gap *GapBound `json:"gap,omitempty"`

	// terms names the comparison terms this finding moves, when its category
	// and location do not say; see gapBound.
	terms []string

	// magnitude overrides the parenthetical the markdown header prints for
	// this item. The default reads "(N% of GPU time, M ms)", which is a claim
	// about device work: a finding whose numbers are idle time renders there
	// as though the idle were work a kernel did. An item whose numbers mean
	// something else says what they mean here.
	magnitude string

	// rule names the audit rule that produced the item, when that rule can
	// fire once per kernel or kernel pair; see groupRules. Not serialized.
	rule string

	// grouped reports that Patterns are the findings of one rule, which
	// Markdown lists by location instead of repeating each paragraph.
	grouped bool

	// severityByKind exempts this item from the Critical share floor. A
	// finding that says the module may not run, or that the report's own
	// ordering is misleading, is Critical for what it is and not for how much
	// device time it accounts for. Everything else earns Critical by cost and
	// must clear the floor. Not serialized: it is how severity was decided,
	// which the severity field already reports.
	severityByKind bool
}

// Report contains all recommendations resulting from analyzing a capture.
type Report struct {
	Items            []Item `json:"items"`
	EvidenceGuidance string `json:"evidence_guidance,omitempty"`

	// SampleOnly reports that the capture carries PC samples but no device
	// timing. A share of samples then says where execution was, never how
	// long anything took, and never whether the device was idle between
	// launches. That limit is a property of the instrument, not a defect,
	// but it has to be stated: a reader who does not know it reads
	// "17.66% of observed samples" as "17.66% of elapsed time".
	SampleOnly bool `json:"sample_only,omitempty"`

	// SpanNanos is the captured device span, when a duty-cycle reading
	// established one. Cost headers state a finding's share of device time,
	// which is a share of however long the device was busy -- on an idle
	// capture that is a small fraction of the elapsed span, and a reader who
	// does not know the difference reads "59.74% of device time" as most of
	// the run. With a span the header can say both.
	SpanNanos int64 `json:"span_nanos,omitempty"`

	// CaptureWindows is the number of capture ranges SpanNanos sums over,
	// when the target opened the range more than once. The time between
	// windows was not captured, and is counted neither busy nor idle.
	CaptureWindows int `json:"capture_windows,omitempty"`

	// SourceFusion reports that the capture's PTX and SASS are not from one
	// compilation, so LoadCode refused the source binding and every finding
	// below carries an empty file and a zero line. Without this a reader cannot
	// tell a refused binding from a compiler that emitted no line information.
	SourceFusion string `json:"source_fusion_failure,omitempty"`

	// Base is the comparison against a base capture that ranks the items,
	// when [Report.RankAgainst] was given one. It is marshaled as itself.
	Base Baseline `json:"base,omitempty"`
}

// sampleOnlyCaveat names what PC sampling cannot see. Activity tracing is the
// only tier that records launch counts and inter-launch gaps, so on a
// sample-only capture device idle is unmeasured rather than absent.
const sampleOnlyCaveat = "> **This capture is PC samples only.** A share of observed samples says where " +
	"execution was, and is *not a share of elapsed time*. PC sampling cannot observe " +
	"device idle between launches; capture with activity tracing to measure that.\n\n"

// DefaultLimit is the number of findings [Report.Markdown] prints.
const DefaultLimit = 10

// Markdown formats the advice report as human-readable Markdown with the default limit of findings.
func (r *Report) Markdown() string {
	return r.MarkdownWithLimit(DefaultLimit)
}

// MarkdownWithLimit formats up to limit findings as human-readable Markdown.
// If limit <= 0, all findings are printed without truncation.
func (r *Report) MarkdownWithLimit(limit int) string {
	if r == nil {
		return "# CUDA Kernel Advice Report\n\nNo optimization issues or architecture fallbacks identified.\n"
	}

	var sb strings.Builder
	sb.WriteString("# CUDA Kernel Advice Report\n\n")
	if r.SampleOnly {
		sb.WriteString(sampleOnlyCaveat)
	}
	if r.SourceFusion != "" {
		sb.WriteString(sourceFusionMarkdownCaveat(r.SourceFusion))
	}
	if r.CaptureWindows > 1 {
		fmt.Fprintf(&sb, "> **This capture holds %d capture windows.** The captured span, %.3f ms, is the sum of the device spans inside them; the time between windows was not captured and is not counted as idle.\n\n",
			r.CaptureWindows, float64(r.SpanNanos)/1e6)
	}
	if r.Base != nil {
		r.writeBase(&sb)
	}
	// The caveats above are written before this return, not after it: a report
	// with no findings on a capture that refused its source binding is not the
	// same as a clean report, and rendering them identically is how a refusal
	// reads as an all-clear.
	if len(r.Items) == 0 {
		sb.WriteString("No optimization issues or architecture fallbacks identified.\n")
		return demangleText(sb.String())
	}

	var readFirst, cost, context int
	for _, item := range r.Items {
		switch item.Severity {
		case SeverityReadFirst:
			readFirst++
		case SeverityCost:
			cost++
		case SeverityContext:
			context++
		}
	}

	sb.WriteString("## Summary\n\n")
	sb.WriteString(fmt.Sprintf("- **Read first**: %d\n", readFirst))
	sb.WriteString(fmt.Sprintf("- **Cost**: %d\n", cost))
	sb.WriteString(fmt.Sprintf("- **Context**: %d\n\n", context))

	if r.Base == nil {
		sb.WriteString("## Findings\n\n")
	}
	count := len(r.Items)
	if limit > 0 && limit < count {
		count = limit
		// The Summary above counts findings this report does not print. Say
		// so here, beside the count, rather than only in a footer the reader
		// reaches after forming an impression of how much there was.
		sb.WriteString(fmt.Sprintf("*Showing %d of %d findings (use --format=json for all, or -limit=0).*\n\n", count, len(r.Items)))
	}
	section := -1
	for i := 0; i < count; i++ {
		item := r.Items[i]
		if r.Base != nil && gapSection(item) != section {
			section = gapSection(item)
			sb.WriteString(gapHeadings[section] + "\n\n")
		}
		header := fmt.Sprintf("### %d. [%s] %s: %s\n\n", i+1, item.Severity, item.Category, item.Finding)
		if item.magnitude != "" {
			header = fmt.Sprintf("### %d. [%s] (%s) %s: %s\n\n", i+1, item.Severity, item.magnitude, item.Category, item.Finding)
		} else if item.ShareOfProfile > 0 || item.DeviceTimeNanos > 0 {
			if item.DeviceTimeNanos > 0 {
				durMs := float64(item.DeviceTimeNanos) / 1e6
				// "GPU time" named the busy union, not the elapsed span, and
				// the two diverge by an order of magnitude on an idle
				// capture. Both are printed where the span is known so the
				// larger share cannot be read as the smaller one.
				scope := fmt.Sprintf("%.2f%% of device time, %.2f ms", item.ShareOfProfile, durMs)
				if r.SpanNanos > 0 {
					scope = fmt.Sprintf("%s, %.2f%% of the captured span", scope, float64(item.DeviceTimeNanos)/float64(r.SpanNanos)*100.0)
				}
				header = fmt.Sprintf("### %d. [%s] (%s) %s: %s\n\n", i+1, item.Severity, scope, item.Category, item.Finding)
			} else {
				header = fmt.Sprintf("### %d. [%s] (%.2f%% of observed samples) %s: %s\n\n", i+1, item.Severity, item.ShareOfProfile, item.Category, item.Finding)
			}
		}
		sb.WriteString(header)
		writeAdviceBody(&sb, item, "")
		for j, p := range item.Patterns {
			if j == maxPatternLines {
				fmt.Fprintf(&sb, "\n*... %d more (use --format=json for all).*\n", len(item.Patterns)-j)
				break
			}
			if item.grouped {
				fmt.Fprintf(&sb, "%d. %s\n", j+1, groupedLine(p))
				continue
			}
			fmt.Fprintf(&sb, "%d. %s: %s\n", j+1, p.Category, p.Finding)
			writeAdviceBody(&sb, p, "   ")
		}
		sb.WriteString("\n")
	}

	if limit > 0 && len(r.Items) > limit {
		omitted := len(r.Items) - limit
		sb.WriteString(fmt.Sprintf("*... %d additional findings omitted (use --format=json for all, -e, or -limit=N).*\n", omitted))
	}
	if r.EvidenceGuidance != "" {
		fmt.Fprintf(&sb, "**Evidence note:** %s\n", r.EvidenceGuidance)
	}

	return demangleText(sb.String())
}

// maxPatternLines is the number of patterns Markdown prints under one finding.
const maxPatternLines = 5

// groupedLine describes one finding of a group by where it is and its size;
// the group's header and body already say what it is.
func groupedLine(p Item) string {
	where := p.Location.String()
	if where == "" {
		where = p.Finding
	}
	switch {
	case p.magnitude != "":
		return fmt.Sprintf("%s (%s)", where, p.magnitude)
	case p.DeviceTimeNanos > 0:
		return fmt.Sprintf("%s (%.2f ms)", where, float64(p.DeviceTimeNanos)/1e6)
	case p.ShareOfProfile > 0:
		return fmt.Sprintf("%s (%.2f%% of observed samples)", where, p.ShareOfProfile)
	}
	return where
}

// writeAdviceBody writes the list under a finding's header, each line
// prefixed by indent.
func writeAdviceBody(sb *strings.Builder, item Item, indent string) {
	if item.Gap != nil {
		fmt.Fprintf(sb, "%s- **Gap**: %s\n", indent, gapLine(item.Gap))
	}
	if item.EvidenceLevel != "" {
		fmt.Fprintf(sb, "%s- **Evidence**: %s\n", indent, item.EvidenceLevel)
	}
	if loc := item.Location.String(); loc != "" {
		fmt.Fprintf(sb, "%s- **Location**: %s\n", indent, loc)
	}
	if item.ActionableRecommendation != "" {
		fmt.Fprintf(sb, "%s- **Recommendation**: %s\n", indent, item.ActionableRecommendation)
	}
	if item.NextMeasurement != "" {
		fmt.Fprintf(sb, "%s- **Next Measurement**: %s\n", indent, item.NextMeasurement)
	}
}

// JSON formats the advice report as formatted JSON.
func (r *Report) JSON() (string, error) {
	if r == nil {
		return "{\n  \"items\": []\n}", nil
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal advice report: %w", err)
	}
	return string(data), nil
}

// sourceFusionMarkdownCaveat states a source-fusion failure as a report
// header, in the shape sampleOnlyCaveat established: above the findings, where it is read
// before the locations it qualifies.
func sourceFusionMarkdownCaveat(failure string) string {
	return "> **Source attribution unavailable.** " + failure +
		"\n>\n> Findings below carry no source file or line: the PTX and SASS of this capture are not from " +
		"one compilation, so the binding was *refused*, not attempted and lost. Kernel names, PC offsets " +
		"and SASS text are unaffected.\n\n"
}

// mangledName matches an Itanium-mangled symbol in running text.
var mangledName = regexp.MustCompile(`\b_Z[0-9A-Za-z_$.]+`)

// demangleText replaces each mangled kernel name in s with its demangled
// name without parameters, leaving a name the demangler rejects as it is.
// The Markdown report is read by people; the JSON keeps the names as
// recorded, which is what the other commands match.
func demangleText(s string) string {
	return mangledName.ReplaceAllStringFunc(s, func(name string) string {
		d, err := demangle.ToString(name, demangle.NoParams)
		if err != nil {
			return name
		}
		return d
	})
}
