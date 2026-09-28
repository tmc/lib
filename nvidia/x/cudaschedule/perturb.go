package cudaschedule

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// DefaultMaxSleepNs is the default maximum nanosleep delay ceiling (64 µs).
const DefaultMaxSleepNs = 64000

// Options configures seeded perturbation of a PTX module.
type Options struct {
	Seed       uint64
	Intensity  float64
	MaxSleepNs uint32
	Focus      string
}

// InjectionDecision records the sleep durations injected around a specific site.
type InjectionDecision struct {
	Site     Site   `json:"site"`
	BeforeNs uint32 `json:"before_ns"`
	AfterNs  uint32 `json:"after_ns"`
}

// RewriteReport summarizes the perturbation applied to a PTX module.
type RewriteReport struct {
	Seed               uint64              `json:"seed"`
	Intensity          float64             `json:"intensity"`
	SitesTotal         int                 `json:"sites_total"`
	SitesInjected      int                 `json:"sites_injected"`
	InjectedNsPerVisit uint64              `json:"injected_ns_per_visit"`
	Decisions          []InjectionDecision `json:"decisions"`
}

// Rewrite is the result of perturbing a PTX module.
type Rewrite struct {
	PTX    string
	Report RewriteReport
}

// Perturb applies seeded deterministic schedule perturbations to a PTX module.
// The same seed and options produce byte-identical output across runs.
// A draw that selects no site returns the original text and reports zero injected sites.
func Perturb(src string, opts Options) (Rewrite, error) {
	if math.IsNaN(opts.Intensity) || math.IsInf(opts.Intensity, 0) || opts.Intensity < 0.0 {
		return Rewrite{}, errors.New("cudaschedule: intensity must be finite and non-negative")
	}

	analysis, err := Analyze(src)
	if err != nil {
		return Rewrite{}, err
	}

	intensity := opts.Intensity
	maxSleepNs := opts.MaxSleepNs
	if maxSleepNs == 0 {
		maxSleepNs = DefaultMaxSleepNs
	}

	rng := newSplitMix64(opts.Seed)
	decisions := make([]InjectionDecision, len(analysis))
	injectedSites := 0
	var totalNs uint64

	type edit struct {
		offset int
		order  int // 0 for before, 1 for after
		text   string
	}
	var edits []edit

	for i, site := range analysis {
		beforePoint, afterPoint := injectionPoints(src, site)
		placement := rng.unit()
		hit := opts.Focus != "" && (strings.Contains(site.Head, opts.Focus) || strings.Contains(site.Text, opts.Focus))

		var selected bool
		if intensity == 0.0 {
			selected = false
		} else if opts.Focus != "" {
			var prob float64
			if hit {
				prob = math.Min(1.0, 0.95*intensity)
			} else {
				prob = math.Min(1.0, 0.15*intensity)
			}
			selected = rng.unit() < prob
		} else {
			selected = rng.unit() < math.Min(1.0, 0.75*intensity)
		}

		var beforeNs, afterNs uint32
		if !selected {
			beforeNs, afterNs = 0, 0
		} else if site.Kind == SiteBackedge {
			beforeNs = drawDelay(rng, intensity, maxSleepNs)
			afterNs = 0
		} else if hit {
			beforeNs = 0
			afterNs = drawLong(rng, intensity, maxSleepNs)
		} else if placement < 0.4 {
			beforeNs = drawDelay(rng, intensity, maxSleepNs)
			afterNs = 0
		} else if placement < 0.8 {
			beforeNs = 0
			afterNs = drawDelay(rng, intensity, maxSleepNs)
		} else {
			beforeNs = drawDelay(rng, intensity, maxSleepNs)
			afterNs = drawDelay(rng, intensity, maxSleepNs)
		}
		if selected && beforeNs == 0 && afterNs == 0 {
			if site.Kind == SiteBackedge {
				beforeNs = drawLong(rng, intensity, maxSleepNs)
			} else {
				afterNs = drawLong(rng, intensity, maxSleepNs)
			}
		}

		if beforeNs > 0 {
			indent := lineIndent(src, beforePoint)
			insertText := fmt.Sprintf("%snanosleep.u32 %d; // ptx_schedule before\n", indent, beforeNs)
			edits = append(edits, edit{offset: beforePoint, order: 0, text: insertText})
		}
		if afterNs > 0 {
			insertText := fmt.Sprintf("\nnanosleep.u32 %d; // ptx_schedule after", afterNs)
			edits = append(edits, edit{offset: afterPoint, order: 1, text: insertText})
		}

		if beforeNs > 0 || afterNs > 0 {
			injectedSites++
			totalNs += uint64(beforeNs) + uint64(afterNs)
		}

		decisions[i] = InjectionDecision{
			Site:     site,
			BeforeNs: beforeNs,
			AfterNs:  afterNs,
		}
	}

	report := RewriteReport{
		Seed:               opts.Seed,
		Intensity:          intensity,
		SitesTotal:         len(analysis),
		SitesInjected:      injectedSites,
		InjectedNsPerVisit: totalNs,
		Decisions:          decisions,
	}

	// An empty draw is reported as such and returns the original text.
	if injectedSites == 0 {
		return Rewrite{
			PTX:    src,
			Report: report,
		}, nil
	}

	// Sort edits by offset ascending; for identical offsets, order 0 precedes order 1.
	sort.SliceStable(edits, func(i, j int) bool {
		if edits[i].offset == edits[j].offset {
			return edits[i].order < edits[j].order
		}
		return edits[i].offset < edits[j].offset
	})

	var b strings.Builder
	header := fmt.Sprintf(
		"// ptx_schedule: seed=%d intensity=%g sites_total=%d sites_injected=%d injected_ns_per_visit=%d\n",
		opts.Seed,
		intensity,
		len(analysis),
		injectedSites,
		totalNs,
	)
	b.WriteString(header)

	last := 0
	for _, e := range edits {
		if e.offset > last {
			b.WriteString(src[last:e.offset])
			last = e.offset
		}
		b.WriteString(e.text)
	}
	if last < len(src) {
		b.WriteString(src[last:])
	}

	return Rewrite{
		PTX:    b.String(),
		Report: report,
	}, nil
}

func injectionPoints(src string, site Site) (int, int) {
	beforeStart := strings.LastIndex(src[:site.Span.Start], "// begin inline asm")
	beforeEnd := strings.LastIndex(src[:site.Span.Start], "// end inline asm")
	if beforeStart < 0 || (beforeEnd >= 0 && beforeEnd > beforeStart) {
		return site.Span.Start, site.Span.End
	}

	begin := beforeStart
	lineStart := strings.LastIndex(src[:begin], "\n")
	beginLine := 0
	if lineStart >= 0 {
		beginLine = lineStart + 1
	}

	endMarker := site.Span.End
	if offset := strings.Index(src[site.Span.End:], "// end inline asm"); offset >= 0 {
		endMarker = site.Span.End + offset + len("// end inline asm")
	}
	return beginLine, endMarker
}

func lineIndent(src string, offset int) string {
	lineStart := strings.LastIndex(src[:offset], "\n")
	if lineStart < 0 {
		lineStart = 0
	} else {
		lineStart++
	}
	indentEnd := lineStart
	for indentEnd < offset && (src[indentEnd] == ' ' || src[indentEnd] == '\t') {
		indentEnd++
	}
	return src[lineStart:indentEnd]
}

func drawDelay(rng *splitMix64, intensity float64, maxSleepNs uint32) uint32 {
	if rng.unit() < 0.25 {
		return 0
	}
	high := scaledMax(intensity, maxSleepNs)
	if rng.unit() < 0.5 {
		return rng.rangeU32(1, minU32(high, 2000))
	}
	return rng.rangeU32(minU32(2000, high), high)
}

func drawLong(rng *splitMix64, intensity float64, maxSleepNs uint32) uint32 {
	high := scaledMax(intensity, maxSleepNs)
	return rng.rangeU32(minU32(2000, high), high)
}

func scaledMax(intensity float64, maxSleepNs uint32) uint32 {
	scaled := math.Round(float64(maxSleepNs) * math.Min(1.0, intensity))
	if scaled < 1 {
		return 1
	}
	return uint32(scaled)
}

func minU32(a, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}

type splitMix64 struct {
	state uint64
}

func newSplitMix64(seed uint64) *splitMix64 {
	return &splitMix64{state: seed}
}

func (s *splitMix64) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (s *splitMix64) unit() float64 {
	return float64(s.next()>>11) / float64(uint64(1)<<53)
}

func (s *splitMix64) rangeU32(low, high uint32) uint32 {
	if low >= high {
		return low
	}
	diff := uint64(high) - uint64(low) + 1
	return low + uint32(s.next()%diff)
}
