package cudapipeline

import (
	"fmt"
	"strings"
)

// Severity indicates the seriousness of a static validation issue.
type Severity int

const (
	SeverityError Severity = iota + 1
	SeverityWarning
)

func (s Severity) String() string {
	switch s {
	case SeverityError:
		return "ERROR"
	case SeverityWarning:
		return "WARNING"
	default:
		return "UNKNOWN"
	}
}

// Category identifies the validation check category.
type Category string

const (
	CatTransactionMismatch Category = "transaction_mismatch"
	CatPhaseParity         Category = "phase_parity"
	CatLatencyHiding       Category = "latency_hiding"
	CatAlignmentStride     Category = "alignment_stride"
	CatBufferCapacity      Category = "buffer_capacity"
	CatHardwareConstraint  Category = "hardware_constraint"
)

// Issue describes a specific problem found during pipeline verification.
type Issue struct {
	StageIndex int      // Index of the affected stage (-1 for pipeline-wide issues)
	Severity   Severity // SeverityError or SeverityWarning
	Category   Category // Issue classification
	Message    string   // Description of the problem and hardware consequence
}

func (iss Issue) String() string {
	if iss.StageIndex >= 0 {
		return fmt.Sprintf("[%s] %s (stage %d): %s", iss.Severity, iss.Category, iss.StageIndex, iss.Message)
	}
	return fmt.Sprintf("[%s] %s: %s", iss.Severity, iss.Category, iss.Message)
}

// Report collects all issues and performance metrics from pipeline analysis.
type Report struct {
	Issues                []Issue
	MemoryLatencyCycles   int
	ComputeIntervalCycles int
	StagesNeeded          int
	ActualStages          int
	LatencyHiddenPercent  float64
}

// Valid reports whether the pipeline passed validation without any errors.
func (r *Report) Valid() bool {
	for _, iss := range r.Issues {
		if iss.Severity == SeverityError {
			return false
		}
	}
	return true
}

// Errors returns all issues with SeverityError.
func (r *Report) Errors() []Issue {
	var errs []Issue
	for _, iss := range r.Issues {
		if iss.Severity == SeverityError {
			errs = append(errs, iss)
		}
	}
	return errs
}

// Warnings returns all issues with SeverityWarning.
func (r *Report) Warnings() []Issue {
	var warns []Issue
	for _, iss := range r.Issues {
		if iss.Severity == SeverityWarning {
			warns = append(warns, iss)
		}
	}
	return warns
}

func (r *Report) String() string {
	var b strings.Builder
	if r.Valid() {
		fmt.Fprintf(&b, "Pipeline validation: PASS (%d stages, %.1f%% latency hidden)\n",
			r.ActualStages, r.LatencyHiddenPercent)
	} else {
		fmt.Fprintf(&b, "Pipeline validation: FAIL (%d error(s), %d warning(s))\n",
			len(r.Errors()), len(r.Warnings()))
	}
	for _, iss := range r.Issues {
		fmt.Fprintf(&b, "  %s\n", iss.String())
	}
	return b.String()
}

// Validate performs comprehensive static verification of the pipeline.
func (p *Pipeline) Validate() *Report {
	return ValidatePipeline(p)
}

// ValidatePipeline performs comprehensive static verification of a pipeline,
// checking transaction counts, phase parity, latency hiding, alignment, and hardware constraints.
func ValidatePipeline(p *Pipeline) *Report {
	r := &Report{
		ActualStages: len(p.Stages),
	}

	if p == nil {
		r.Issues = append(r.Issues, Issue{
			StageIndex: -1,
			Severity:   SeverityError,
			Category:   CatHardwareConstraint,
			Message:    "pipeline is nil",
		})
		return r
	}

	if len(p.Stages) < 2 {
		r.Issues = append(r.Issues, Issue{
			StageIndex: -1,
			Severity:   SeverityError,
			Category:   CatHardwareConstraint,
			Message:    fmt.Sprintf("pipeline has %d stages; minimum 2 required for multi-stage buffering", len(p.Stages)),
		})
	}

	// 1. Architecture instruction support.
	if !p.Arch.SupportsTMA {
		r.Issues = append(r.Issues, Issue{
			StageIndex: -1,
			Severity:   SeverityError,
			Category:   CatHardwareConstraint,
			Message:    fmt.Sprintf("architecture %s does not support TMA bulk tensor operations", p.Arch.Name),
		})
	}
	if !p.Arch.SupportsMBarrier {
		r.Issues = append(r.Issues, Issue{
			StageIndex: -1,
			Severity:   SeverityError,
			Category:   CatHardwareConstraint,
			Message:    fmt.Sprintf("architecture %s does not support PTX mbarrier operations", p.Arch.Name),
		})
	}

	hasTMEM := false
	for _, s := range p.Stages {
		if s.Buffer.Space == MemorySpaceTMEM {
			hasTMEM = true
			break
		}
	}
	if hasTMEM && !p.Arch.SupportsTMEM {
		r.Issues = append(r.Issues, Issue{
			StageIndex: -1,
			Severity:   SeverityError,
			Category:   CatHardwareConstraint,
			Message:    fmt.Sprintf("architecture %s does not support Tensor Memory (TMEM)", p.Arch.Name),
		})
	}
	if hasTMEM && p.Arch.SupportsTMEM && p.Arch.MaxTMEMCapacityBytes > 0 {
		totalTMEM := p.TotalTMEMBytes()
		if totalTMEM > p.Arch.MaxTMEMCapacityBytes {
			r.Issues = append(r.Issues, Issue{
				StageIndex: -1,
				Severity:   SeverityError,
				Category:   CatBufferCapacity,
				Message: fmt.Sprintf(
					"total TMEM allocation (%d bytes) exceeds maximum SM capacity limit (%d bytes) on %s",
					totalTMEM, p.Arch.MaxTMEMCapacityBytes, p.Arch.Name),
			})
		}
	}

	// 2. Stage-by-stage validation.
	for i := range p.Stages {
		s := &p.Stages[i]
		payload := s.TMABox.PayloadBytes()
		rowBytes := s.TMABox.RowBytes()

		// Transaction byte count check.
		tx := s.Barrier.TransactionBytes
		if tx != payload {
			if tx > payload {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityError,
					Category:   CatTransactionMismatch,
					Message: fmt.Sprintf(
						"barrier expects %d transaction bytes, but TMA box payload is %d bytes (under-counting will cause permanent deadlock)",
						tx, payload),
				})
			} else {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityError,
					Category:   CatTransactionMismatch,
					Message: fmt.Sprintf(
						"barrier expects %d transaction bytes, but TMA box payload is %d bytes (over-counting will cause premature release and race conditions)",
						tx, payload),
				})
			}
		}

		// Buffer capacity check.
		if s.Buffer.SizeBytes < payload {
			bufName := "buffer"
			if s.Buffer.Space == MemorySpaceTMEM {
				bufName = "TMEM buffer"
			}
			r.Issues = append(r.Issues, Issue{
				StageIndex: s.Index,
				Severity:   SeverityError,
				Category:   CatBufferCapacity,
				Message: fmt.Sprintf(
					"%s size (%d bytes) is smaller than TMA payload (%d bytes)",
					bufName, s.Buffer.SizeBytes, payload),
			})
		}

		// TMEM buffer boundary check.
		if s.Buffer.Space == MemorySpaceTMEM && p.Arch.MaxTMEMCapacityBytes > 0 {
			bufEnd := s.Buffer.Offset + s.Buffer.SizeBytes
			if bufEnd > p.Arch.MaxTMEMCapacityBytes {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityError,
					Category:   CatBufferCapacity,
					Message: fmt.Sprintf(
						"TMEM buffer [%d, %d) exceeds maximum SM TMEM capacity (%d bytes)",
						s.Buffer.Offset, bufEnd, p.Arch.MaxTMEMCapacityBytes),
				})
			}
		}

		// Buffer alignment conformance.
		if s.Buffer.Space == MemorySpaceTMEM {
			if s.Buffer.Offset%16 != 0 {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityError,
					Category:   CatAlignmentStride,
					Message: fmt.Sprintf(
						"TMEM buffer offset %d is not 16-byte aligned (required by async bulk transfer to TMEM)",
						s.Buffer.Offset),
				})
			} else if s.Buffer.Offset%128 != 0 {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityWarning,
					Category:   CatAlignmentStride,
					Message: fmt.Sprintf(
						"TMEM buffer offset %d is not 128-byte aligned (recommended for optimal TMEM throughput)",
						s.Buffer.Offset),
				})
			}
		} else {
			if s.Buffer.Offset%16 != 0 {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityError,
					Category:   CatAlignmentStride,
					Message: fmt.Sprintf(
						"shared memory buffer offset %d is not 16-byte aligned (required by TMA cp.async.bulk)",
						s.Buffer.Offset),
				})
			} else if s.Buffer.Offset%128 != 0 {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityWarning,
					Category:   CatAlignmentStride,
					Message: fmt.Sprintf(
						"shared memory buffer offset %d is not 128-byte aligned (recommended for maximum TMA throughput)",
						s.Buffer.Offset),
				})
			}
		}

		// Barrier alignment conformance.
		if s.Barrier.Offset%8 != 0 {
			r.Issues = append(r.Issues, Issue{
				StageIndex: s.Index,
				Severity:   SeverityError,
				Category:   CatAlignmentStride,
				Message: fmt.Sprintf(
					"mbarrier offset %d is not 8-byte aligned (required by mbarrier.b64)",
					s.Barrier.Offset),
			})
		}

		// TMA box row width constraints.
		if rowBytes%16 != 0 {
			r.Issues = append(r.Issues, Issue{
				StageIndex: s.Index,
				Severity:   SeverityError,
				Category:   CatAlignmentStride,
				Message: fmt.Sprintf(
					"TMA box row width (%d bytes) is not a multiple of 16 bytes",
					rowBytes),
			})
		}
		if p.Arch.MaxBoxWidthBytes > 0 && rowBytes > p.Arch.MaxBoxWidthBytes {
			r.Issues = append(r.Issues, Issue{
				StageIndex: s.Index,
				Severity:   SeverityError,
				Category:   CatAlignmentStride,
				Message: fmt.Sprintf(
					"TMA box row width (%d bytes) exceeds maximum %d bytes for %s",
					rowBytes, p.Arch.MaxBoxWidthBytes, p.Arch.Name),
			})
		}

		// Stride constraints.
		if s.TMABox.StrideBytes > 0 {
			if s.TMABox.StrideBytes%16 != 0 {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityError,
					Category:   CatAlignmentStride,
					Message: fmt.Sprintf(
						"TMA stride (%d bytes) is not a multiple of 16 bytes",
						s.TMABox.StrideBytes),
				})
			}
			if s.TMABox.StrideBytes < rowBytes {
				r.Issues = append(r.Issues, Issue{
					StageIndex: s.Index,
					Severity:   SeverityError,
					Category:   CatAlignmentStride,
					Message: fmt.Sprintf(
						"TMA stride (%d bytes) is smaller than row width (%d bytes)",
						s.TMABox.StrideBytes, rowBytes),
				})
			}
		}

		// Initial phase parity: mbarriers must start at phase 0.
		if s.Barrier.Phase != 0 {
			r.Issues = append(r.Issues, Issue{
				StageIndex: s.Index,
				Severity:   SeverityError,
				Category:   CatPhaseParity,
				Message: fmt.Sprintf(
					"initial mbarrier phase is %d; mbarrier must be initialized at phase 0",
					s.Barrier.Phase),
			})
		}

		// Arrival count must be > 0.
		if s.Barrier.ArrivalCount <= 0 {
			r.Issues = append(r.Issues, Issue{
				StageIndex: s.Index,
				Severity:   SeverityError,
				Category:   CatHardwareConstraint,
				Message: fmt.Sprintf(
					"mbarrier arrival count is %d; must be at least 1",
					s.Barrier.ArrivalCount),
			})
		}
	}

	// 3. Overlap check among stage buffers and barriers.
	for i := 0; i < len(p.Stages); i++ {
		b1Start := p.Stages[i].Buffer.Offset
		b1End := b1Start + p.Stages[i].Buffer.SizeBytes
		b1Space := p.Stages[i].Buffer.Space
		if b1Space == "" {
			b1Space = MemorySpaceShared
		}

		for j := i + 1; j < len(p.Stages); j++ {
			b2Space := p.Stages[j].Buffer.Space
			if b2Space == "" {
				b2Space = MemorySpaceShared
			}
			if b1Space == b2Space {
				b2Start := p.Stages[j].Buffer.Offset
				b2End := b2Start + p.Stages[j].Buffer.SizeBytes
				if !(b1End <= b2Start || b2End <= b1Start) {
					label := "buffer"
					if b1Space == MemorySpaceTMEM {
						label = "TMEM buffer"
					}
					r.Issues = append(r.Issues, Issue{
						StageIndex: j,
						Severity:   SeverityError,
						Category:   CatBufferCapacity,
						Message: fmt.Sprintf(
							"stage %d %s [%d, %d) overlaps with stage %d %s [%d, %d)",
							j, label, b2Start, b2End, i, label, b1Start, b1End),
					})
				}
			}
		}

		// Stage barriers reside in shared memory. Only check overlap with shared memory stage buffers.
		if b1Space == MemorySpaceShared {
			for j := 0; j < len(p.Stages); j++ {
				barStart := p.Stages[j].Barrier.Offset
				barEnd := barStart + 8
				if !(b1End <= barStart || barEnd <= b1Start) {
					r.Issues = append(r.Issues, Issue{
						StageIndex: i,
						Severity:   SeverityError,
						Category:   CatBufferCapacity,
						Message: fmt.Sprintf(
							"stage %d buffer [%d, %d) overlaps with stage %d mbarrier [%d, %d)",
							i, b1Start, b1End, j, barStart, barEnd),
					})
				}
			}
		}
	}

	// 4. Latency hiding analysis.
	memLat := p.Config.MemoryLatencyCycles
	computeInt := p.Config.ComputeIntervalCycles
	r.MemoryLatencyCycles = memLat
	r.ComputeIntervalCycles = computeInt

	if memLat > 0 && computeInt > 0 {
		actualStages := len(p.Stages)
		hiddenCycles := (actualStages - 1) * computeInt
		stagesNeeded := (memLat+computeInt-1)/computeInt + 1
		pct := (float64(hiddenCycles) / float64(memLat)) * 100.0
		if pct > 100.0 {
			pct = 100.0
		}
		r.StagesNeeded = stagesNeeded
		r.LatencyHiddenPercent = pct

		if actualStages < stagesNeeded {
			r.Issues = append(r.Issues, Issue{
				StageIndex: -1,
				Severity:   SeverityWarning,
				Category:   CatLatencyHiding,
				Message: fmt.Sprintf(
					"buffer depth of %d stages hides %d/%d cycles (%.1f%%) of memory latency; %d stages required to prevent memory stalls",
					actualStages, hiddenCycles, memLat, pct, stagesNeeded),
			})
		}
	}

	return r
}

// ValidatePhaseParity checks a sequence of producer arrival parities and consumer
// wait parities across pipeline iterations, detecting phase inversions and deadlocks.
func ValidatePhaseParity(numStages int, producerParities, consumerParities []int) []Issue {
	var issues []Issue
	n := len(producerParities)
	if len(consumerParities) < n {
		n = len(consumerParities)
	}

	for iter := 0; iter < n; iter++ {
		stageIdx := iter % numStages
		prodP := producerParities[iter]
		consP := consumerParities[iter]

		// Theoretical parity for this stage: (iter / numStages) & 1
		expectedP := (iter / numStages) & 1

		if prodP != expectedP {
			issues = append(issues, Issue{
				StageIndex: stageIdx,
				Severity:   SeverityError,
				Category:   CatPhaseParity,
				Message: fmt.Sprintf(
					"iteration %d (stage %d): producer parity %d does not match expected parity %d",
					iter, stageIdx, prodP, expectedP),
			})
		}
		if consP != expectedP {
			issues = append(issues, Issue{
				StageIndex: stageIdx,
				Severity:   SeverityError,
				Category:   CatPhaseParity,
				Message: fmt.Sprintf(
					"iteration %d (stage %d): consumer wait parity %d does not match expected parity %d (phase inversion will deadlock)",
					iter, stageIdx, consP, expectedP),
			})
		}
		if prodP != consP {
			issues = append(issues, Issue{
				StageIndex: stageIdx,
				Severity:   SeverityError,
				Category:   CatPhaseParity,
				Message: fmt.Sprintf(
					"iteration %d (stage %d): producer parity (%d) != consumer parity (%d); consumer will wait indefinitely",
					iter, stageIdx, prodP, consP),
			})
		}
	}

	return issues
}
