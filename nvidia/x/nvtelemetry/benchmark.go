package nvtelemetry

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

// BenchmarkOptions configures request-to-telemetry correlation.
type BenchmarkOptions struct {
	CountFailedTokens       bool    // whether to count tokens from failed requests
	ClockProvenance         string  // clock synchronization provenance (e.g. "ptp", "ntp_synchronized", "local")
	EstimatedClockSkewNanos int64   // estimated clock skew between generator and telemetry clock
	MaxClockSkewNanos       int64   // maximum acceptable clock skew between benchmark generator and telemetry host
	MinCoverageRatio        float64 // minimum acceptable energy window coverage ratio (defaults to 1.0)
}

// CorrelateBenchmark joins telemetry energy summaries with benchmark request records for a specific window.
func CorrelateBenchmark(windowName string, startMono, endMono int64, energy []EnergySummary, requests []BenchmarkRecord, opts BenchmarkOptions) (BenchmarkCorrelation, error) {
	corr := BenchmarkCorrelation{
		WindowName:     windowName,
		StartMonoNanos: startMono, EndMonoNanos: endMono,
	}

	if endMono <= startMono || endMono-startMono <= 0 {
		return corr, errors.New("invalid window bounds")
	}

	if strings.TrimSpace(opts.ClockProvenance) == "" || strings.EqualFold(strings.TrimSpace(opts.ClockProvenance), "unknown") {
		corr.RefusalReason = "unknown_clock_provenance: precise join requires explicit clock synchronization provenance"
		return corr, nil
	}
	if opts.MaxClockSkewNanos < 0 {
		return corr, errors.New("negative maximum clock skew")
	}
	// Compare both signs without overflowing on MinInt64.
	if opts.EstimatedClockSkewNanos > opts.MaxClockSkewNanos || opts.EstimatedClockSkewNanos < -opts.MaxClockSkewNanos {
		corr.RefusalReason = fmt.Sprintf("%s: estimated skew %d ns exceeds limit %d ns", ReasonMisalignedClock, opts.EstimatedClockSkewNanos, opts.MaxClockSkewNanos)
		return corr, nil
	}
	corr.ClockAligned = true

	seenRequests := make(map[string]bool)
	for _, req := range requests {
		if strings.TrimSpace(req.RequestID) == "" || seenRequests[req.RequestID] || req.OutputTokens < 0 || req.PromptTokens < 0 {
			corr.RefusalReason = "invalid or duplicate request identity or token count"
			return corr, nil
		}
		seenRequests[req.RequestID] = true
		if req.EndTimeNanos < req.StartTimeNanos {
			corr.ClockAligned = false
			corr.RefusalReason = fmt.Sprintf("%s: request %s has negative duration", ReasonMisalignedClock, req.RequestID)
			return corr, nil
		}
	}

	// 2. Sum energy across all valid devices in this window with strict coverage checks
	var totalEnergyJoules float64
	var hasValidEnergy bool

	minCoverage := opts.MinCoverageRatio
	if math.IsNaN(minCoverage) || math.IsInf(minCoverage, 0) || minCoverage < 0 || minCoverage > 1 {
		return corr, errors.New("invalid minimum coverage ratio")
	}
	if minCoverage == 0 {
		minCoverage = 1.0 // partial energy window/device coverage must not become a full-window tokens/joule value
	}

	seenDevices := make(map[string]bool)
	for _, e := range energy {
		if e.WindowName != windowName {
			continue
		}
		if strings.TrimSpace(e.DeviceUUID) == "" || seenDevices[e.DeviceUUID] {
			corr.RefusalReason = "missing or duplicate energy device identity"
			return corr, nil
		}
		seenDevices[e.DeviceUUID] = true
		if math.IsNaN(e.CoverageRatio) || math.IsInf(e.CoverageRatio, 0) || e.CoverageRatio < 0 || e.CoverageRatio > 1 || math.IsNaN(e.EnergyConsumedJoules) || math.IsInf(e.EnergyConsumedJoules, 0) || e.EnergyConsumedJoules < 0 {
			corr.RefusalReason = "invalid energy or coverage"
			return corr, nil
		}
		duration := endMono - startMono
		if e.StartMonoNanos != startMono || e.EndMonoNanos != endMono || e.TotalDurationNanos != duration || e.CoveredDurationNanos < 0 || e.CoveredDurationNanos > duration || math.Abs(e.CoverageRatio-float64(e.CoveredDurationNanos)/float64(duration)) > 1e-12 {
			corr.RefusalReason = "energy interval or coverage duration mismatch"
			return corr, nil
		}
		if !e.Valid {
			corr.EnergyAvailable = false
			corr.RefusalReason = fmt.Sprintf("device %s energy invalid: %s", e.DeviceUUID, e.Reason)
			return corr, nil
		}
		if e.CoverageRatio < minCoverage {
			corr.EnergyAvailable = false
			corr.RefusalReason = fmt.Sprintf("device %s partial energy coverage %.1f%% below required %.1f%%", e.DeviceUUID, e.CoverageRatio*100, minCoverage*100)
			return corr, nil
		}
		totalEnergyJoules += e.EnergyConsumedJoules
		if math.IsInf(totalEnergyJoules, 0) {
			return corr, errors.New("energy sum overflow")
		}
		hasValidEnergy = true
	}

	if !hasValidEnergy || totalEnergyJoules <= 0 {
		corr.EnergyAvailable = false
		corr.RefusalReason = "no valid positive energy recorded"
		return corr, nil
	}
	corr.TotalEnergyJoules = totalEnergyJoules
	corr.EnergyAvailable = true

	// Filter requests falling within the window
	var totalTokens int
	var totalReqs int
	var successReqs int

	for _, req := range requests {
		// Check if request overlaps with the window
		if req.StartTimeNanos >= startMono && req.EndTimeNanos <= endMono {
			totalReqs++
			if req.Success {
				successReqs++
				if req.OutputTokens > int(^uint(0)>>1)-totalTokens {
					return corr, errors.New("token sum overflow")
				}
				totalTokens += req.OutputTokens
			} else if opts.CountFailedTokens {
				if req.OutputTokens > int(^uint(0)>>1)-totalTokens {
					return corr, errors.New("token sum overflow")
				}
				totalTokens += req.OutputTokens
			}
		}
	}

	corr.TotalRequests = totalReqs
	corr.SuccessfulRequests = successReqs
	corr.TotalOutputTokens = totalTokens

	if totalTokens <= 0 {
		corr.TokensAvailable = false
		corr.RefusalReason = ReasonZeroTokens
		return corr, nil
	}
	corr.TokensAvailable = true

	corr.JoulesPerOutputToken = totalEnergyJoules / float64(totalTokens)
	corr.OutputTokensPerJoule = float64(totalTokens) / totalEnergyJoules

	return corr, nil
}
