package advise

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

const (
	// microKernelMedianNs is the median kernel duration below which a sequence
	// is short enough for per-launch host cost to be worth testing for.
	microKernelMedianNs = 10000
	// lowDutyCyclePct is the GPU duty cycle below which a sequence's device
	// span is mostly idle.
	lowDutyCyclePct = 75.0
	// hostBehindFraction is the share of a sequence's measured launches that
	// must start after the preceding kernel ended before the host counts as
	// persistently behind the device.
	hostBehindFraction = 0.75
	// warpSize is the hardware warp width, constant across every CUDA
	// architecture cudaprof supports.
	warpSize = 32
)

// sequenceDutyCycle returns the percentage of the sequence's device span spent
// executing its kernels. ok is false when the capture does not carry usable
// timestamps for every kernel in the sequence.
func sequenceDutyCycle(seq []timeline.Event) (pct float64, ok bool) {
	if len(seq) == 0 {
		return 0, false
	}
	var busy, first, last uint64
	for _, ev := range seq {
		if ev.Start == 0 || ev.End < ev.Start {
			return 0, false
		}
		busy += ev.DurationNs
		if first == 0 || ev.Start < first {
			first = ev.Start
		}
		if ev.End > last {
			last = ev.End
		}
	}
	span := last - first
	if span == 0 {
		return 0, false
	}
	return float64(busy) * 100.0 / float64(span), true
}

// hostBehindLaunches counts the sequence launches whose host API call started
// only after the preceding kernel had already finished on the device, i.e. the
// launches for which the device waited on the host. measured counts the
// launches for which both timestamps were available.
func hostBehindLaunches(seq []timeline.Event, apis map[uint32]timeline.APICall) (behind, measured int) {
	if len(apis) == 0 {
		return 0, 0
	}
	for i := 1; i < len(seq); i++ {
		prevEnd := seq[i-1].End
		call, ok := apis[seq[i].Correlation]
		if !ok || prevEnd == 0 || call.Start == 0 {
			continue
		}
		measured++
		if call.Start > prevEnd {
			behind++
		}
	}
	return behind, measured
}

func auditStreamKernelSequences(cap *cudacapture.Capture, report *Report) {
	if cap == nil {
		return
	}

	streamEvents, _ := timeline.Events(cap)
	if len(streamEvents) == 0 {
		return
	}
	apis := timeline.APIRecords(cap)
	graphReplays := capturedGraphReplays(cap)
	seenDutyUnmeasured := make(map[uint32]bool)

	streams := make([]uint32, 0, len(streamEvents))
	for s := range streamEvents {
		streams = append(streams, s)
	}
	sort.Slice(streams, func(i, j int) bool { return streams[i] < streams[j] })

	for _, streamID := range streams {
		events := streamEvents[streamID]
		if len(events) == 0 {
			continue
		}

		// Stream-wide micro-kernel telemetry (< 10 µs)
		var streamMicroCount int
		var streamMicroTotalNs uint64
		streamMicroKernelCounts := make(map[string]int)
		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			if (sub == "kernel" || sub == "") && ev.DurationNs < microKernelMedianNs && ev.DurationNs > 0 {
				streamMicroCount++
				streamMicroTotalNs += ev.DurationNs
				streamMicroKernelCounts[cleanKernelName(ev.Name)]++
			}
		}

		// 1. Heuristic 4: CUDA Graph Launch Sequence Candidates
		var kernelSeq []timeline.Event
		checkGraphCandidate := func() {
			if len(kernelSeq) >= 5 {
				durations := make([]uint64, len(kernelSeq))
				for i, k := range kernelSeq {
					durations[i] = k.DurationNs
				}
				sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
				var median uint64
				n := len(durations)
				if n%2 == 1 {
					median = durations[n/2]
				} else {
					median = (durations[n/2-1] + durations[n/2]) / 2
				}

				if median < microKernelMedianNs {
					detail := fmt.Sprintf("sequence of %d consecutive kernels on stream %d has median duration %.2f µs (< %d µs)", len(kernelSeq), streamID, float64(median)/1000.0, microKernelMedianNs/1000)
					if streamMicroCount >= 5 {
						detail += fmt.Sprintf(" (stream totals %d micro-kernels accumulating %.2f ms)", streamMicroCount, float64(streamMicroTotalNs)/1e6)
					}
					if streamMicroCount >= 10 {
						type kCount struct {
							name  string
							count int
						}
						var sortedK []kCount
						for k, c := range streamMicroKernelCounts {
							sortedK = append(sortedK, kCount{name: k, count: c})
						}
						sort.Slice(sortedK, func(i, j int) bool {
							if sortedK[i].count != sortedK[j].count {
								return sortedK[i].count > sortedK[j].count
							}
							return sortedK[i].name < sortedK[j].name
						})
						var topList []string
						for i := 0; i < len(sortedK) && i < 3; i++ {
							topList = append(topList, fmt.Sprintf("%s (%dx)", sortedK[i].name, sortedK[i].count))
						}
						detail += fmt.Sprintf(" [top micro-kernels: %s]", strings.Join(topList, ", "))
					}

					// A short kernel does not imply that the host cannot keep the
					// device fed. Report submission overhead only where the device
					// is measurably idle and the host is measurably behind it.
					duty, dutyOK := sequenceDutyCycle(kernelSeq)
					switch {
					case !dutyOK:
						if !seenDutyUnmeasured[streamID] {
							seenDutyUnmeasured[streamID] = true
							report.Items = append(report.Items, Item{
								rule:          "unmeasured-sequence",
								Severity:      SeverityContext,
								Category:      CategoryArchitectureDispatch,
								EvidenceLevel: EvidenceUnmeasured,
								Location: Location{
									Kernel: kernelSeq[0].Name,
								},
								Finding:                  fmt.Sprintf("Short-kernel dispatch sequence with unmeasured GPU duty cycle: %s, but the timestamps this reader could reach carry no activity start and end, so neither device idle time nor host submission was measured%s", detail, journalRemedy(cap)),
								ActionableRecommendation: recaptureOrRead(cap),
								NextMeasurement:          "activity start/end timestamps in profile or capture-status.json",
							})
						}
					case duty >= lowDutyCyclePct:
						// The device is busy across the sequence span; per-launch
						// host cost is not costing device time here.
					default:
						behind, measured := hostBehindLaunches(kernelSeq, apis)
						sev := SeverityContext
						why, rec := unmeasuredLaunchReason(len(apis))
						finding := fmt.Sprintf("Low GPU duty cycle across a short-kernel sequence: %s and its kernels occupy only %.1f%% of the sequence device span (< %.0f%%); %s, so the cause of the idle time was not measured", detail, duty, lowDutyCyclePct, why)
						evLevel := EvidenceSuspected
						if measured == 0 {
							evLevel = EvidenceUnmeasured
						} else if float64(behind) >= hostBehindFraction*float64(measured) {
							sev = SeverityCost
							evLevel = EvidenceObserved
							finding = fmt.Sprintf("Driver submission overhead dominates short kernel execution: %s, its kernels occupy only %.1f%% of the sequence device span (< %.0f%%), and %d of %d measured launches (%.1f%%) began only after the preceding kernel had already finished", detail, duty, lowDutyCyclePct, behind, measured, float64(behind)*100.0/float64(measured))
							rec = "Capture sequence into a CUDA Graph (cudaGraphLaunch) to eliminate host submission latency."
							if graphReplays > 0 {
								rec = "This sequence already runs as a CUDA graph replay, so submission latency is not removable by graph capture; look for host work between replays."
							}
						} else if measured > 0 {
							finding = fmt.Sprintf("Low GPU duty cycle across a short-kernel sequence: %s and its kernels occupy only %.1f%% of the sequence device span (< %.0f%%), but only %d of %d measured launches began after the preceding kernel had finished, so the host was not behind the device for most of the sequence", detail, duty, lowDutyCyclePct, behind, measured)
							rec = "Idle device time here is not explained by host launch submission; look for stream dependencies, synchronization, or concurrent work on other streams."
						}
						report.Items = append(report.Items, Item{
							rule:          "short-sequence",
							Severity:      sev,
							Category:      CategoryArchitectureDispatch,
							EvidenceLevel: evLevel,
							Location: Location{
								Kernel: kernelSeq[0].Name,
							},
							Finding:                  finding,
							ActionableRecommendation: rec,
							NextMeasurement:          "host launch API records (cudaLaunchKernel) with correlation timestamps or A/B wall-clock timing",
						})
					}
				}
			}
			kernelSeq = kernelSeq[:0]
		}

		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			nameLower := strings.ToLower(ev.Name)
			if sub == "memcpy" || strings.Contains(nameLower, "memcpy") ||
				sub == "synchronization" || sub == "sync" || strings.Contains(nameLower, "synchroniz") {
				checkGraphCandidate()
				continue
			}
			if sub == "kernel" || sub == "" {
				kernelSeq = append(kernelSeq, ev)
			}
		}
		checkGraphCandidate()

		// 1b. Repeated fine-grained host loop dispatch detection
		consecutiveKernelCounts := make(map[string]int)
		var maxConsecutive int
		var maxKernel string
		var curRun int
		var curK string
		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			if sub == "kernel" || sub == "" {
				kClean := cleanKernelName(ev.Name)
				if kClean == curK {
					curRun++
				} else {
					curK = kClean
					curRun = 1
				}
				if curRun > consecutiveKernelCounts[kClean] {
					consecutiveKernelCounts[kClean] = curRun
				}
				if curRun > maxConsecutive {
					maxConsecutive = curRun
					maxKernel = curK
				}
			} else {
				curK = ""
				curRun = 0
			}
		}

		if maxConsecutive >= 50 && maxKernel != "" {
			report.Items = append(report.Items, Item{
				rule:          "host-loop",
				Severity:      SeverityCost,
				Category:      CategoryArchitectureDispatch,
				EvidenceLevel: EvidenceObserved,
				Location: Location{
					Kernel: maxKernel,
				},
				Finding:                  fmt.Sprintf("Repeated fine-grained host loop dispatch: kernel %q was launched %d consecutive times on stream %d. This indicates an unbatched host-side token or row loop.", maxKernel, maxConsecutive, streamID),
				ActionableRecommendation: "Restructure caller to batch loop iterations into a single multi-row or multi-token cooperative grid launch, or capture the host loop into a CUDA Graph.",
				NextMeasurement:          "A/B wall-clock timing with batched dispatches or CUDA Graph",
			})
		}

		// 2. Heuristic 5: Pointwise Producer-Consumer Fusion & Sequential Misses
		type pairStat struct {
			count    int
			totalDur uint64
			ev1      timeline.Event
			ev2      timeline.Event
		}
		var pairKeys []string
		pairStats := make(map[string]*pairStat)
		for i := 0; i < len(events)-1; i++ {
			ev1 := events[i]
			ev2 := events[i+1]
			sub1 := strings.ToLower(ev1.Subtype)
			sub2 := strings.ToLower(ev2.Subtype)
			if (sub1 != "kernel" && sub1 != "") || (sub2 != "kernel" && sub2 != "") {
				continue
			}
			k1 := cleanKernelName(ev1.Name)
			k2 := cleanKernelName(ev2.Name)
			key := fmt.Sprintf("%s->%s:%d", k1, k2, streamID)
			ps := pairStats[key]
			if ps == nil {
				ps = &pairStat{ev1: ev1, ev2: ev2}
				pairStats[key] = ps
				pairKeys = append(pairKeys, key)
			}
			ps.count++
			ps.totalDur += ev1.DurationNs + ev2.DurationNs
		}

		seenFusion := make(map[string]bool)
		for _, key := range pairKeys {
			ps := pairStats[key]
			ev1 := ps.ev1
			ev2 := ps.ev2
			cleanName1 := cleanKernelName(ev1.Name)
			cleanName2 := cleanKernelName(ev2.Name)

			// Elementwise producer-consumer fusion
			if isElementwise(ev1) && isElementwise(ev2) && matchingShapesOrBuffers(ev1, ev2) {
				seenFusion[key] = true
				finding := fmt.Sprintf("Redundant global memory round-trips for intermediate tensors: sequential back-to-back elementwise kernels (%s followed by %s) on stream %d operate on matching tensors", ev1.Name, ev2.Name, streamID)
				if ps.count > 1 {
					finding += fmt.Sprintf(" (occurred %d times", ps.count)
					if ps.totalDur > 0 {
						finding += fmt.Sprintf(", accumulating %s", time.Duration(ps.totalDur).String())
					}
					finding += ")"
				}
				rec := fmt.Sprintf("Fuse into a single kernel (e.g. fused %s + %s) to eliminate intermediate DRAM round-trips.", cleanName1, cleanName2)

				report.Items = append(report.Items, Item{
					rule:          "round-trip",
					Severity:      SeverityCost,
					Category:      CategoryMemoryCompute,
					EvidenceLevel: EvidenceSuspected,
					// The pair is the unit of device time here, so the item carries the
					// sum the finding already names. Without it the item reaches a
					// consumer with neither device_time_nanos nor share_of_profile --
					// Location.Kernel is "A -> B", which is no kernel in the profile, so
					// the fill-in below looks it up and finds nothing. Pairs overlap, so
					// these shares do not sum with the per-kernel ones.
					DeviceTimeNanos: int64(ps.totalDur),
					Location: Location{
						Kernel: fmt.Sprintf("%s -> %s", ev1.Name, ev2.Name),
					},
					Finding:                  finding,
					ActionableRecommendation: rec,
					NextMeasurement:          "A/B wall-clock timing with fused kernel",
				})
			}

			// Sequential RMSNorm -> Block Quantization Fusion
			if isNormKernel(ev1.Name) && isQuantKernel(ev2.Name) {
				seenFusion[key] = true
				finding := fmt.Sprintf("Redundant global memory round-trips for normalized activations: %s writes normalized activations to DRAM and %s immediately reads them back for quantization on stream %d", ev1.Name, ev2.Name, streamID)
				if ps.count > 1 {
					finding += fmt.Sprintf(" (occurred %d times", ps.count)
					if ps.totalDur > 0 {
						finding += fmt.Sprintf(", accumulating %s", time.Duration(ps.totalDur).String())
					}
					finding += ")"
				}
				rec := fmt.Sprintf("Fuse normalization and quantization into a single kernel (e.g. fused %s_%s) that normalizes in shared memory/registers and directly stores quantized blocks to eliminate %d DRAM round-trips.", cleanName1, cleanName2, ps.count)

				report.Items = append(report.Items, Item{
					rule:          "round-trip",
					Severity:      SeverityCost,
					Category:      CategoryMemoryCompute,
					EvidenceLevel: EvidenceSuspected,
					// The pair is the unit of device time here, so the item carries the
					// sum the finding already names. Without it the item reaches a
					// consumer with neither device_time_nanos nor share_of_profile --
					// Location.Kernel is "A -> B", which is no kernel in the profile, so
					// the fill-in below looks it up and finds nothing. Pairs overlap, so
					// these shares do not sum with the per-kernel ones.
					DeviceTimeNanos: int64(ps.totalDur),
					Location: Location{
						Kernel: fmt.Sprintf("%s -> %s", ev1.Name, ev2.Name),
					},
					Finding:                  finding,
					ActionableRecommendation: rec,
					NextMeasurement:          "A/B wall-clock timing with fused kernel",
				})
			}

			// Dual-GEMV Paired Projections
			if isGEMVKernel(ev1.Name) && isGEMVKernel(ev2.Name) {
				n1 := strings.ToLower(ev1.Name)
				n2 := strings.ToLower(ev2.Name)
				isPaired := (ev1.Buffer != "" && ev1.Buffer == ev2.Buffer) ||
					(ev1.Shape != "" && ev1.Shape == ev2.Shape) ||
					(strings.Contains(n1, "qa") && strings.Contains(n2, "kv")) ||
					(strings.Contains(n1, "gate") && strings.Contains(n2, "up")) ||
					(strings.Contains(n1, "q_proj") && strings.Contains(n2, "k_proj"))
				if isPaired {
					seenFusion[key] = true
					finding := fmt.Sprintf("Redundant activation DRAM traffic: sequential GEMVs (%s and %s) on stream %d independently load the identical input activation vector", ev1.Name, ev2.Name, streamID)
					if ps.count > 1 {
						finding += fmt.Sprintf(" (occurred %d times", ps.count)
						if ps.totalDur > 0 {
							finding += fmt.Sprintf(", accumulating %s", time.Duration(ps.totalDur).String())
						}
						finding += ")"
					}
					rec := fmt.Sprintf("Fuse into a paired dual-GEMV dispatch (e.g. fused %s + %s) that loads the activation vector into shared memory once and accumulates dual projection outputs.", cleanName1, cleanName2)

					report.Items = append(report.Items, Item{
						rule:          "shared-activation",
						Severity:      SeverityContext,
						Category:      CategoryMemoryCompute,
						EvidenceLevel: EvidenceSuspected,
						// The pair is the unit of device time here, so the item carries the
						// sum the finding already names. Without it the item reaches a
						// consumer with neither device_time_nanos nor share_of_profile --
						// Location.Kernel is "A -> B", which is no kernel in the profile, so
						// the fill-in below looks it up and finds nothing. Pairs overlap, so
						// these shares do not sum with the per-kernel ones.
						DeviceTimeNanos: int64(ps.totalDur),
						Location: Location{
							Kernel: fmt.Sprintf("%s + %s", ev1.Name, ev2.Name),
						},
						Finding:                  finding,
						ActionableRecommendation: rec,
						NextMeasurement:          "A/B wall-clock timing with paired GEMV dispatch",
					})
				}
			}

			// Pointwise Fusion Opportunity for generic short kernel pairs
			if !seenFusion[key] && ps.count >= 20 && cleanName1 != cleanName2 {
				avgDur := uint64(0)
				if ps.totalDur > 0 {
					avgDur = ps.totalDur / uint64(ps.count*2)
				}
				if avgDur == 0 || avgDur < 30000 {
					seenFusion[key] = true
					finding := fmt.Sprintf("Pointwise kernel fusion opportunity: sequential kernel pair (%s followed by %s) was launched %d times on stream %d", cleanName1, cleanName2, ps.count, streamID)
					if ps.totalDur > 0 {
						finding += fmt.Sprintf(" (accumulating %s)", time.Duration(ps.totalDur).String())
					}
					rec := fmt.Sprintf("Fuse %s and %s into a single kernel to keep intermediate activations in registers/shared memory, eliminating %d kernel launches and DRAM traffic.", cleanName1, cleanName2, ps.count)

					report.Items = append(report.Items, Item{
						rule:          "fusion",
						Severity:      SeverityCost,
						Category:      CategoryMemoryCompute,
						EvidenceLevel: EvidenceSuspected,
						// The pair is the unit of device time here, so the item carries the
						// sum the finding already names. Without it the item reaches a
						// consumer with neither device_time_nanos nor share_of_profile --
						// Location.Kernel is "A -> B", which is no kernel in the profile, so
						// the fill-in below looks it up and finds nothing. Pairs overlap, so
						// these shares do not sum with the per-kernel ones.
						DeviceTimeNanos: int64(ps.totalDur),
						Location: Location{
							Kernel: fmt.Sprintf("%s -> %s", cleanName1, cleanName2),
						},
						Finding:                  finding,
						ActionableRecommendation: rec,
						NextMeasurement:          "A/B wall-clock timing with fused kernel",
					})
				}
			}
		}
	}
}

// journalRemedy and recaptureOrRead keep a diagnosis of missing timestamps
// from turning into a recommendation to produce a file that already exists.
//
// A re-capture costs a run under GPU lock. Telling someone to pay it for a
// capture whose journal is attached is the most expensive thing this tool can
// say, and it was saying it whenever the pprof projection -- which is an
// aggregate and carries no timestamps by construction -- was all that had been
// read.
func captureHasJournal(cap *cudacapture.Capture) bool {
	if cap == nil {
		return false
	}
	for name, data := range cap.Files {
		if cudacapture.IsJournalName(name) && len(data) > 0 {
			return true
		}
	}
	return false
}

func journalRemedy(cap *cudacapture.Capture) string {
	if !captureHasJournal(cap) {
		return ""
	}
	return ". This capture does carry a per-record journal, which holds them"
}

func recaptureOrRead(cap *cudacapture.Capture) string {
	if captureHasJournal(cap) {
		return "Read the journal already in this capture rather than re-running the workload: cudaprof gaps <capture> reports the duty cycle, the idle attribution and the gap distribution from it."
	}
	return "Re-capture with activity timestamps and host launch API records so duty cycle and submission latency can be measured before restructuring dispatch."
}

// unmeasuredLaunchReason states why host submission went unmeasured for a
// sequence, and what to do about it.
//
// Absent from the capture and absent for THESE kernels are different
// statements, and only the first is a reason to capture anything. A capture
// holding host launch records whose correlation ids do not reach a sequence is
// not missing data; something between the launch and the kernel lost the id,
// and that is where to look. Telling someone to capture what they already have
// costs a run under GPU lock and does not produce the missing link.
func unmeasuredLaunchReason(apiRecords int) (why, recommendation string) {
	if apiRecords > 0 {
		return fmt.Sprintf("this capture holds %d host launch record(s) but none correlates to these kernels", apiRecords),
			"Find why the correlation ids of these launches do not reach their kernels before capturing anything further: the host launch records are already here."
	}
	return "host launch timestamps are absent from this capture",
		"Capture host launch API records (correlation-matched) to test whether host submission is behind the device before restructuring dispatch."
}
