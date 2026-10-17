package advise

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

func architectureCompatible(target, req string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	req = strings.ToLower(strings.TrimSpace(req))
	if req == "" || req == "all" || target == req {
		return true
	}
	targetVal := parseSM(target)
	reqVal := parseSM(req)
	if targetVal > 0 && reqVal > 0 {
		return targetVal >= reqVal
	}
	return target >= req
}

func parseSM(s string) int {
	s = strings.TrimPrefix(s, "sm_")
	var num int
	for _, ch := range s {
		if ch >= '0' && ch <= '9' {
			num = num*10 + int(ch-'0')
		} else {
			break
		}
	}
	return num
}

// The device facts below come from the capture alone. Advise is an offline
// analysis: it never opens a local device to fill in what the capture did not
// record, because the analyzing host is not the host that ran the workload.
// A rule that needs a missing fact is skipped and says so.

// determineDeviceLimits looks up the physical max warps per SM and max blocks
// per SM for the capture's architecture via x/cudaoccupancy. It returns
// ok=false when the capture does not record a known architecture.
func determineDeviceLimits(cap *cudacapture.Capture) (maxWarps int, maxBlocks int, ok bool) {
	if cap == nil {
		return 0, 0, false
	}
	arch := parseArchFromCapture(cap)
	if arch == "" {
		return 0, 0, false
	}
	dev, err := cudaoccupancy.LookupDevice(arch)
	if err != nil || dev.MaxWarpsPerSM <= 0 {
		return 0, 0, false
	}
	return dev.MaxWarpsPerSM, dev.MaxBlocksPerSM, true
}

// determineSMCount returns the device SM count recorded in the capture's
// metadata, or else implied by the capture's target architecture.
// It returns ok=false when the capture determines neither.
func determineSMCount(cap *cudacapture.Capture) (int, bool) {
	if cap == nil {
		return 0, false
	}
	if count := parseSMCountFromCapture(cap); count > 0 {
		return count, true
	}
	if arch := parseArchFromCapture(cap); arch != "" {
		if count := smCountForArch(arch); count > 0 {
			return count, true
		}
	}
	return 0, false
}

func parseSMCountFromCapture(cap *cudacapture.Capture) int {
	if cap == nil {
		return 0
	}
	if len(cap.Metadata) > 0 {
		if count := parseSMCountFromJSON(cap.Metadata); count > 0 {
			return count
		}
	}
	for _, name := range []string{"capture-provenance.json", "capture-status.json", "status.json", "manifest.json", "provenance.json"} {
		if data, ok := cap.Files[name]; ok && len(data) > 0 {
			if count := parseSMCountFromJSON(data); count > 0 {
				return count
			}
		}
	}
	if cap.Profile != nil {
		for _, c := range cap.Profile.Comments {
			cLower := strings.ToLower(c)
			switch {
			case strings.Contains(cLower, "gb10"):
				return 48
			case strings.Contains(cLower, "gb200"):
				return 136
			case strings.Contains(cLower, "b200") || strings.Contains(cLower, "b100"):
				return 132
			case strings.Contains(cLower, "h100"):
				return 132
			case strings.Contains(cLower, "rtx 4090"):
				return 128
			case strings.Contains(cLower, "a100"):
				return 108
			}
		}
	}
	return 0
}

func parseSMCountFromJSON(data []byte) int {
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return 0
	}
	return findSMCountInMap(m)
}

func findSMCountInMap(m map[string]any) int {
	keys := []string{"sm_count", "smCount", "multiprocessor_count", "multiprocessorCount", "sms", "sm_num"}
	for _, k := range keys {
		if v, ok := m[k]; ok {
			if f, ok := v.(float64); ok && f > 0 {
				return int(f)
			}
			if s, ok := v.(string); ok {
				if n, err := strconv.Atoi(s); err == nil && n > 0 {
					return n
				}
			}
		}
	}
	if dev, ok := m["device"].(string); ok && dev != "" {
		devLower := strings.ToLower(dev)
		switch {
		case strings.Contains(devLower, "gb10"):
			return 48
		case strings.Contains(devLower, "gb200"):
			return 136
		case strings.Contains(devLower, "b200") || strings.Contains(devLower, "b100"):
			return 132
		case strings.Contains(devLower, "h100"):
			return 132
		case strings.Contains(devLower, "rtx 4090"):
			return 128
		case strings.Contains(devLower, "a100"):
			return 108
		}
	}
	for _, v := range m {
		if subMap, ok := v.(map[string]any); ok {
			if count := findSMCountInMap(subMap); count > 0 {
				return count
			}
		}
	}
	return 0
}

func parseArchFromCapture(cap *cudacapture.Capture) string {
	if cap == nil {
		return ""
	}
	if ptxData, _, _ := capturefiles.PTX(cap); len(ptxData) > 0 {
		if mod, err := cudaptx.ParseBytes(ptxData); err == nil && mod != nil && mod.Target != "" {
			return mod.Target
		}
	}
	if len(cap.Metadata) > 0 {
		var m map[string]any
		if json.Unmarshal(cap.Metadata, &m) == nil {
			if arch := findArchInMap(m); arch != "" {
				return arch
			}
		}
	}
	for _, name := range []string{"capture-provenance.json", "capture-status.json", "status.json", "manifest.json", "provenance.json"} {
		if data, ok := cap.Files[name]; ok && len(data) > 0 {
			var m map[string]any
			if json.Unmarshal(data, &m) == nil {
				if arch := findArchInMap(m); arch != "" {
					return arch
				}
			}
		}
	}
	if sassData, ok := capturefiles.SASS(cap); ok && len(sassData) > 0 {
		lines := strings.SplitN(string(sassData), "\n", 30)
		for _, line := range lines {
			lineLower := strings.ToLower(line)
			if idx := strings.Index(lineLower, "sm_"); idx != -1 {
				end := idx + 3
				for end < len(lineLower) && ((lineLower[end] >= '0' && lineLower[end] <= '9') || lineLower[end] == 'a') {
					end++
				}
				if end > idx+3 {
					return lineLower[idx:end]
				}
			}
		}
	}
	return ""
}

func findArchInMap(m map[string]any) string {
	for _, k := range []string{"arch", "architecture", "target", "compute_capability", "computeCapability", "chip", "chip_name", "sm"} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok && s != "" {
				return s
			}
		}
	}
	for _, v := range m {
		if subMap, ok := v.(map[string]any); ok {
			if arch := findArchInMap(subMap); arch != "" {
				return arch
			}
		}
	}
	return ""
}

func smCountForArch(arch string) int {
	arch = strings.ToLower(strings.TrimSpace(arch))
	arch = strings.TrimPrefix(arch, "compute_")
	arch = strings.TrimPrefix(arch, "sm_")
	switch {
	case strings.HasPrefix(arch, "121"):
		// GB10 (SM 12.1). Measured on hardware; see x/cudaoccupancy SM121.
		// GB200 is sm_100a, not sm_121a.
		return 48
	case strings.HasPrefix(arch, "120"):
		return 136 // Blackwell sm_120
	case strings.HasPrefix(arch, "100"):
		return 132 // Blackwell sm_100 (B100)
	case strings.HasPrefix(arch, "90"):
		return 132 // Hopper H100 SXM (132 SMs) / PCIe (114 SMs)
	case strings.HasPrefix(arch, "89"):
		return 128 // Ada Lovelace AD102 (RTX 4090: 128, L40S: 142)
	case strings.HasPrefix(arch, "86"):
		return 84 // Ampere GA102
	case strings.HasPrefix(arch, "80"):
		return 108 // Ampere A100 SXM (108 SMs)
	case strings.HasPrefix(arch, "75"):
		return 72 // Turing TU102
	case strings.HasPrefix(arch, "70"):
		return 80 // Volta GV100
	default:
		return 0
	}
}

type launchRecord struct {
	Kernel       string
	StreamID     uint32
	BlockThreads int
	GridBlocks   int
	Location     Location
}

func extractLaunches(cap *cudacapture.Capture) []launchRecord {
	if cap == nil {
		return nil
	}
	var launches []launchRecord

	// 1. From cap.Profile.Sample
	if cap.Profile != nil {
		for _, s := range cap.Profile.Sample {
			kName := extractKernelNameFromSample(s)
			if kName == "" {
				continue
			}
			var bThreads, gBlocks int
			if len(s.NumLabel["block_threads"]) > 0 {
				bThreads = int(s.NumLabel["block_threads"][0])
			} else if len(s.NumLabel["threads"]) > 0 {
				bThreads = int(s.NumLabel["threads"][0])
			} else if len(s.NumLabel["block_x"]) > 0 {
				bx := int(s.NumLabel["block_x"][0])
				by, bz := 1, 1
				if len(s.NumLabel["block_y"]) > 0 && s.NumLabel["block_y"][0] > 0 {
					by = int(s.NumLabel["block_y"][0])
				}
				if len(s.NumLabel["block_z"]) > 0 && s.NumLabel["block_z"][0] > 0 {
					bz = int(s.NumLabel["block_z"][0])
				}
				bThreads = bx * by * bz
			} else if len(s.Label["block"]) > 0 {
				bThreads = parseDimProduct(s.Label["block"][0])
			}

			if len(s.NumLabel["grid_blocks"]) > 0 {
				gBlocks = int(s.NumLabel["grid_blocks"][0])
			} else if len(s.NumLabel["blocks"]) > 0 {
				gBlocks = int(s.NumLabel["blocks"][0])
			} else if len(s.NumLabel["grid_x"]) > 0 {
				gx := int(s.NumLabel["grid_x"][0])
				gy, gz := 1, 1
				if len(s.NumLabel["grid_y"]) > 0 && s.NumLabel["grid_y"][0] > 0 {
					gy = int(s.NumLabel["grid_y"][0])
				}
				if len(s.NumLabel["grid_z"]) > 0 && s.NumLabel["grid_z"][0] > 0 {
					gz = int(s.NumLabel["grid_z"][0])
				}
				gBlocks = gx * gy * gz
			} else if len(s.Label["grid"]) > 0 {
				gBlocks = parseDimProduct(s.Label["grid"][0])
			}

			if bThreads > 0 || gBlocks > 0 {
				var loc Location
				loc.Kernel = kName
				if len(s.Location) > 0 && len(s.Location[0].Line) > 0 {
					line := s.Location[0].Line[0]
					if line.Function != nil && line.Function.Filename != "" {
						loc.File = line.Function.Filename
						loc.Line = int(line.Line)
					}
				}
				launches = append(launches, launchRecord{
					Kernel:       kName,
					BlockThreads: bThreads,
					GridBlocks:   gBlocks,
					Location:     loc,
				})
			}
		}
	}

	// 2. From cap.Files (activities.json, capture-status.json, etc.)
	for _, name := range []string{"activities.json", "capture-status.json", "status.json", "trace.json"} {
		if data, ok := cap.Files[name]; ok && len(data) > 0 {
			fileLaunches := parseLaunchesFromJSON(data)
			launches = append(launches, fileLaunches...)
		}
	}

	// 3. From PTX
	if ptxData, ptxFile, _ := capturefiles.PTX(cap); len(ptxData) > 0 {
		if mod, err := cudaptx.ParseBytes(ptxData); err == nil && mod != nil {
			for _, k := range mod.Kernels {
				threads := 0
				if k.Tuning.ReqNTID[0] > 0 {
					threads = k.Tuning.ReqNTID[0] * max(1, k.Tuning.ReqNTID[1]) * max(1, k.Tuning.ReqNTID[2])
				} else if k.Tuning.MaxNTID[0] > 0 {
					threads = k.Tuning.MaxNTID[0] * max(1, k.Tuning.MaxNTID[1]) * max(1, k.Tuning.MaxNTID[2])
				}
				if threads > 0 {
					launches = append(launches, launchRecord{
						Kernel:       k.Name,
						BlockThreads: threads,
						Location: Location{
							Kernel: k.Name,
							File:   ptxFile,
							Line:   k.StartLine,
						},
					})
				}
			}
		}
	}
	return launches
}

func parseDimProduct(s string) int {
	s = strings.Trim(s, "()[]{} ")
	parts := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == 'x' || r == 'X' || r == ' ' || r == '	'
	})
	prod := 1
	valid := false
	for _, p := range parts {
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && n > 0 {
			prod *= n
			valid = true
		}
	}
	if !valid {
		return 0
	}
	return prod
}

func parseDim3FromSlice(items []any) [3]int {
	var res [3]int
	res[0], res[1], res[2] = 1, 1, 1
	for i, it := range items {
		if i >= 3 {
			break
		}
		if f, ok := it.(float64); ok && f > 0 {
			res[i] = int(f)
		}
	}
	return res
}

func parseLaunchesFromJSON(data []byte) []launchRecord {
	var launches []launchRecord
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil
	}

	var items []any
	switch v := raw.(type) {
	case []any:
		items = v
	case map[string]any:
		if acts, ok := v["activities"].([]any); ok {
			items = acts
		} else if status, ok := v["status"].(map[string]any); ok {
			if acts, ok := status["activities"].([]any); ok {
				items = acts
			}
		}
	}

	for _, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		var kName string
		if s, ok := m["name"].(string); ok {
			kName = s
		} else if s, ok := m["kernel"].(string); ok {
			kName = s
		}
		if kName == "" {
			continue
		}

		var bThreads, gBlocks int
		if t, ok := m["threads"].(float64); ok && t > 0 {
			bThreads = int(t)
		}
		if t, ok := m["block_threads"].(float64); ok && t > 0 {
			bThreads = int(t)
		}
		if g, ok := m["grid_blocks"].(float64); ok && g > 0 {
			gBlocks = int(g)
		}
		if g, ok := m["blocks"].(float64); ok && g > 0 {
			gBlocks = int(g)
		}

		if blk, ok := m["block"].([]any); ok && len(blk) > 0 {
			dim := parseDim3FromSlice(blk)
			bThreads = dim[0] * max(1, dim[1]) * max(1, dim[2])
		}
		if grd, ok := m["grid"].([]any); ok && len(grd) > 0 {
			dim := parseDim3FromSlice(grd)
			gBlocks = dim[0] * max(1, dim[1]) * max(1, dim[2])
		}

		if bThreads > 0 || gBlocks > 0 {
			launches = append(launches, launchRecord{
				Kernel:       kName,
				BlockThreads: bThreads,
				GridBlocks:   gBlocks,
				Location:     Location{Kernel: kName},
			})
		}
	}
	return launches
}

func extractKernelNameFromSample(s *profile.Sample) string {
	if s == nil {
		return ""
	}
	if len(s.Label["kernel"]) > 0 && s.Label["kernel"][0] != "" {
		return s.Label["kernel"][0]
	}
	if len(s.Label["name"]) > 0 && s.Label["name"][0] != "" {
		return s.Label["name"][0]
	}
	if len(s.Location) > 0 {
		for _, loc := range s.Location {
			for _, line := range loc.Line {
				if line.Function != nil && line.Function.Name != "" {
					name := line.Function.Name
					name = strings.TrimSuffix(name, ".SASS")
					name = strings.TrimSuffix(name, ".PTX")
					name = strings.TrimSuffix(name, ".TILEIR")
					name = strings.TrimSuffix(name, ".inlined")
					return name
				}
			}
		}
	}
	return ""
}

// sampleGraphID returns the CUDA graph ID recorded on a kernel sample, or zero
// when the sample carries none. CUPTI reports graphId zero for a kernel that is
// not a graph node, and cudacapture omits the label entirely in that case, so
// absence and zero mean the same thing and both answer "not a graph node".
func sampleGraphID(s *profile.Sample) uint32 {
	if s == nil {
		return 0
	}
	if len(s.NumLabel["graph_id"]) > 0 && s.NumLabel["graph_id"][0] > 0 {
		return uint32(s.NumLabel["graph_id"][0])
	}
	if len(s.Label["graph_id"]) > 0 {
		if n, err := strconv.ParseUint(s.Label["graph_id"][0], 10, 32); err == nil {
			return uint32(n)
		}
	}
	return 0
}

// capturedGraphReplays reports how many CUDA graph launches a capture holds.
// It answers zero for a capture with no graph work, and the callers use it only
// to suppress advice that recommends graph capture, never to raise advice.
//
// Two sources of evidence, in order of strength:
//
// Direct. CUPTI records graphId and graphNodeId on every kernel activity record
// that ran as a node of a graph. Each replay is one cudaGraphLaunch and so one
// correlation ID, so counting the distinct correlations among kernel samples
// that carry a graph ID counts replays.
//
// Indirect, for captures taken before the agent forwarded graphId. Under replay
// every node inherits the correlation ID of the single launch that issued it,
// so one correlation spread across several distinct kernel records is the
// device-side signature of a graph launch.
//
// The indirect form has a false negative the direct form does not: a graph of a
// SINGLE node gives each replay one kernel record with its own correlation,
// which is indistinguishable from an ordinary launch by multiplicity alone.
// Captures that carry graphId are not subject to this.
//
// Measured on GB10 (sm_121), one kernel, identical geometry, 20 executions per
// arm: launched 20 times directly the capture held 20 kernel records with 20
// distinct correlation IDs; a four-node graph replayed five times held 20
// kernel records with 5 correlation IDs, four records each, and each of those
// five resolved to a runtime cudaGraphLaunch API record.
func capturedGraphReplays(cap *cudacapture.Capture) int {
	if cap == nil || cap.Profile == nil {
		return 0
	}
	graphLaunches := make(map[uint32]struct{})
	counts := make(map[uint32]int)
	for _, s := range cap.Profile.Sample {
		if extractKernelNameFromSample(s) == "" {
			continue
		}
		corr := timeline.SampleCorrelation(s)
		if sampleGraphID(s) != 0 {
			// A graph node with no correlation still witnesses a replay; it
			// just cannot be told apart from another replay of the same graph.
			graphLaunches[corr] = struct{}{}
			continue
		}
		if corr != 0 {
			counts[corr]++
		}
	}
	if len(graphLaunches) > 0 {
		return len(graphLaunches)
	}
	shared := 0
	for _, n := range counts {
		if n > 1 {
			shared++
		}
	}
	return shared
}

// hasDeviceTiming reports whether the capture carries measured device time.
// A wall-clock duration is not device time: every sample-only profile has one,
// and treating it as timing is what let a share of PC samples be rendered as a
// millisecond figure. Only a kernel_time or gpu_time sample type counts.
func hasDeviceTiming(cap *cudacapture.Capture) bool {
	if cap == nil {
		return false
	}
	if cap.Profile != nil {
		for i, vt := range cap.Profile.SampleType {
			if vt.Type == "kernel_time" || vt.Type == "gpu_time" {
				for _, s := range cap.Profile.Sample {
					if i < len(s.Value) && s.Value[i] > 0 {
						return true
					}
				}
			}
		}
	}
	return false
}

func computeProfileTimes(cap *cudacapture.Capture) (map[string]int64, int64) {
	kernelTimes := make(map[string]int64)
	var totalTime int64
	if cap == nil || cap.Profile == nil {
		return kernelTimes, totalTime
	}
	p := cap.Profile
	timeIdx := -1
	for i, vt := range p.SampleType {
		if vt.Type == "kernel_time" || vt.Type == "gpu_time" {
			timeIdx = i
			break
		}
	}
	if timeIdx >= 0 {
		for _, s := range p.Sample {
			val := int64(0)
			if timeIdx < len(s.Value) {
				val = s.Value[timeIdx]
			}
			totalTime += val
			kName := extractKernelNameFromSample(s)
			if kName != "" {
				kernelTimes[kName] += val
			}
		}
	}
	if totalTime == 0 {
		// Fall back to hardware PC samples or samples when timing is 0 or absent.
		sampleIdx := -1
		for i, vt := range p.SampleType {
			if vt.Type == "pc_samples" || vt.Type == "samples" {
				sampleIdx = i
				break
			}
		}
		if sampleIdx >= 0 {
			var totalSamples int64
			sampleCounts := make(map[string]int64)
			for _, s := range p.Sample {
				val := int64(0)
				if sampleIdx < len(s.Value) {
					val = s.Value[sampleIdx]
				}
				totalSamples += val
				kName := extractKernelNameFromSample(s)
				if kName != "" {
					sampleCounts[kName] += val
				}
			}
			if totalSamples > 0 {
				// Sample counts, not times. Scaling them by the capture's
				// wall-clock duration produces a per-kernel nanosecond figure
				// that was never measured; the caller distinguishes the two by
				// hasDeviceTiming and labels the share accordingly.
				totalTime = totalSamples
				for k, sc := range sampleCounts {
					kernelTimes[k] = sc
				}
			}
		}
	}
	return kernelTimes, totalTime
}
