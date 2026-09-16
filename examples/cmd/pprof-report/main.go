// Command pprof-report reads a cudaprof / pprof capture (such as a PC-sampled GPU profile)
// and produces a wide, token-efficient summary table across all sample types and stall categories.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

type kernelMetrics struct {
	name   string
	values []int64
}

type locationMetric struct {
	kernel   string
	file     string
	line     int64
	address  uint64
	values   []int64
	key      string
}

func main() {
	topN := flag.Int("top", 10, "number of top kernels / locations to display per category")
	asJSON := flag.Bool("json", false, "output report as structured JSON")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: pprof-report [flags] <capture.pprof>\n\nFlags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}

	targetPath := flag.Arg(0)
	if err := runReport(os.Stdout, targetPath, *topN, *asJSON); err != nil {
		fmt.Fprintf(os.Stderr, "pprof-report: %v\n", err)
		os.Exit(1)
	}
}

func runReport(w io.Writer, path string, topN int, outputJSON bool) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open profile: %w", err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return fmt.Errorf("read profile data: %w", err)
	}

	var p *profile.Profile
	var metaBytes []byte

	// 1. Attempt reading as embedded cudacapture
	cap, capErr := cudacapture.Read(bytes.NewReader(data))
	if capErr == nil && cap != nil && cap.Profile != nil {
		p = cap.Profile
		metaBytes = cap.Metadata
	} else {
		// 2. Fall back to standard pprof parser
		parsed, err := profile.ParseData(data)
		if err != nil {
			return fmt.Errorf("parse pprof: %w", err)
		}
		p = parsed
	}

	if len(p.SampleType) == 0 {
		return fmt.Errorf("profile has no sample types")
	}

	numTypes := len(p.SampleType)
	typeTotals := make([]int64, numTypes)
	kernels := make(map[string]*kernelMetrics)
	locations := make(map[string]*locationMetric)

	for _, s := range p.Sample {
		if len(s.Value) < numTypes {
			continue
		}
		for i := 0; i < numTypes; i++ {
			typeTotals[i] += s.Value[i]
		}

		kName := "unknown"
		var leafLoc *profile.Location
		if len(s.Location) > 0 {
			leafLoc = s.Location[0]
			for _, line := range leafLoc.Line {
				if line.Function != nil && line.Function.Name != "" {
					kName = cleanKernelName(line.Function.Name)
					break
				}
			}
		}
		if kName == "unknown" && s.Label != nil && len(s.Label["kernel"]) > 0 {
			kName = s.Label["kernel"][0]
		}

		km, ok := kernels[kName]
		if !ok {
			km = &kernelMetrics{name: kName, values: make([]int64, numTypes)}
			kernels[kName] = km
		}
		for i := 0; i < numTypes; i++ {
			km.values[i] += s.Value[i]
		}

		if leafLoc != nil {
			var file string
			var lineNum int64
			if len(leafLoc.Line) > 0 {
				if leafLoc.Line[0].Function != nil {
					file = leafLoc.Line[0].Function.Filename
				}
				lineNum = leafLoc.Line[0].Line
			}
			locKey := fmt.Sprintf("%s|0x%x|%s:%d", kName, leafLoc.Address, filepath.Base(file), lineNum)
			lm, ok := locations[locKey]
			if !ok {
				lm = &locationMetric{
					kernel:  kName,
					file:    filepath.Base(file),
					line:    lineNum,
					address: leafLoc.Address,
					values:  make([]int64, numTypes),
					key:     locKey,
				}
				locations[locKey] = lm
			}
			for i := 0; i < numTypes; i++ {
				lm.values[i] += s.Value[i]
			}
		}
	}

	if outputJSON {
		return emitJSON(w, p, metaBytes, typeTotals, kernels, locations, topN)
	}

	emitMarkdown(w, path, p, metaBytes, typeTotals, kernels, locations, topN)
	return nil
}

func cleanKernelName(name string) string {
	name = strings.TrimSuffix(name, ".SASS")
	name = strings.TrimSuffix(name, ".PTX")
	name = strings.TrimSuffix(name, ".TILEIR")
	name = strings.TrimSuffix(name, ".inlined")
	return name
}

func emitMarkdown(w io.Writer, path string, p *profile.Profile, metaBytes []byte, totals []int64, kernels map[string]*kernelMetrics, locations map[string]*locationMetric, topN int) {
	fmt.Fprintf(w, "# cudaprof Profile Report: %s\n\n", filepath.Base(path))

	// Metadata comments
	if len(p.Comments) > 0 {
		fmt.Fprintf(w, "### Profile Provenance & Comments\n")
		for _, c := range p.Comments {
			if !strings.HasPrefix(c, "#cudaprof:capture:") {
				fmt.Fprintf(w, "- %s\n", c)
			}
		}
		fmt.Fprintln(w)
	}

	if len(metaBytes) > 0 {
		var meta map[string]any
		if err := json.Unmarshal(metaBytes, &meta); err == nil {
			fmt.Fprintf(w, "### Device & Capture Metadata\n")
			if dev, ok := meta["device"]; ok {
				fmt.Fprintf(w, "- **Device**: %v (%v)\n", dev, meta["architecture"])
			}
			if dur, ok := meta["duration_ms"]; ok {
				fmt.Fprintf(w, "- **Workload Duration**: %v ms\n", dur)
			}
			if total, ok := meta["total_samples"]; ok {
				fmt.Fprintf(w, "- **Total Hardware Samples**: %v\n", total)
			}
			if res, ok := meta["resolution_percent"]; ok {
				fmt.Fprintf(w, "- **PTX Resolution Rate**: %.2f%%\n", res)
			}
			if dropped, ok := meta["dropped_samples"]; ok {
				fmt.Fprintf(w, "- **Dropped Samples**: %v\n", dropped)
			}
			fmt.Fprintln(w)
		}
	}

	// 1. Metric Overview Ledger
	fmt.Fprintf(w, "### 1. Global Metric Ledger\n\n")
	fmt.Fprintf(w, "| Metric Index | Sample Type | Unit | Total Count | Relative Share (of Primary) |\n")
	fmt.Fprintf(w, "|:---:|:---|:---:|---:|---:|\n")

	primaryIdx := 0
	for i, st := range p.SampleType {
		if st.Type == "pc_samples" || st.Type == "samples" {
			primaryIdx = i
			break
		}
	}
	primaryTotal := totals[primaryIdx]

	for i, st := range p.SampleType {
		rel := "-"
		if primaryTotal > 0 && (strings.HasPrefix(st.Type, "stall_") || st.Type == "inst_executed") {
			rel = fmt.Sprintf("%.2f%%", 100.0*float64(totals[i])/float64(primaryTotal))
		}
		fmt.Fprintf(w, "| %2d | `%s` | %s | %12d | %s |\n", i, st.Type, st.Unit, totals[i], rel)
	}
	fmt.Fprintln(w)

	// 2. Wide Kernel Matrix
	type colSpec struct {
		name  string
		index int
	}
	var selectedCols []colSpec
	for i, st := range p.SampleType {
		selectedCols = append(selectedCols, colSpec{name: st.Type, index: i})
	}

	kList := make([]*kernelMetrics, 0, len(kernels))
	for _, km := range kernels {
		kList = append(kList, km)
	}

	// Sort kernels by primary metric descending
	sort.Slice(kList, func(i, j int) bool {
		return kList[i].values[primaryIdx] > kList[j].values[primaryIdx]
	})

	fmt.Fprintf(w, "### 2. Kernel Distribution Across Sample Types (Wide Token-Efficient Matrix)\n\n")
	header := "| Kernel |"
	divider := "|:---|:"
	for _, col := range selectedCols {
		header += fmt.Sprintf(" %s |", col.name)
		divider += "---:|"
	}
	fmt.Fprintln(w, header)
	fmt.Fprintln(w, divider)

	for i, km := range kList {
		if i >= topN && km.values[primaryIdx] == 0 {
			break
		}
		row := fmt.Sprintf("| `%s` |", km.name)
		for _, col := range selectedCols {
			val := km.values[col.index]
			tot := totals[col.index]
			if tot > 0 {
				row += fmt.Sprintf(" %d (%.1f%%) |", val, 100.0*float64(val)/float64(tot))
			} else {
				row += fmt.Sprintf(" %d |", val)
			}
		}
		fmt.Fprintln(w, row)
		if i+1 >= topN && len(kList) > topN {
			break
		}
	}
	fmt.Fprintln(w)

	// 3. Top Hotspots per Major Category
	fmt.Fprintf(w, "### 3. Hotspot Instructions & Locations by Category\n\n")
	locList := make([]*locationMetric, 0, len(locations))
	for _, lm := range locations {
		locList = append(locList, lm)
	}

	for _, col := range selectedCols {
		if totals[col.index] == 0 {
			continue
		}
		sort.Slice(locList, func(i, j int) bool {
			return locList[i].values[col.index] > locList[j].values[col.index]
		})

		if locList[0].values[col.index] == 0 {
			continue
		}

		fmt.Fprintf(w, "#### Top Hotspots: `%s` (Total: %d)\n\n", col.name, totals[col.index])
		fmt.Fprintf(w, "| Kernel | Offset / PC | File:Line | Count | %% of Metric |\n")
		fmt.Fprintf(w, "|:---|---:|:---|---:|---:|\n")

		shown := 0
		for _, lm := range locList {
			c := lm.values[col.index]
			if c == 0 || shown >= 5 {
				break
			}
			pcStr := fmt.Sprintf("0x%04x", lm.address)
			fileLine := "-"
			if lm.file != "" && lm.line > 0 {
				fileLine = fmt.Sprintf("%s:%d", lm.file, lm.line)
			}
			pct := 100.0 * float64(c) / float64(totals[col.index])
			fmt.Fprintf(w, "| `%s` | `%s` | %s | %d | %.2f%% |\n", lm.kernel, pcStr, fileLine, c, pct)
			shown++
		}
		fmt.Fprintln(w)
	}
}

func emitJSON(w io.Writer, p *profile.Profile, metaBytes []byte, totals []int64, kernels map[string]*kernelMetrics, locations map[string]*locationMetric, topN int) error {
	out := map[string]any{
		"sample_types": p.SampleType,
		"totals":       totals,
		"comments":     p.Comments,
	}
	if len(metaBytes) > 0 {
		var meta any
		if err := json.Unmarshal(metaBytes, &meta); err == nil {
			out["metadata"] = meta
		}
	}
	kOut := make([]map[string]any, 0, len(kernels))
	for _, k := range kernels {
		kOut = append(kOut, map[string]any{
			"kernel": k.name,
			"values": k.values,
		})
	}
	out["kernels"] = kOut
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}
