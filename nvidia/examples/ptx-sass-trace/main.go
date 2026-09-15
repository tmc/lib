// Ptx-sass-trace captures a checked DeepSeek-V4 CUDA micro-workload for Go trace and
// pprof, including the real PTX, CUBIN, and SASS needed for offline inspection.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
)

type config struct {
	elements int
	passes   int
	rounds   int
	nvdisasm string
	ncclMode string
	devices  []int

	pcSampling bool
	pcPeriod   uint

	hotspots    bool
	graphMode   string
	sassMetrics bool
	pmSampling  bool
}

func main() {
	log.SetFlags(0)
	var cfg config
	out := flag.String("out", "", "new output directory (default: ~/tmp/ptx-sass-trace-*)")
	flag.IntVar(&cfg.elements, "elements", 1<<20, "number of float32 elements")
	flag.IntVar(&cfg.passes, "passes", 12, "measured passes through both kernels, after one warmup")
	flag.IntVar(&cfg.rounds, "rounds", 512, "dependent fused multiply-add operations per compute thread")
	flag.StringVar(&cfg.nvdisasm, "nvdisasm", "nvdisasm", "path to CUDA nvdisasm")
	ncclFlag := flag.String("nccl", "off", "optional NCCL collective mode: off, allreduce, allgather, both")
	flag.BoolVar(&cfg.pcSampling, "pcsampling", false, "collect hardware PC samples; perturbs the CUDA event intervals and requires Linux with CUPTI 130001 or 130400")
	flag.UintVar(&cfg.pcPeriod, "pcsampling-period", 10, "PC sampling period exponent: 2^n GPU cycles, 5 through 31")
	flag.BoolVar(&cfg.hotspots, "hotspots", false, "model per-instruction cost from a static opcode table; reported as modeled_cost, never as a measured counter, and not combinable with -pcsampling")
	graphFlag := flag.String("graph", "off", "CUDA graph execution mode: off, capture, or pdl")
	flag.BoolVar(&cfg.sassMetrics, "sass-metrics", false, "collect CUPTI SASS metrics instrumentation")
	flag.BoolVar(&cfg.pmSampling, "pmsampling", false, "collect CUPTI Performance Monitor (PM) counter availability")
	devicesFlag := flag.String("devices", "", "ordered comma-separated CUDA device ordinals (e.g. 0,1) for NCCL")
	flag.Parse()

	if len(flag.Args()) > 0 {
		log.Fatalf("unexpected positional arguments: %v", flag.Args())
	}

	cfg.graphMode = strings.ToLower(strings.TrimSpace(*graphFlag))
	switch cfg.graphMode {
	case "off", "capture", "pdl":
	default:
		log.Fatalf("unknown -graph mode %q; allowed: off, capture, pdl", *graphFlag)
	}

	cfg.ncclMode = strings.ToLower(strings.TrimSpace(*ncclFlag))
	switch cfg.ncclMode {
	case "off", "allreduce", "allgather", "both":
	default:
		log.Fatalf("unknown -nccl mode %q; allowed: off, allreduce, allgather, both", *ncclFlag)
	}

	// Both modes write a per-instruction row, one measured by the device and
	// one computed from an opcode table. Allowing both would put the two kinds
	// of number in one profile, where a reader has nothing to separate them by:
	// the sample types are the only provenance a pprof file carries.
	if cfg.pcSampling && cfg.hotspots {
		log.Fatalf("-pcsampling and -hotspots are mutually exclusive: -pcsampling measures per-instruction cost, -hotspots models it")
	}

	if cfg.ncclMode == "off" {
		if *devicesFlag != "" {
			log.Fatalf("-devices flag requires -nccl to be set to allreduce, allgather, or both")
		}
	} else {
		if *devicesFlag == "" {
			log.Fatalf("-devices flag is required when -nccl is enabled (e.g. -devices=0,1)")
		}
		parts := strings.Split(*devicesFlag, ",")
		if len(parts) < 2 {
			log.Fatalf("-devices requires at least 2 distinct device ordinals, got %q", *devicesFlag)
		}
		seen := make(map[int]bool)
		for _, part := range parts {
			ord, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil || ord < 0 {
				log.Fatalf("invalid device ordinal %q in -devices", part)
			}
			if seen[ord] {
				log.Fatalf("duplicate device ordinal %d in -devices", ord)
			}
			seen[ord] = true
			cfg.devices = append(cfg.devices, ord)
		}
	}

	if err := run(*out, cfg); err != nil {
		log.Fatal(err)
	}
}

func run(dir string, cfg config) error {
	if cfg.elements < 1 || cfg.elements > 1<<26 || cfg.passes < 1 || cfg.passes > 1000 || cfg.rounds < 1 || cfg.rounds > 1<<20 {
		return fmt.Errorf("require elements in [1, 67108864], passes in [1, 1000], and rounds in [1, 1048576]")
	}
	if dir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		parent := filepath.Join(home, "tmp")
		if err := os.MkdirAll(parent, 0755); err != nil {
			return err
		}
		dir, err = os.MkdirTemp(parent, "ptx-sass-trace-")
		if err != nil {
			return err
		}
	} else if err := os.Mkdir(dir, 0755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	m, err := capture(dir, cfg)
	if err != nil {
		return err
	}
	fmt.Printf("verified %d results; %d elements, %d FMA rounds, %d measured passes\n", m.Verified, cfg.elements, cfg.rounds, cfg.passes)
	names := make([]string, 0, len(m.Intervals))
	for name := range m.Intervals {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		var total int64
		for _, ns := range m.Intervals[name] {
			total += ns
		}
		fmt.Printf("%-24s %d launches, %s CUDA event total\n", name, len(m.Intervals[name]), time.Duration(total))
	}
	if m.NCCL != nil {
		fmt.Printf("nccl mode: %s across %d ranks\n", m.NCCL.Mode, m.NCCL.Ranks)
		for op, rankMap := range m.NCCL.Verified {
			totalVerified := 0
			for _, cnt := range rankMap {
				totalVerified += cnt
			}
			fmt.Printf("nccl %-10s verified %d outputs across all ranks\n", op, totalVerified)
		}
	}
	if m.Graph != nil {
		fmt.Printf("cuda graph mode: %s (%d nodes, %d pdl edges)\n", m.Graph.Mode, m.Graph.NodeCount, m.Graph.PDLEdges)
	}
	if m.SassMetrics != nil {
		fmt.Printf("cupti sass metrics: status=%s (%d census entries, %d supported)\n",
			m.SassMetrics.Status, len(m.SassMetrics.Census), len(m.SassMetrics.SupportedMetrics))
	}
	if m.PMSampling != nil {
		fmt.Printf("cupti pm sampling: status=%s (%d bytes availability image)\n",
			m.PMSampling.Status, m.PMSampling.AvailabilitySize)
	}
	// The interior count is printed beside the divergence count because it is
	// what makes the divergence count meaningful: with no interior NOPs, a
	// Padding divergence is unreachable and the zero says nothing.
	fmt.Printf("ptx->sass: %d divergences, %d NOPs of ptxas padding (%d interior; see lowering.txt)\n",
		m.Divergences, m.PaddingNOPs, m.InteriorNOPs)
	fmt.Printf("capture: %s\n", dir)
	if m.PCSampling != nil {
		fmt.Printf("Hardware PC sampling active: %d samples collected across %d PCs, %.1f%% resolved to PTX.\n",
			m.PCSampling.TotalSamples, m.PCSampling.PCs, 100*m.PCSampling.Resolution())
	} else if m.Hotspots {
		fmt.Println("Instruction hotspot attribution active: opcode cost modeled across SASS/PTX/TileIR instructions.")
	} else {
		fmt.Println("Timings cover whole kernel launches; instruction rows are navigation anchors, not sampled PC costs.")
	}
	return nil
}
