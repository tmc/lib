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
	flag.BoolVar(&cfg.pcSampling, "pcsampling", false, "collect hardware PC samples; perturbs the CUDA event intervals and requires Linux with CUPTI 130001")
	flag.UintVar(&cfg.pcPeriod, "pcsampling-period", 10, "PC sampling period exponent: 2^n GPU cycles, 5 through 31")
	devicesFlag := flag.String("devices", "", "ordered comma-separated CUDA device ordinals (e.g. 0,1) for NCCL")
	flag.Parse()

	if len(flag.Args()) > 0 {
		log.Fatalf("unexpected positional arguments: %v", flag.Args())
	}

	cfg.ncclMode = strings.ToLower(strings.TrimSpace(*ncclFlag))
	switch cfg.ncclMode {
	case "off", "allreduce", "allgather", "both":
	default:
		log.Fatalf("unknown -nccl mode %q; allowed: off, allreduce, allgather, both", *ncclFlag)
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
	fmt.Printf("capture: %s\n", dir)
	fmt.Printf("share: %s\n", filepath.Join(dir, "capture.pprof"))
	fmt.Println("Timings cover whole kernel launches; instruction rows are navigation anchors, not sampled PC costs.")
	return nil
}
