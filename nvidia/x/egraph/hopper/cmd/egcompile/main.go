// Egcompile compiles a HuggingFace model architecture config.json into
// multi-bucket Hopper/Blackwell CUDA C++ kernels, TVM-FFI headers, or JSON
// cost plans.
//
// Usage:
//
//	egcompile -config config.json [-device "NVIDIA H200"] [-m 1,16,128] [-seq 1024] [-format cuda|tvmffi|json]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("egcompile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	configPath := fs.String("config", "", "path to model config.json")
	device := fs.String("device", "NVIDIA H200", "target GPU device name")
	mBuckets := fs.String("m", "1,16,128", "comma-separated M batch/token buckets")
	seqLen := fs.Int("seq", 1024, "maximum sequence length")
	format := fs.String("format", "cuda", "output format: cuda, tvmffi, or json")
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: egcompile -config config.json [-device name] [-m buckets] [-seq len] [-format cuda|tvmffi|json]\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	cfgFile := strings.TrimSpace(*configPath)
	if cfgFile == "" && fs.NArg() == 1 {
		cfgFile = fs.Arg(0)
	}
	if cfgFile == "" || fs.NArg() > 1 {
		fs.Usage()
		return 2
	}
	if err := compileAndWrite(cfgFile, *device, *mBuckets, *seqLen, *format, stdout); err != nil {
		fmt.Fprintf(stderr, "egcompile: %v\n", err)
		return 1
	}
	return 0
}

func compileAndWrite(cfgFile, device, mSpec string, seqLen int, format string, stdout io.Writer) error {
	buckets, err := parseBuckets(mSpec)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(cfgFile)
	if err != nil {
		return err
	}
	cfg, err := hopper.ParseArchConfig(data)
	if err != nil {
		return fmt.Errorf("%s: %w", cfgFile, err)
	}
	bundle, err := hopper.CompileMultiBucketLayer(cfg, buckets, seqLen, device)
	if err != nil {
		return err
	}
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "cuda":
		_, err = io.WriteString(stdout, bundle.CUDASource)
		return err
	case "tvmffi":
		_, err = io.WriteString(stdout, bundle.TVMFFISource)
		return err
	case "json":
		enc, err := json.MarshalIndent(bundle, "", "  ")
		if err != nil {
			return err
		}
		_, err = stdout.Write(append(enc, '\n'))
		return err
	default:
		return fmt.Errorf("unknown output format %q (want cuda, tvmffi, or json)", format)
	}
}

func parseBuckets(spec string) ([]int, error) {
	parts := strings.Split(spec, ",")
	var buckets []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil || v <= 0 {
			return nil, fmt.Errorf("invalid M bucket %q", p)
		}
		buckets = append(buckets, v)
	}
	if len(buckets) == 0 {
		return nil, fmt.Errorf("no M buckets specified")
	}
	return buckets, nil
}
