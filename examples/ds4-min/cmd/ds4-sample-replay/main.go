// Command ds4-sample-replay replays sampler decisions over float32 logits.
package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

type corpusStep struct {
	name   string
	step   int
	path   string
	logits []float32
}

type gridRow struct {
	temperature float32
	topP        float32
	minP        float32
	topK        int
}

type replayReport struct {
	rows     int
	positive int
	changed  int
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "ds4-sample-replay:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("ds4-sample-replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	expectedSteps := fs.Int("steps", 0, "require exactly this many rows in every case")
	decisions := fs.String("verify-decisions", "", "replay actual sampled-decision directory at default parameters")
	prompt := fs.String("prompt-i32", "", "raw int32 LE history for decision verification")
	seed := fs.Uint64("seed", 12345, "decision verification seed")
	headPair := fs.Bool("head-pair", false, "validate and compare two sealed live output-head packets")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *headPair {
		if fs.NArg() != 2 || *decisions != "" || *prompt != "" {
			return errors.New("head-pair requires exactly two packet manifests")
		}
		return compareHeadPackets(fs.Arg(0), fs.Arg(1), stdout)
	}
	if *decisions != "" {
		if fs.NArg() != 0 || *prompt == "" {
			return errors.New("verify-decisions requires prompt-i32 and no positional arguments")
		}
		if err := verifyDecisions(*decisions, *prompt, *expectedSteps, *seed); err != nil {
			return err
		}
		fmt.Fprintln(stdout, "actual-decision sampler replay PASS; stop identity UNREAD (terminal placement only)")
		return nil
	}
	if fs.NArg() != 2 {
		return errors.New("usage: ds4-sample-replay corpus grid")
	}
	steps, err := readCorpus(fs.Arg(0))
	if err != nil {
		return fmt.Errorf("read corpus: %w", err)
	}
	grid, err := readGrid(fs.Arg(1))
	if err != nil {
		return fmt.Errorf("read grid: %w", err)
	}
	if len(steps) == 0 {
		return errors.New("corpus has no steps")
	}
	if len(grid) == 0 {
		return errors.New("grid has no rows")
	}
	if *expectedSteps <= 0 {
		return errors.New("-steps must pin the expected terminal count")
	}
	counts := make(map[string]int)
	for _, step := range steps {
		counts[step.name]++
	}
	for name, count := range counts {
		if count != *expectedSteps {
			return fmt.Errorf("case %q: %d steps, want %d", name, count, *expectedSteps)
		}
	}
	report, err := replay(steps, grid, stdout)
	if err != nil {
		return err
	}
	fmt.Fprintf(stderr, "seeded replay: rows=%d positive-temperature=%d rng-moved=%d\n", report.rows, report.positive, report.changed)
	if report.positive == 0 {
		return errors.New("positive-temperature replay did not advance the RNG")
	}
	return nil
}

func readCorpus(path string) ([]corpusStep, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return readManifest(path)
	}
	var names []string
	err = filepath.Walk(path, func(name string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.EqualFold(filepath.Ext(name), ".f32") {
			names = append(names, name)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(names)
	steps := make([]corpusStep, 0, len(names))
	for _, name := range names {
		base := strings.TrimSuffix(filepath.Base(name), filepath.Ext(name))
		caseName, step := splitStep(base, len(steps))
		steps = append(steps, corpusStep{name: caseName, step: step, path: name})
	}
	sort.SliceStable(steps, func(i, j int) bool {
		if steps[i].name != steps[j].name {
			return steps[i].name < steps[j].name
		}
		return steps[i].step < steps[j].step
	})
	return steps, nil
}

func readManifest(path string) ([]corpusStep, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	baseDir := filepath.Dir(path)
	var steps []corpusStep
	sc := bufio.NewScanner(f)
	line := 0
	for sc.Scan() {
		line++
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 || strings.HasPrefix(fields[0], "#") {
			continue
		}
		if len(fields) != 3 {
			return nil, fmt.Errorf("line %d: want case step path", line)
		}
		step, err := strconv.Atoi(fields[1])
		if err != nil || step < 0 {
			return nil, fmt.Errorf("line %d: invalid step %q", line, fields[1])
		}
		stepPath := fields[2]
		if !filepath.IsAbs(stepPath) {
			stepPath = filepath.Join(baseDir, stepPath)
		}
		steps = append(steps, corpusStep{name: fields[0], step: step, path: stepPath})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	sort.SliceStable(steps, func(i, j int) bool {
		if steps[i].name != steps[j].name {
			return steps[i].name < steps[j].name
		}
		return steps[i].step < steps[j].step
	})
	return steps, nil
}

func splitStep(base string, fallback int) (string, int) {
	const marker = ".step"
	i := strings.LastIndex(base, marker)
	if i < 0 {
		return base, fallback
	}
	n, err := strconv.Atoi(base[i+len(marker):])
	if err != nil || n < 0 {
		return base, fallback
	}
	return base[:i], n
}

func readGrid(path string) ([]gridRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []gridRow
	sc := bufio.NewScanner(f)
	line := 0
	for sc.Scan() {
		line++
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 || strings.HasPrefix(fields[0], "#") {
			continue
		}
		if len(fields) != 4 {
			return nil, fmt.Errorf("line %d: want temperature top_p min_p top_k", line)
		}
		t, err := strconv.ParseFloat(fields[0], 32)
		if err != nil {
			return nil, fmt.Errorf("line %d temperature: %w", line, err)
		}
		topP, err := strconv.ParseFloat(fields[1], 32)
		if err != nil {
			return nil, fmt.Errorf("line %d top_p: %w", line, err)
		}
		minP, err := strconv.ParseFloat(fields[2], 32)
		if err != nil {
			return nil, fmt.Errorf("line %d min_p: %w", line, err)
		}
		topK, err := strconv.Atoi(fields[3])
		if err != nil {
			return nil, fmt.Errorf("line %d top_k: %w", line, err)
		}
		rows = append(rows, gridRow{float32(t), float32(topP), float32(minP), topK})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func replay(steps []corpusStep, grid []gridRow, out io.Writer) (replayReport, error) {
	if len(steps) == 0 || len(grid) == 0 {
		return replayReport{}, errors.New("replay requires steps and sampling parameters")
	}
	for i := range steps {
		if steps[i].logits != nil {
			continue
		}
		data, err := os.ReadFile(steps[i].path)
		if err != nil {
			return replayReport{}, fmt.Errorf("read %s: %w", steps[i].path, err)
		}
		if len(data)%4 != 0 {
			return replayReport{}, fmt.Errorf("%s: %d bytes is not float32 data", steps[i].path, len(data))
		}
		steps[i].logits = make([]float32, len(data)/4)
		for j := range steps[i].logits {
			steps[i].logits[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[j*4:]))
		}
	}
	seen := make(map[string]bool)
	for i, step := range steps {
		want := 0
		if i > 0 && steps[i-1].name == step.name {
			want = steps[i-1].step + 1
		} else {
			if seen[step.name] {
				return replayReport{}, fmt.Errorf("case %q reappears", step.name)
			}
			seen[step.name] = true
		}
		if step.name == "" || step.step != want {
			return replayReport{}, fmt.Errorf("case %q: step %d, want %d", step.name, step.step, want)
		}
		if len(step.logits) == 0 || len(step.logits) != len(steps[0].logits) {
			return replayReport{}, fmt.Errorf("case %q step %d: incompatible vocabulary", step.name, step.step)
		}
		for _, v := range step.logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return replayReport{}, fmt.Errorf("case %q step %d: non-finite logits", step.name, step.step)
			}
		}
	}
	for _, p := range grid {
		if math.IsNaN(float64(p.temperature)) || math.IsInf(float64(p.temperature), 0) || p.temperature < 0 || math.IsNaN(float64(p.topP)) || p.topP < 0 || p.topP > 1 || math.IsNaN(float64(p.minP)) || p.minP < 0 || p.minP > 1 || p.topK < 0 {
			return replayReport{}, errors.New("invalid sampling parameters")
		}
	}
	var report replayReport
	for _, p := range grid {
		for seed := uint64(0); seed <= 255; seed++ {
			if err := replayCase(steps, p, seed, out, &report); err != nil {
				return report, err
			}
		}
		if err := replayCase(steps, p, 12345, out, &report); err != nil {
			return report, err
		}
	}
	return report, nil
}

func replayCase(steps []corpusStep, p gridRow, seed uint64, out io.Writer, report *replayReport) error {
	rng := sampler.RNG{State: seed}
	smp := sampler.New(len(steps[0].logits))
	for _, step := range steps {
		if step.step == 0 {
			rng.State = seed
		}
		before := rng.State
		id := smp.Sample(step.logits, sampler.Params{Temperature: p.temperature, TopP: p.topP, MinP: p.minP, TopK: p.topK}, &rng)
		report.rows++
		if p.temperature > 0 {
			report.positive++
			if rng.State != before {
				report.changed++
			}
		}
		if _, err := fmt.Fprintf(out, "%s %d %d %d %016x\n", step.name, seed, step.step, id, rng.State); err != nil {
			return fmt.Errorf("write replay: %w", err)
		}
	}
	return nil
}
