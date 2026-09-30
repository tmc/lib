package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// campaign runs independent ABBA blocks; each launch gets an ordinary-process
// control. Injection and collection overhead is end-to-end, not kernel causal.
type campaignOptions struct {
	domain   string
	dir      string
	commands [2][]string
	blocks   int
	timeout  time.Duration
	marker   *regexp.Regexp
	expect   [2]int
	skip     int
	expected string
	stamps   map[string]string
	flags    []string
	run      func(context.Context, campaignJob) error
	measure  func(string, int) (timeline.Records, int64, error)
	shape    func(timeline.Records, timeline.Records) error
}

type campaignJob struct {
	Command                               []string
	Capture, Stdout, Stderr, CollectorLog string
	Flags                                 []string
	Timeout                               time.Duration
}

type campaignRun struct {
	Arm          string   `json:"arm"`
	Profiled     bool     `json:"profiled"`
	Command      []string `json:"command"`
	Stdout       string   `json:"stdout"`
	Stderr       string   `json:"stderr"`
	CollectorLog string   `json:"collector_log,omitempty"`
	Capture      string   `json:"capture,omitempty"`
	WallNanos    int64    `json:"wall_ns"`
	WindowNanos  int64    `json:"window_ns,omitempty"`
	OutputSHA256 string   `json:"output_sha256,omitempty"`
	Failure      string   `json:"failure,omitempty"`
}

type campaignBlock struct {
	Number                int           `json:"number"`
	Runs                  []campaignRun `json:"runs"`
	Excluded              []string      `json:"excluded,omitempty"`
	DeviceWindowEffectPct *float64      `json:"device_window_effect_pct,omitempty"`
	OverheadPct           []*float64    `json:"profile_overhead_pct,omitempty"`
}

type campaignInterval struct {
	Blocks     int     `json:"independent_blocks"`
	Median     float64 `json:"median_effect_pct"`
	Lower      float64 `json:"conditional_95pct_lower"`
	Upper      float64 `json:"conditional_95pct_upper"`
	Seed       int64   `json:"bootstrap_seed"`
	Assumption string  `json:"assumption"`
}

type campaignReport struct {
	Domain            string            `json:"api_domain"`
	Version           int               `json:"version"`
	ExpectedSHA256    string            `json:"expected_output_sha256"`
	ArtifactSHA256    map[string]string `json:"artifact_sha256"`
	EnvironmentSHA256 string            `json:"environment_sha256"`
	Marker            string            `json:"iteration_marker"`
	ExpectedMarkers   [2]int            `json:"expected_markers"`
	SkipIterations    int               `json:"skip_iterations"`
	CaptureFlags      []string          `json:"capture_flags"`
	Blocks            []campaignBlock   `json:"blocks"`
	Interval          *campaignInterval `json:"interval,omitempty"`
	Limits            string            `json:"limits"`
}

type campaignStamps map[string]string

func (s campaignStamps) String() string { return "role=path" }
func (s campaignStamps) Set(value string) error {
	role, path, ok := strings.Cut(value, "=")
	if !ok || role == "" || path == "" || s[role] != "" {
		return errors.New("stamp-file requires a unique role=path")
	}
	s[role] = path
	return nil
}

func runCampaignCommand(args []string, stdout io.Writer) error {
	o, err := parseCampaignArgs(args, stdout)
	if err != nil || o == nil {
		return err
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		return errors.New("campaign: GPU_LOCK_HELD is unset; run under one existing gpu-lock hold")
	}
	o.run = runCampaignJob
	o.measure = func(path string, arm int) (timeline.Records, int64, error) {
		cap, _, err := readAlignmentCapture(path, cudacapture.Limits{})
		if err != nil {
			return timeline.Records{}, 0, err
		}
		_, records, err := decodeCompleteCapture(cap, 0, 0)
		if err != nil {
			return records, 0, err
		}
		r, err := iteration.Analyze(records, iteration.Options{Marker: o.marker, Expect: o.expect[arm], Skip: o.skip})
		if err != nil {
			return records, 0, err
		}
		if !r.Gate.Pass {
			return records, 0, fmt.Errorf("iteration gate: %s", strings.Join(r.Gate.Failures, "; "))
		}
		var start, end int64
		for _, it := range r.Iterations {
			if it.Steady {
				if start == 0 {
					start = it.Start
				}
				end = it.End
			}
		}
		records, err = campaignSteadyRecords(records, start, end)
		if err != nil {
			return records, 0, err
		}
		return records, r.Steady.WindowNanos, nil
	}
	o.shape = func(a, b timeline.Records) error {
		r := shape.Compare(a, b, o.domain)
		if !r.Pass {
			return fmt.Errorf("shape gate: %s", strings.Join(r.Failures, "; "))
		}
		return nil
	}
	return runCampaign(o, stdout)
}

func parseCampaignArgs(args []string, stdout io.Writer) (*campaignOptions, error) {
	o := &campaignOptions{stamps: make(map[string]string)}
	fs := flag.NewFlagSet("campaign", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: cudaprof campaign -o dir -base command -target command [options] -- capture flags")
		fs.PrintDefaults()
	}
	fs.SetOutput(stdout)
	fs.StringVar(&o.domain, "domain", "driver", "submission API domain: driver or runtime")
	fs.StringVar(&o.dir, "o", "", "new output directory")
	base := fs.String("base", "", "reference command")
	target := fs.String("target", "", "target command")
	fs.IntVar(&o.blocks, "blocks", 3, "independent ABBA blocks (3 to 100)")
	fs.DurationVar(&o.timeout, "timeout", time.Minute, "per-launch timeout (at most 1 hour)")
	marker := fs.String("iteration-marker", "", "required kernel boundary regex")
	fs.IntVar(&o.expect[0], "base-expect-markers", 0, "required reference boundary count")
	fs.IntVar(&o.expect[1], "target-expect-markers", 0, "required target boundary count")
	fs.IntVar(&o.skip, "skip-iterations", -1, "explicit warmup iteration count")
	fs.StringVar(&o.expected, "expected-output", "", "file containing exact deterministic target stdout")
	fs.Var(campaignStamps(o.stamps), "stamp-file", "required source=path and input=path; optional model=path")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil, nil
		}
		return nil, err
	}
	if o.dir == "" || *base == "" || *target == "" || o.expected == "" || o.stamps["source"] == "" || o.stamps["input"] == "" || *marker == "" || o.expect[0] < 2 || o.expect[1] < 2 || o.skip < 0 {
		return nil, errors.New("campaign: output, commands, expected output, source/input stamps, iteration marker, per-arm marker counts and explicit warmup are required")
	}
	if o.domain != "driver" && o.domain != "runtime" {
		return nil, errors.New("campaign: invalid API domain")
	}
	if o.blocks < 3 || o.blocks > 100 || o.timeout <= 0 || o.timeout > time.Hour || o.skip >= min(o.expect[0], o.expect[1])-1 {
		return nil, errors.New("campaign: invalid block count, timeout or warmup")
	}
	var err error
	o.marker, err = regexp.Compile(*marker)
	if err != nil {
		return nil, fmt.Errorf("campaign marker: %w", err)
	}
	for i, s := range []string{*base, *target} {
		o.commands[i], err = splitWords(s)
		if err != nil || len(o.commands[i]) == 0 {
			return nil, errors.New("campaign: invalid command")
		}
	}
	o.flags = fs.Args()
	for _, f := range o.flags {
		name, _, _ := strings.Cut(f, "=")
		if name == "-o" || name == "--output" || name == "--" {
			return nil, errors.New("campaign: capture output and command are managed by campaign")
		}
	}
	// Parse before any workload starts, including collector configuration errors.
	for _, cmd := range o.commands {
		capture, err := parseCaptureArgs(append(append(slices.Clone(o.flags), "-o", filepath.Join(o.dir, "preflight.pprof"), "--"), cmd...))
		if err != nil {
			return nil, fmt.Errorf("campaign capture flags: %w", err)
		}
		if capture.NoInject || capture.Inject == "" {
			return nil, errors.New("campaign: an injection library is required for profiled runs")
		}
	}
	return o, nil
}

func campaignHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func campaignArtifacts(o *campaignOptions) (map[string]string, error) {
	out := make(map[string]string)
	for role, path := range o.stamps {
		h, err := campaignHash(path)
		if err != nil {
			return nil, fmt.Errorf("stamp %s: %w", role, err)
		}
		out[role] = h
	}
	for i, cmd := range o.commands {
		p, err := exec.LookPath(cmd[0])
		if err != nil {
			return nil, err
		}
		h, err := campaignHash(p)
		if err != nil {
			return nil, err
		}
		out[[]string{"base_binary", "target_binary"}[i]] = h
	}
	return out, nil
}

func runCampaign(o *campaignOptions, stdout io.Writer) error {
	expected, err := campaignHash(o.expected)
	if err != nil {
		return err
	}
	artifacts, err := campaignArtifacts(o)
	if err != nil {
		return err
	}
	env := os.Environ()
	slices.Sort(env)
	envHash := sha256.Sum256([]byte(strings.Join(env, "\x00")))
	r := campaignReport{Version: 1, Domain: o.domain, ExpectedSHA256: expected, ArtifactSHA256: artifacts, EnvironmentSHA256: hex.EncodeToString(envHash[:]), Marker: o.marker.String(), ExpectedMarkers: o.expect, SkipIterations: o.skip, CaptureFlags: o.flags,
		Limits: "ABBA block delivery uses retained steady device windows. Bootstrap is conditional on unverified exchangeability of independent blocks. Ordinary-process controls measure end-to-end launch/injection/collection overhead, not kernel causal overhead. Output hashes establish only equality to supplied expected bytes. GPU_LOCK_HELD is an inherited lock declaration, not proof of absence of other GPU users."}
	if err := os.Mkdir(o.dir, 0700); err != nil {
		return fmt.Errorf("campaign output: %w", err)
	}
	check := func() error {
		now, err := campaignArtifacts(o)
		if err != nil {
			return err
		}
		for role, h := range artifacts {
			if now[role] != h {
				return fmt.Errorf("artifact changed: %s", role)
			}
		}
		h, err := campaignHash(o.expected)
		if err != nil {
			return err
		}
		if h != expected {
			return errors.New("expected output changed")
		}
		return nil
	}
	var effects []float64
	for block := 0; block < o.blocks; block++ {
		b := campaignBlock{Number: block + 1}
		var windows [4]int64
		var walls [4][2]int64
		var records [4]timeline.Records
		for pos, arm := range []int{0, 1, 1, 0} {
			modes := []bool{false, true}
			if pos >= 2 {
				modes = []bool{true, false}
			}
			for _, profiled := range modes {
				stem := filepath.Join(o.dir, fmt.Sprintf("block-%03d-%s-%s", block+1, qualifyArms[pos], map[bool]string{false: "ordinary", true: "profiled"}[profiled]))
				run := campaignRun{Arm: qualifyArms[pos], Profiled: profiled, Command: o.commands[arm], Stdout: stem + ".stdout", Stderr: stem + ".stderr"}
				if profiled {
					run.Capture = stem + ".pprof"
					run.CollectorLog = stem + ".collector.log"
				}
				err := check()
				if err == nil {
					ctx, cancel := context.WithTimeout(context.Background(), o.timeout)
					start := time.Now()
					err = o.run(ctx, campaignJob{Command: run.Command, Capture: run.Capture, Stdout: run.Stdout, Stderr: run.Stderr, CollectorLog: run.CollectorLog, Flags: o.flags, Timeout: o.timeout})
					run.WallNanos = time.Since(start).Nanoseconds()
					cancel()
					if err == nil {
						err = check()
					}
				}
				if h, e := campaignHash(run.Stdout); e == nil {
					run.OutputSHA256 = h
					if err == nil && h != expected {
						err = errors.New("stdout differs from expected output")
					}
				} else if err == nil {
					err = e
				}
				mode := 0
				if profiled {
					mode = 1
				}
				walls[pos][mode] = run.WallNanos
				if profiled && err == nil {
					records[pos], run.WindowNanos, err = o.measure(run.Capture, arm)
					windows[pos] = run.WindowNanos
				}
				if err != nil {
					run.Failure = err.Error()
					b.Excluded = append(b.Excluded, run.Arm+": "+run.Failure)
				}
				b.Runs = append(b.Runs, run)
			}
		}
		if len(b.Excluded) == 0 {
			for _, pair := range [][2]int{{0, 1}, {3, 2}} {
				if err := o.shape(records[pair[0]], records[pair[1]]); err != nil {
					b.Excluded = append(b.Excluded, err.Error())
				}
			}
		}
		if len(b.Excluded) == 0 {
			if windows[0] <= 0 || windows[3] <= 0 {
				b.Excluded = append(b.Excluded, "empty reference window")
			} else {
				effect := (float64(windows[1]-windows[0]) + float64(windows[2]-windows[3])) / float64(windows[0]+windows[3]) * 100
				b.DeviceWindowEffectPct = &effect
				effects = append(effects, effect)
				b.OverheadPct = make([]*float64, 2)
				for arm, positions := range [][2]int{{0, 3}, {1, 2}} {
					ordinary := walls[positions[0]][0] + walls[positions[1]][0]
					profiled := walls[positions[0]][1] + walls[positions[1]][1]
					if ordinary > 0 {
						overhead := float64(profiled-ordinary) / float64(ordinary) * 100
						b.OverheadPct[arm] = &overhead
					}
				}
			}
		}
		r.Blocks = append(r.Blocks, b)
	}
	if len(effects) >= 3 {
		v := campaignBootstrap(effects)
		r.Interval = &v
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.WriteFile(filepath.Join(o.dir, "campaign.json"), data, 0600); err != nil {
		return err
	}
	if _, err := stdout.Write(data); err != nil {
		return err
	}
	if r.Interval == nil {
		return errors.New("campaign: fewer than three qualified independent blocks; see campaign.json exclusions")
	}
	return nil
}

func campaignBootstrap(effects []float64) campaignInterval {
	const seed = 1
	rng := rand.New(rand.NewSource(seed))
	med := func(v []float64) float64 {
		slices.Sort(v)
		if len(v)%2 != 0 {
			return v[len(v)/2]
		}
		return (v[len(v)/2-1] + v[len(v)/2]) / 2
	}
	draws := make([]float64, 10000)
	for i := range draws {
		sample := make([]float64, len(effects))
		for j := range sample {
			sample[j] = effects[rng.Intn(len(effects))]
		}
		draws[i] = med(sample)
	}
	slices.Sort(draws)
	return campaignInterval{Blocks: len(effects), Median: med(slices.Clone(effects)), Lower: draws[249], Upper: draws[9749], Seed: seed, Assumption: "conditional percentile block bootstrap; independent exchangeable blocks assumed, exchangeability unverified; launches are not independent samples"}
}

func runCampaignJob(ctx context.Context, job campaignJob) error {
	out, err := os.OpenFile(job.Stdout, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer out.Close()
	stderr, err := os.OpenFile(job.Stderr, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer stderr.Close()
	if job.Capture == "" {
		cmd := exec.CommandContext(ctx, job.Command[0], job.Command[1:]...)
		cmd.Stdout, cmd.Stderr = out, stderr
		cmd.Stdin = os.Stdin
		cmd.WaitDelay = time.Second
		return cmd.Run()
	}
	log, err := os.OpenFile(job.CollectorLog, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer log.Close()
	opts, err := parseCaptureArgs(append(append(slices.Clone(job.Flags), "-o", job.Capture, "--"), job.Command...))
	if err != nil {
		return err
	}
	opts.Stdout, opts.Stderr, opts.Log, opts.Timeout = out, stderr, log, job.Timeout
	result, runErr := runCapture(ctx, opts)
	budget := captureArtifactBudget{FileBytes: opts.CaptureFileBytes, TotalBytes: opts.CaptureTotalBytes}
	result.recordLimits(opts.MaxJournalRecords, budget)
	if _, err := writeCapture(job.Capture, result, budget); err != nil {
		return err
	}
	if runErr != nil {
		return runErr
	}
	if !result.CaptureSuccess {
		return fmt.Errorf("capture unsuccessful: %s", result.Outcome)
	}
	return nil
}

// campaignSteadyRecords refuses partially observed work at the selected edges.
// Full containment is required rather than silently dropping or clipping a
// launch whose duration extends beyond the qualified observation window.
func campaignSteadyRecords(records timeline.Records, start, end int64) (timeline.Records, error) {
	if end <= start {
		return records, errors.New("campaign: empty steady window")
	}
	var acts []timeline.Activity
	for i, a := range records.Activities {
		if a.Start < end && a.End > start {
			if a.Start < start || a.End > end {
				return records, fmt.Errorf("campaign: activity %d in journal %q crosses steady window boundary", i, a.Journal)
			}
			acts = append(acts, a)
		}
	}
	if len(acts) == 0 {
		return records, errors.New("campaign: no fully contained steady activities")
	}
	records.Activities = acts
	return records, nil
}
