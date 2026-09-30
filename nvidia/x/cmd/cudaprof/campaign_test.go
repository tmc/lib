package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func testCampaignOptions(t *testing.T) *campaignOptions {
	t.Helper()
	root := t.TempDir()
	stamp := filepath.Join(root, "stamp")
	if err := os.WriteFile(stamp, []byte("expected\n"), 0600); err != nil {
		t.Fatal(err)
	}
	o := &campaignOptions{domain: "driver", dir: filepath.Join(root, "runs"), commands: [2][]string{{"true"}, {"true"}}, blocks: 3, timeout: time.Second, marker: regexp.MustCompile("boundary"), expect: [2]int{3, 3}, skip: 0, expected: stamp, stamps: map[string]string{"source": stamp, "input": stamp}}
	o.run = func(ctx context.Context, job campaignJob) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := os.WriteFile(job.Stderr, nil, 0600); err != nil {
			return err
		}
		return os.WriteFile(job.Stdout, []byte("expected\n"), 0600)
	}
	o.measure = func(_ string, arm int) (timeline.Records, int64, error) {
		return timeline.Records{}, int64(100 + 10*arm), nil
	}
	o.shape = func(a, b timeline.Records) error { return nil }
	return o
}

func TestCampaignOrder(t *testing.T) {
	o := testCampaignOptions(t)
	var order []bool
	run := o.run
	o.run = func(ctx context.Context, j campaignJob) error {
		order = append(order, j.Capture != "")
		return run(ctx, j)
	}
	var out bytes.Buffer
	if err := runCampaign(o, &out); err != nil {
		t.Fatal(err)
	}
	want := []bool{false, true, false, true, true, false, true, false}
	for i := 0; i < 3; i++ {
		if !reflect.DeepEqual(order[i*8:(i+1)*8], want) {
			t.Fatalf("order = %v", order)
		}
	}
	var r campaignReport
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Interval == nil || r.Interval.Blocks != 3 || r.Interval.Median != 10 {
		t.Fatalf("interval = %+v", r.Interval)
	}
	if r.Blocks[0].DeviceWindowEffectPct == nil || *r.Blocks[0].DeviceWindowEffectPct != 10 || len(r.Blocks[0].OverheadPct) != 2 {
		t.Fatalf("qualified block effects = %+v", r.Blocks[0])
	}
	if len(r.Blocks[0].Runs) != 8 || r.EnvironmentSHA256 == "" || r.ArtifactSHA256["base_binary"] == "" {
		t.Fatalf("receipt = %+v", r)
	}
	if _, err := os.Stat(filepath.Join(o.dir, "campaign.json")); err != nil {
		t.Fatal(err)
	}
}

func TestCampaignExclusions(t *testing.T) {
	for _, reason := range []string{"output", "process", "iteration", "shape", "stamp"} {
		t.Run(reason, func(t *testing.T) {
			o := testCampaignOptions(t)
			run := o.run
			o.run = func(ctx context.Context, j campaignJob) error {
				if err := run(ctx, j); err != nil {
					return err
				}
				switch reason {
				case "output":
					return os.WriteFile(j.Stdout, []byte("wrong"), 0600)
				case "process":
					return errors.New("target crashed")
				case "stamp":
					return os.WriteFile(o.stamps["source"], []byte("changed"), 0600)
				}
				return nil
			}
			if reason == "iteration" {
				o.measure = func(string, int) (timeline.Records, int64, error) {
					return timeline.Records{}, 0, errors.New("iteration gate failed")
				}
			}
			if reason == "shape" {
				o.shape = func(timeline.Records, timeline.Records) error { return errors.New("missing shape") }
			}
			var out bytes.Buffer
			if err := runCampaign(o, &out); err == nil {
				t.Fatal("expected refusal")
			}
			var r campaignReport
			if err := json.Unmarshal(out.Bytes(), &r); err != nil {
				t.Fatal(err)
			}
			if r.Interval != nil || len(r.Blocks) != 3 || len(r.Blocks[0].Excluded) == 0 || r.Blocks[0].DeviceWindowEffectPct != nil || r.Blocks[0].OverheadPct != nil {
				t.Fatalf("receipt = %+v", r)
			}
		})
	}
}

func TestCampaignBootstrap(t *testing.T) {
	a := campaignBootstrap([]float64{1, 2, 9, 12})
	b := campaignBootstrap([]float64{1, 2, 9, 12})
	if a != b || a.Blocks != 4 || a.Median != 5.5 || a.Lower < 1 || a.Upper > 12 || a.Assumption == "" {
		t.Fatalf("bootstrap = %+v, %+v", a, b)
	}
}

func TestCampaignRequiredPreflight(t *testing.T) {
	for _, args := range [][]string{nil, {"-blocks=101"}, {"-timeout=2h"}, {"-domain=automatic"}, {"-stamp-file=source=x", "-stamp-file=source=y"}} {
		if _, err := parseCampaignArgs(args, &bytes.Buffer{}); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestCampaignOrdinaryOutput(t *testing.T) {
	dir := t.TempDir()
	j := campaignJob{Command: []string{"sh", "-c", "printf expected; printf warning >&2"}, Stdout: filepath.Join(dir, "stdout"), Stderr: filepath.Join(dir, "stderr")}
	if err := runCampaignJob(context.Background(), j); err != nil {
		t.Fatal(err)
	}
	out, err := os.ReadFile(j.Stdout)
	if err != nil {
		t.Fatal(err)
	}
	stderr, err := os.ReadFile(j.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "expected" || string(stderr) != "warning" {
		t.Fatalf("stdout/stderr = %q/%q", out, stderr)
	}
}

func TestCampaignSteadyBoundaries(t *testing.T) {
	tests := []struct {
		name       string
		start, end int64
		wantErr    bool
	}{
		{"contained", 100, 200, false},
		{"crosses start", 90, 150, true},
		{"crosses end", 150, 210, true},
		{"spans window", 90, 210, true},
		{"touches start", 50, 100, false},
		{"touches end", 200, 250, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			records := timeline.Records{Activities: []timeline.Activity{{Journal: "generation", Start: 100, End: 200}, {Journal: "generation", Start: tt.start, End: tt.end}}}
			got, err := campaignSteadyRecords(records, 100, 200)
			if (err != nil) != tt.wantErr {
				t.Fatalf("window = %+v, %v; want error %v", got, err, tt.wantErr)
			}
			if !tt.wantErr && len(got.Activities) == 0 {
				t.Fatal("contained work omitted")
			}
		})
	}
}

func TestCampaignAPIDomain(t *testing.T) {
	o := testCampaignOptions(t)
	args := []string{"-o", filepath.Join(t.TempDir(), "run"), "-base", "true", "-target", "true", "-expected-output", o.expected, "-stamp-file", "source=" + o.expected, "-stamp-file", "input=" + o.expected, "-iteration-marker", "boundary", "-base-expect-markers", "3", "-target-expect-markers", "3", "-skip-iterations", "0", "-domain", "runtime", "--", "--inject", o.expected}
	parsed, err := parseCampaignArgs(args, &bytes.Buffer{})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.domain != "runtime" {
		t.Fatalf("domain = %q", parsed.domain)
	}
	for i, arg := range args {
		if arg == "runtime" {
			args[i] = "automatic"
		}
	}
	if _, err := parseCampaignArgs(args, &bytes.Buffer{}); err == nil {
		t.Fatal("accepted invalid API domain")
	}
}
