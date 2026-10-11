package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"
)

const schema = "devicequalify/v1"

func main() {
	out := flag.String("o", "", "write the observation JSON to this file (default stdout)")
	width := flag.Int("expect-link-width", 0, "refuse a PCIe link narrower than this (0 records without gating)")
	gen := flag.Int("expect-link-gen", 0, "refuse a PCIe link older than this generation (0 records without gating)")
	uuid := flag.String("device-uuid", "", "required physical GPU UUID to qualify")
	timeout := flag.Duration("smi-timeout", 10*time.Second, "nvidia-smi deadline")
	flag.Parse()
	if *uuid == "" || *timeout <= 0 {
		fmt.Fprintln(os.Stderr, "devicequalify: device UUID and positive timeout are required")
		os.Exit(2)
	}
	if *width < 0 || *gen < 0 {
		fmt.Fprintln(os.Stderr, "devicequalify: link expectations must not be negative")
		os.Exit(2)
	}

	obs := qualify(*timeout, expectation{LinkWidth: *width, LinkGen: *gen, UUID: *uuid})
	text, err := json.MarshalIndent(obs, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "devicequalify:", err)
		os.Exit(2)
	}
	text = append(text, '\n')
	if err := emit(*out, text); err != nil {
		fmt.Fprintln(os.Stderr, "devicequalify:", err)
		os.Exit(2)
	}
	if !obs.Qualifies {
		for _, r := range obs.Refusals {
			fmt.Fprintln(os.Stderr, "devicequalify:", r)
		}
		os.Exit(1)
	}
}

// emit writes the observation, failing rather than dropping it. A
// qualifying device with no retained JSON is not a receipt, so a write
// that goes nowhere must not exit 0.
func emit(path string, text []byte) error {
	if path != "" {
		return os.WriteFile(path, text, 0o644)
	}
	return writeAll(os.Stdout, text)
}

func writeAll(w io.Writer, text []byte) error {
	n, err := w.Write(text)
	if err != nil {
		return err
	}
	if n != len(text) {
		return io.ErrShortWrite
	}
	return nil
}

func qualify(timeout time.Duration, expect expectation) observation {
	host, _ := os.Hostname()
	obs := observation{
		Schema: schema,
		UTC:    time.Now().UTC().Format(time.RFC3339),
		Host:   host,
		Expect: expect,
		CUDA:   probeCUDA(expect.UUID),
		Scope:  "device health at this instant only; it does not describe the interval between two runs",
	}
	obs.GPUs, obs.SMIExit, obs.SMIError = runSMI(timeout, expect.UUID)
	obs.Refusals = judge(obs.GPUs, obs.CUDA, obs.SMIExit, obs.SMIError, expect)
	if obs.Refusals == nil {
		obs.Refusals = []string{}
	}
	obs.Qualifies = len(obs.Refusals) == 0
	return obs
}

func runSMI(timeout time.Duration, uuid string) (gpus []map[string]string, exit int, errText string) {
	query := "--query-gpu=" + strings.Join(smiFields, ",")
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "nvidia-smi", query, "--format=csv,noheader", "--id="+uuid)
	cmd.WaitDelay = time.Second
	out, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return nil, -1, ctx.Err().Error()
		}
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, ee.ExitCode(), strings.TrimSpace(string(ee.Stderr))
		}
		return nil, -1, err.Error()
	}
	gpus, err = parseSMI(string(out))
	if err != nil {
		return nil, 0, err.Error()
	}
	return gpus, 0, ""
}
