package main

import (
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
	flag.Parse()
	if *width < 0 || *gen < 0 {
		fmt.Fprintln(os.Stderr, "devicequalify: link expectations must not be negative")
		os.Exit(2)
	}

	obs := qualify(expectation{LinkWidth: *width, LinkGen: *gen})
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

func qualify(expect expectation) observation {
	host, _ := os.Hostname()
	obs := observation{
		Schema: schema,
		UTC:    time.Now().UTC().Format(time.RFC3339),
		Host:   host,
		Expect: expect,
		CUDA:   probeCUDA(),
		Scope:  "device health at this instant only; it does not describe the interval between two runs",
	}
	obs.GPUs, obs.SMIExit, obs.SMIError = runSMI()
	obs.Refusals = judge(obs.GPUs, obs.CUDA, obs.SMIExit, obs.SMIError, expect)
	if obs.Refusals == nil {
		obs.Refusals = []string{}
	}
	obs.Qualifies = len(obs.Refusals) == 0
	return obs
}

func runSMI() (gpus []map[string]string, exit int, errText string) {
	query := "--query-gpu=" + strings.Join(smiFields, ",")
	cmd := exec.Command("nvidia-smi", query, "--format=csv,noheader")
	out, err := cmd.Output()
	if err != nil {
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
