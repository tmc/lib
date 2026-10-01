package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
)

func runIsolationCommand(args []string, stdout io.Writer) error {
	if len(args) == 0 || args[0] == "help" || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintln(stdout, "usage: cudaprof isolation bind -receipt FILE -o NEW.pprof INPUT.pprof\nretain late parent evidence in a new artifact; never replace the original\n\nExamples:\n  cudaprof isolation bind -receipt observer.json -o bound.pprof capture.pprof\n  cudaprof trace check -require isolation bound.pprof\n\nBinding writes diagnostic evidence; successful publication does not qualify performance. Existing contamination cannot be relabeled clear.")
		return nil
	}
	if args[0] != "bind" {
		return fmt.Errorf("unknown isolation command %q", args[0])
	}
	args, limits, err := parseReadLimits(args[1:])
	if err != nil {
		return err
	}
	fs := flag.NewFlagSet("isolation bind", flag.ContinueOnError)
	fs.SetOutput(stdout)
	receiptPath := fs.String("receipt", "", "parent v1 observer receipt")
	output := fs.String("o", "", "new capture path; existing outputs refused")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return err
	}
	if fs.NArg() != 1 || *receiptPath == "" || *output == "" {
		return fmt.Errorf("usage: cudaprof isolation bind -receipt FILE -o NEW.pprof INPUT.pprof")
	}
	input := fs.Arg(0)
	if err := isolationOutputPath(*output, input, *receiptPath); err != nil {
		return err
	}
	cap, captureHash, err := readAlignmentCapture(input, limits)
	if err != nil {
		return fmt.Errorf("isolation bind: read capture: %w", err)
	}
	source, err := os.Open(*receiptPath)
	if err != nil {
		return fmt.Errorf("isolation bind: open receipt: %w", err)
	}
	raw, readErr := io.ReadAll(io.LimitReader(source, isolation.MaxBytes+1))
	closeErr := source.Close()
	if readErr != nil {
		return readErr
	}
	if closeErr != nil {
		return closeErr
	}
	if len(raw) > isolation.MaxBytes {
		return fmt.Errorf("isolation bind: receipt exceeds size limit")
	}
	_, receiptHash, err := isolation.Read(bytes.NewReader(raw))
	if err != nil {
		return err
	}
	if err := bindIsolationCapture(cap, captureHash, raw, receiptHash); err != nil {
		return err
	}
	inputInfo, err := os.Stat(input)
	if err != nil {
		return err
	}
	if err := publishIsolationCapture(*output, cap, trustedLimits(limits, inputInfo.Size())); err != nil {
		return err
	}
	result := captureIsolation(cap)
	return json.NewEncoder(stdout).Encode(struct {
		Output      string           `json:"output"`
		InputSHA256 string           `json:"input_capture_sha256"`
		Isolation   isolation.Result `json:"gpu_isolation"`
	}{*output, captureHash, result})
}

func isolationOutputPath(output string, inputs ...string) error {
	absolute, err := filepath.Abs(output)
	if err != nil {
		return err
	}
	for _, input := range inputs {
		path, err := filepath.Abs(input)
		if err != nil {
			return err
		}
		if absolute == path {
			return fmt.Errorf("isolation output collides with input")
		}
	}
	if _, err := os.Lstat(output); err == nil {
		info, statErr := os.Stat(output)
		if statErr == nil {
			for _, input := range inputs {
				other, e := os.Stat(input)
				if e == nil && os.SameFile(info, other) {
					return fmt.Errorf("isolation output inode collides with input")
				}
			}
		}
		return fmt.Errorf("isolation output already exists")
	} else if !os.IsNotExist(err) {
		return err
	}
	return nil
}

func isolationCaptureBinding(cap *cudacapture.Capture) (isolation.Binding, error) {
	var envelope struct {
		Status struct {
			Binding isolation.Binding `json:"isolation_binding"`
		} `json:"status"`
	}
	raw := cap.Files["capture-status.json"]
	if len(raw) == 0 {
		return isolation.Binding{}, nil
	}
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return isolation.Binding{}, fmt.Errorf("isolation binding status is null")
	}
	var structure map[string]json.RawMessage
	if err := json.Unmarshal(raw, &structure); err != nil {
		return isolation.Binding{}, err
	}
	if bytes.Equal(bytes.TrimSpace(structure["status"]), []byte("null")) {
		return isolation.Binding{}, fmt.Errorf("isolation capture status is null")
	}
	if err := checkCounterJSONKeys(json.NewDecoder(bytes.NewReader(raw))); err != nil {
		return isolation.Binding{}, fmt.Errorf("isolation binding status keys: %w", err)
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return isolation.Binding{}, fmt.Errorf("isolation binding status: %w", err)
	}
	return envelope.Status.Binding, nil
}
func bindIsolationCapture(cap *cudacapture.Capture, captureHash string, raw []byte, receiptHash string) error {
	if cap.Files == nil {
		cap.Files = map[string][]byte{}
	}
	binding, err := isolationCaptureBinding(cap)
	if err != nil {
		return err
	}
	receipt, digest, err := isolation.Read(bytes.NewReader(raw))
	if err != nil {
		return err
	}
	if digest != receiptHash {
		return fmt.Errorf("isolation receipt supplied hash mismatch")
	}
	if binding.Session != "" {
		left, right := binding, receipt.Binding
		left.GPUs = slices.Clone(left.GPUs)
		right.GPUs = slices.Clone(right.GPUs)
		slices.Sort(left.GPUs)
		slices.Sort(right.GPUs)
		if !reflect.DeepEqual(left, right) {
			return fmt.Errorf("parent isolation receipt contradicts independently retained capture binding")
		}
	}
	var priorStatus struct {
		Status struct {
			Isolation    *isolation.Result `json:"gpu_isolation"`
			ManifestHash string            `json:"isolation_binding_manifest_sha256"`
		} `json:"status"`
	}
	if data := cap.Files["capture-status.json"]; len(data) > 0 {
		if err := json.Unmarshal(data, &priorStatus); err != nil {
			return err
		}
	}
	if data := cap.Files[isolation.Attachment]; len(data) > 0 && priorStatus.Status.Isolation != nil && priorStatus.Status.Isolation.SHA256 != "" && len(cap.Files[isolation.ManifestAttachment]) == 0 {
		h := sha256.Sum256(data)
		if hex.EncodeToString(h[:]) != priorStatus.Status.Isolation.SHA256 {
			return fmt.Errorf("prior isolation receipt hash contradicts capture status")
		}
	}
	m := &isolation.Manifest{Version: 1, Role: "gpu-isolation-binding", Binding: binding, Consumer: "cudaprof/isolation-v1"}
	oldManifest := cap.Files[isolation.ManifestAttachment]
	oldManifestHash := ""
	if len(oldManifest) > 0 {
		m, oldManifestHash, err = isolation.ReadManifest(bytes.NewReader(oldManifest))
		if err != nil {
			return err
		}
		if priorStatus.Status.ManifestHash != oldManifestHash {
			return fmt.Errorf("prior isolation manifest hash contradicts capture status")
		}
		if !reflect.DeepEqual(m.Binding, binding) {
			return fmt.Errorf("isolation manifest contradicts independently retained capture binding")
		}
		previousResult, e := evaluateIsolationManifest(cap, m)
		if e != nil {
			return e
		}
		if priorStatus.Status.Isolation != nil && priorStatus.Status.Isolation.SHA256 != "" && priorStatus.Status.Isolation.SHA256 != previousResult.SHA256 {
			return fmt.Errorf("prior isolation source hash contradicts capture status")
		}
	} else if previous := cap.Files[isolation.Attachment]; len(previous) > 0 {
		hash := sha256.Sum256(previous)
		m.Sources = append(m.Sources, isolation.Source{Attachment: isolation.Attachment, SHA256: hex.EncodeToString(hash[:])})
	}
	if len(m.Sources) >= isolation.MaxSources || len(m.Hops) >= isolation.MaxSources {
		return fmt.Errorf("isolation binding source limit reached")
	}
	name := "gpu-isolation-receipt-" + receiptHash + ".json"
	if previous, ok := cap.Files[name]; ok && !bytes.Equal(previous, raw) {
		return fmt.Errorf("isolation receipt attachment collision")
	}
	cap.Files[name] = append([]byte(nil), raw...)
	found := false
	for _, s := range m.Sources {
		if s.Attachment == name {
			found = true
		}
	}
	if !found {
		m.Sources = append(m.Sources, isolation.Source{Attachment: name, SHA256: receiptHash})
	}
	m.Hops = append(m.Hops, isolation.Hop{CaptureSHA256: captureHash, ReceiptSHA256: receiptHash, PreviousManifestSHA256: oldManifestHash})
	if len(oldManifest) > 0 {
		cap.Files["gpu-isolation-previous-manifest-"+oldManifestHash+".json"] = append([]byte(nil), oldManifest...)
	}
	encoded, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if len(encoded) > isolation.MaxBytes {
		return fmt.Errorf("isolation manifest exceeds size limit")
	}
	if _, _, err := isolation.ReadManifest(bytes.NewReader(encoded)); err != nil {
		return err
	}
	cap.Files[isolation.ManifestAttachment] = encoded
	result, err := evaluateIsolationManifest(cap, m)
	if err != nil {
		return err
	}
	var envelope map[string]json.RawMessage
	originalStatus := cap.Files["capture-status.json"]
	if len(originalStatus) > 0 {
		if err := json.Unmarshal(originalStatus, &envelope); err != nil {
			return err
		}
	} else {
		envelope = map[string]json.RawMessage{}
	}
	var status map[string]json.RawMessage
	if data := envelope["status"]; len(data) > 0 {
		if err := json.Unmarshal(data, &status); err != nil {
			return err
		}
	} else {
		status = map[string]json.RawMessage{}
	}
	status["gpu_isolation"], err = json.Marshal(result)
	if err != nil {
		return err
	}
	hash := sha256.Sum256(encoded)
	status["isolation_binding_manifest_sha256"], _ = json.Marshal(hex.EncodeToString(hash[:]))
	envelope["status"], err = json.Marshal(status)
	if err != nil {
		return err
	}
	if len(originalStatus) > 0 {
		h := sha256.Sum256(originalStatus)
		cap.Files["capture-status-before-isolation-"+hex.EncodeToString(h[:])+".json"] = append([]byte(nil), originalStatus...)
	}
	cap.Files["capture-status.json"], err = json.Marshal(envelope)
	return err
}

func evaluateIsolationManifest(cap *cudacapture.Capture, m *isolation.Manifest) (isolation.Result, error) {
	binding, err := isolationCaptureBinding(cap)
	if err != nil {
		return isolation.Result{}, err
	}
	if !reflect.DeepEqual(binding, m.Binding) {
		return isolation.Result{}, fmt.Errorf("isolation manifest capture binding mismatch")
	}
	for i, hop := range m.Hops {
		if hop.PreviousManifestSHA256 == "" {
			if i > 0 {
				return isolation.Result{}, fmt.Errorf("isolation binding hop missing predecessor")
			}
			continue
		}
		raw, ok := cap.Files["gpu-isolation-previous-manifest-"+hop.PreviousManifestSHA256+".json"]
		if !ok {
			return isolation.Result{}, fmt.Errorf("previous isolation manifest missing")
		}
		previous, digest, err := isolation.ReadManifest(bytes.NewReader(raw))
		if err != nil {
			return isolation.Result{}, err
		}
		for _, old := range previous.Sources {
			found := false
			for _, current := range m.Sources {
				if old == current {
					found = true
				}
			}
			if !found {
				return isolation.Result{}, fmt.Errorf("isolation manifest dropped previous receipt evidence")
			}
		}
		if digest != hop.PreviousManifestSHA256 || !reflect.DeepEqual(previous.Binding, binding) || len(previous.Hops) != i || !reflect.DeepEqual(previous.Hops, m.Hops[:i]) {
			return isolation.Result{}, fmt.Errorf("previous isolation manifest chain mismatch")
		}
	}
	var combined isolation.Result
	for i, s := range m.Sources {
		raw, ok := cap.Files[s.Attachment]
		if !ok {
			return isolation.Result{}, fmt.Errorf("isolation source attachment missing")
		}
		hash := sha256.Sum256(raw)
		digest := hex.EncodeToString(hash[:])
		if digest != s.SHA256 {
			return isolation.Result{}, fmt.Errorf("isolation source attachment hash mismatch")
		}
		receipt, _, readErr := isolation.Read(bytes.NewReader(raw))
		result := isolation.Evaluate(receipt, digest, binding)
		if readErr != nil {
			result.Reasons = []string{readErr.Error()}
		}
		if i == 0 {
			combined = result
			continue
		}
		reasons := append(append([]string(nil), combined.Reasons...), result.Reasons...)
		if result.State == "contaminated" && combined.State != "contaminated" {
			combined = result
		} else if result.State == "unchecked" && combined.State == "clear" {
			combined = result
		}
		combined.Reasons = reasons
	}
	return combined, nil
}

func publishIsolationCapture(output string, cap *cudacapture.Capture, limits cudacapture.Limits) error {
	if err := os.MkdirAll(filepath.Dir(output), 0755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(output), ".cudaprof-isolation-")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	rewritten := *cap
	rewritten.Profile = cap.Profile.Copy()
	comments := rewritten.Profile.Comments[:0]
	for _, comment := range rewritten.Profile.Comments {
		if !strings.HasPrefix(comment, "#cudaprof:capture:") {
			comments = append(comments, comment)
		}
	}
	rewritten.Profile.Comments = comments
	if err := cudacapture.WriteLimits(tmp, &rewritten, limits); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Link(name, output); err != nil {
		return fmt.Errorf("publish isolation capture without replacement: %w", err)
	}
	return nil
}
