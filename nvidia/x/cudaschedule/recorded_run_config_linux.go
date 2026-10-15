//go:build linux

package cudaschedule

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"time"
)

type recordedConfig struct {
	Version       int             `json:"version"`
	Nonce         string          `json:"nonce"`
	Controller    recordedProcess `json:"controller"`
	Directory     string          `json:"directory"`
	Helper        string          `json:"helper"`
	HelperDigest  string          `json:"helper_sha256"`
	Worker        string          `json:"worker"`
	WorkerDigest  string          `json:"worker_sha256"`
	Args          []string        `json:"args"`
	WorkDir       string          `json:"work_dir"`
	PTXPath       string          `json:"ptx_path"`
	PTXDigest     string          `json:"ptx_sha256"`
	LockScript    string          `json:"lock_script"`
	LockDigest    string          `json:"lock_sha256"`
	LockDirectory string          `json:"lock_directory"`
	Label         string          `json:"label"`
	Limits        recordedLimits  `json:"limits"`
	Expected      time.Duration   `json:"expected"`
	OutputLimit   int             `json:"output_limit"`
}

func (c recordedConfig) validate() error {
	if c.Version != recordedProtocolVersion || !validRecordedNonce(c.Nonce) {
		return fmt.Errorf("recorded helper: invalid configuration identity")
	}
	if c.Controller.PID <= 1 || c.Controller.Parent <= 0 || c.Controller.Group <= 1 || c.Controller.Start == 0 || !validPTXDigest(c.Controller.Executable) {
		return fmt.Errorf("recorded helper: invalid controller identity")
	}
	for _, path := range []string{c.Directory, c.Helper, c.Worker, c.WorkDir, c.PTXPath, c.LockScript, c.LockDirectory} {
		if !filepath.IsAbs(path) || len(path) > 4096 || strings.IndexByte(path, 0) >= 0 {
			return fmt.Errorf("recorded helper: invalid absolute input path")
		}
	}
	for _, digest := range []string{c.HelperDigest, c.WorkerDigest, c.PTXDigest} {
		if !validPTXDigest(digest) {
			return fmt.Errorf("recorded helper: invalid configuration digest")
		}
	}
	if c.LockDigest != recordedLockDigest {
		return fmt.Errorf("recorded helper: unreviewed lock script")
	}
	if c.Args == nil || len(c.Args) > 64 {
		return fmt.Errorf("recorded helper: argument list missing or exceeds limit")
	}
	argSize := 0
	for _, arg := range c.Args {
		argSize += len(arg)
		if len(arg) > 4096 || argSize > 32<<10 || strings.IndexByte(arg, 0) >= 0 {
			return fmt.Errorf("recorded helper: argument size exceeds limit")
		}
	}
	if len(c.Label) == 0 || len(c.Label) > 256 || strings.ContainsAny(c.Label, "\x00\r\n") {
		return fmt.Errorf("recorded helper: invalid lock label")
	}
	if _, err := c.Limits.validate(); err != nil {
		return err
	}
	// gpu-lock accepts whole seconds for queue/estimate/lease settings. Runtime
	// and admission can be shorter; their independent keeper clocks remain exact.
	if c.Limits.Queue%time.Second != 0 || c.Limits.Lease%time.Second != 0 || c.Expected <= 0 || c.Expected > c.Limits.Lease || c.Expected%time.Second != 0 {
		return fmt.Errorf("recorded helper: lock durations require bounded whole seconds")
	}
	if _, err := newRecordedOutput(c.OutputLimit); err != nil {
		return err
	}
	if filepath.Dir(c.Helper) != c.Directory || filepath.Dir(c.Worker) != c.Directory || filepath.Dir(c.PTXPath) != c.Directory || filepath.Dir(c.LockScript) != c.Directory {
		return fmt.Errorf("recorded helper: executable or PTX is outside owned directory")
	}
	return nil
}

func readRecordedConfig(data []byte) (recordedConfig, error) {
	var config recordedConfig
	if len(data) == 0 || len(data) > recordedMessageLimit {
		return config, fmt.Errorf("recorded helper: configuration size out of range")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&config); err != nil {
		return config, fmt.Errorf("recorded helper: configuration: %w", err)
	}
	// Configurations are generated canonical documents, not an open-ended JSON
	// API. Exact round-trip equality also refuses duplicate/case-folded keys,
	// ignored zero-valued fields, null aliases and trailing documents.
	canonical, err := json.Marshal(config)
	if err != nil || !bytes.Equal(data, canonical) {
		return config, fmt.Errorf("recorded helper: noncanonical configuration")
	}
	if err := config.validate(); err != nil {
		return config, err
	}
	return config, nil
}
