package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"unicode"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

const healthCapacityUsage = "usage: cudaprof health-capacity -health FILE -telemetry FILE -input FILE [-nvsentinel FILE -nvsentinel-sha256 HEX] [-max-bytes N -output-bytes N]"

// This command-specific input keeps the existing admission options separate:
// HealthCapacityInput.Telemetry deliberately has json:"-" in the public API.
type healthCapacityCommandInput struct {
	Telemetry      cudadiag.TelemetryOptions     `json:"telemetry"`
	HealthLimits   nvtelemetry.HealthLimits      `json:"health_limits"`
	HealthFindings []tracecontract.Finding       `json:"health_findings,omitempty"`
	Work           []nvtelemetry.AccountingWork  `json:"work,omitempty"`
	Limits         cudadiag.HealthCapacityLimits `json:"limits"`
}

type healthCapacityCommandOutput struct {
	ConfigurationSHA256 string                          `json:"configuration_sha256"`
	Incident            cudadiag.HealthCapacityIncident `json:"incident"`
	Validation          *nvSentinelImport               `json:"validation,omitempty"`
}

func runHealthCapacityCommand(args []string, out io.Writer) error {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	return runHealthCapacity(ctx, args, out)
}

func runHealthCapacity(ctx context.Context, args []string, out io.Writer) error {
	if ctx == nil || out == nil {
		return errors.New(healthCapacityUsage)
	}
	fs := flag.NewFlagSet("health-capacity", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	healthPath := fs.String("health", "", "health snapshot JSON file")
	telemetryPath := fs.String("telemetry", "", "canonical telemetry JSONL file")
	inputPath := fs.String("input", "", "telemetry options, findings, work and limits JSON file")
	validationPath := fs.String("nvsentinel", "", "optional NVSentinel v1.26 ValidationRequest JSON file")
	expected := fs.String("nvsentinel-sha256", "", "optional expected raw file SHA256; does not authenticate origin")
	maxBytes := fs.Int64("max-bytes", 8<<20, "maximum bytes per file (up to 64 MiB)")
	outputBytes := fs.Int64("output-bytes", 8<<20, "maximum output bytes (up to 64 MiB)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			_, err = fmt.Fprintln(out, healthCapacityUsage)
			return err
		}
		return err
	}
	if ctx == nil || out == nil || fs.NArg() != 0 || *healthPath == "" || *telemetryPath == "" || *inputPath == "" || *maxBytes <= 0 || *maxBytes > 64<<20 || *outputBytes <= 0 || *outputBytes > 64<<20 || *expected != "" && *validationPath == "" {
		return errors.New(healthCapacityUsage)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	configJSON, err := readHealthCapacityFile(ctx, *inputPath, *maxBytes)
	if err != nil {
		return err
	}
	var config healthCapacityCommandInput
	if err := decodeHealthCapacityJSON(configJSON, &config); err != nil {
		return fmt.Errorf("health capacity input: %w", err)
	}
	healthJSON, err := readHealthCapacityFile(ctx, *healthPath, *maxBytes)
	if err != nil {
		return err
	}
	// Validate duplicate/depth bounds before invoking the existing health reader.
	if err := checkHealthCapacityJSON(healthJSON); err != nil {
		return fmt.Errorf("health capacity health: %w", err)
	}
	telemetryJSONL, err := readHealthCapacityFile(ctx, *telemetryPath, *maxBytes)
	if err != nil {
		return err
	}
	input := cudadiag.HealthCapacityInput{HealthJSON: healthJSON, HealthLimits: config.HealthLimits, TelemetryJSONL: telemetryJSONL, Telemetry: config.Telemetry, HealthFindings: config.HealthFindings, Work: config.Work}
	output := healthCapacityCommandOutput{ConfigurationSHA256: healthCapacityDigest(configJSON)}
	if *validationPath != "" {
		raw, err := readHealthCapacityFile(ctx, *validationPath, *maxBytes)
		if err != nil {
			return err
		}
		imported, err := readNVSentinel(raw, *expected)
		if err != nil {
			return err
		}
		output.Validation = &imported
		// References identify only these exact bytes. They confer no authority or
		// device/provider association, and never change the composer's admission.
		id := "nvsentinel:sha256:" + imported.SHA256
		input.ExternalRefs = append(input.ExternalRefs, cudadiag.HealthCapacityArtifactRef{Kind: "request", ID: id + ":request", SHA256: imported.SHA256, State: "producer_reported_requested"})
		if imported.Request.Status != nil {
			input.ExternalRefs = append(input.ExternalRefs, cudadiag.HealthCapacityArtifactRef{Kind: "result", ID: id + ":status", SHA256: imported.SHA256, State: "producer_reported_" + imported.ProducerPhase})
		}
	}
	incident, err := cudadiag.ComposeHealthCapacity(ctx, input, config.Limits)
	if err != nil {
		return err
	}
	output.Incident = incident
	data, err := json.Marshal(output)
	if err != nil {
		return err
	}
	if int64(len(data))+1 > *outputBytes {
		return errors.New("health capacity output byte limit exceeded")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	data = append(data, '\n')
	n, err := out.Write(data)
	if err == nil && n != len(data) {
		return io.ErrShortWrite
	}
	return err
}

func healthCapacityDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func readHealthCapacityFile(ctx context.Context, name string, limit int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Nonblocking open also refuses a FIFO substituted between path lookup and
	// open, before any read can wait for a writer. Only regular files are read.
	f, err := os.OpenFile(name, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, fmt.Errorf("health capacity file %q: %w", name, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return nil, errors.New("health capacity input is nonregular or exceeds byte limit")
	}
	raw, err := io.ReadAll(io.LimitReader(healthCapacityFileReader{ctx: ctx, r: f}, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, errors.New("health capacity input byte limit exceeded")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return raw, nil
}

type healthCapacityFileReader struct {
	ctx context.Context
	r   io.Reader
}

func (r healthCapacityFileReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

func decodeHealthCapacityJSON(raw []byte, value any) error {
	if err := checkHealthCapacityJSON(raw); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		return err
	}
	return nil
}

// The token pass bounds allocation before typed decoding and rejects ambiguous
// duplicate keys, including encoding/json's Unicode case-folded field aliases.
func checkHealthCapacityJSON(raw []byte) error {
	if !utf8.Valid(raw) {
		return errors.New("health capacity invalid UTF-8 JSON")
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	tokens := 0
	var read func(int) error
	read = func(depth int) error {
		if depth > 32 {
			return errors.New("health capacity JSON nesting limit exceeded")
		}
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		tokens++
		if tokens > 200000 {
			return errors.New("health capacity JSON token limit exceeded")
		}
		switch value := token.(type) {
		case string:
			if len(value) > 32768 {
				return errors.New("health capacity JSON string limit exceeded")
			}
		case json.Delim:
			switch value {
			case '{':
				keys := make(map[string]bool)
				for decoder.More() {
					key, err := decoder.Token()
					if err != nil {
						return err
					}
					name, ok := key.(string)
					if !ok || len(name) > 32768 {
						return errors.New("health capacity JSON key limit exceeded")
					}
					folded := strings.Map(func(r rune) rune {
						minRune := r
						for next := unicode.SimpleFold(r); next != r; next = unicode.SimpleFold(next) {
							if next < minRune {
								minRune = next
							}
						}
						return minRune
					}, name)
					if keys[folded] {
						return errors.New("health capacity duplicate JSON key")
					}
					keys[folded] = true
					tokens++
					if tokens > 200000 {
						return errors.New("health capacity JSON token limit exceeded")
					}
					if err := read(depth + 1); err != nil {
						return err
					}
				}
				end, err := decoder.Token()
				if err != nil {
					return err
				}
				if end != json.Delim('}') {
					return errors.New("health capacity malformed JSON object")
				}
			case '[':
				count := 0
				for decoder.More() {
					count++
					if count > 4096 {
						return errors.New("health capacity JSON record limit exceeded")
					}
					if err := read(depth + 1); err != nil {
						return err
					}
				}
				end, err := decoder.Token()
				if err != nil {
					return err
				}
				if end != json.Delim(']') {
					return errors.New("health capacity malformed JSON array")
				}
			default:
				return errors.New("health capacity malformed JSON delimiter")
			}
		}
		return nil
	}
	if err := read(0); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return errors.New("health capacity trailing JSON data")
	}
	return nil
}
