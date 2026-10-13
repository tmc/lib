package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nsightcompute"
)

const importNCUUsage = "usage: cudaprof import-ncu [-json] [-limit N] [-max-bytes N] [-capture capture.pprof -join-receipt receipt.json] envelope.json"

// The receipt is a project producer record, not NVIDIA report metadata and not
// a driver attestation. An alias prevents permissive Artifact.UnmarshalJSON
// from bypassing the strict decoder's unknown-field checks.
type ncuArtifact cudakernel.Artifact
type ncuJoinReceipt struct {
	SchemaVersion int                        `json:"schema_version"`
	Artifact      *ncuArtifact               `json:"artifact"`
	Workload      cudacache.WorkloadIdentity `json:"workload"`
	CompilerKey   string                     `json:"compiler_key"`
}

type ncuSummary struct {
	SchemaVersion int                 `json:"schema_version"`
	ToolVersion   string              `json:"tool_version"`
	Format        string              `json:"format"`
	SourcePath    string              `json:"source_path"`
	ReportDigest  string              `json:"report_digest"`
	CSVDigest     string              `json:"csv_digest"`
	TotalRows     int                 `json:"total_rows"`
	Measured      int                 `json:"external_measured"`
	Unread        int                 `json:"unread"`
	Rules         int                 `json:"unsupported_rules"`
	Truncated     bool                `json:"truncated"`
	Rows          []nsightcompute.Row `json:"rows"`
}

func runImportNCUCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("import-ncu", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonOutput := fs.Bool("json", false, "write a bounded JSON summary")
	limit := fs.Int("limit", 16, "maximum rows in summary (1..1024)")
	maxBytes := fs.Int("max-bytes", nsightcompute.MaxBytes, "maximum bytes per input and output")
	capturePath := fs.String("capture", "", "retained capture for an explicit identity join")
	receiptPath := fs.String("join-receipt", "", "actual artifact/workload/compiler producer receipt")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			_, err = fmt.Fprintln(stdout, importNCUUsage)
			return err
		}
		return fmt.Errorf("import-ncu: %w", err)
	}
	if fs.NArg() != 1 {
		return errors.New(importNCUUsage)
	}
	if *limit < 1 || *limit > 1024 || *maxBytes <= 0 || *maxBytes > nsightcompute.MaxBytes {
		return fmt.Errorf("import-ncu: invalid summary or byte budget")
	}
	if (*capturePath == "") != (*receiptPath == "") {
		return fmt.Errorf("import-ncu: capture and join-receipt are required together")
	}
	data, err := readNCUFile(fs.Arg(0), *maxBytes)
	if err != nil {
		return err
	}
	envelope, err := nsightcompute.Read(bytes.NewReader(data), *maxBytes)
	if err != nil {
		return err
	}
	var candidates []nsightcompute.Candidate
	if *capturePath != "" {
		b, err := readNCUFile(*receiptPath, *maxBytes)
		if err != nil {
			return err
		}
		var receipt ncuJoinReceipt
		if err := decodeNCUReceipt(b, &receipt); err != nil {
			return fmt.Errorf("import-ncu: join receipt: %w", err)
		}
		if receipt.SchemaVersion != 1 || receipt.Artifact == nil {
			return fmt.Errorf("import-ncu: incomplete or unknown join receipt")
		}
		a := cudakernel.Artifact(*receipt.Artifact)
		if err := a.Validate(); err != nil {
			return fmt.Errorf("import-ncu: join artifact: %w", err)
		}
		b, err = readNCUFile(*capturePath, *maxBytes)
		if err != nil {
			return err
		}
		if len(b) > 1 && b[0] == 0x1f && b[1] == 0x8b {
			z, err := gzip.NewReader(bytes.NewReader(b))
			if err != nil {
				return fmt.Errorf("import-ncu: capture: %w", err)
			}
			expanded, readErr := io.ReadAll(io.LimitReader(z, int64(*maxBytes)+1))
			closeErr := z.Close()
			if readErr != nil {
				return fmt.Errorf("import-ncu: capture: %w", readErr)
			}
			if closeErr != nil {
				return fmt.Errorf("import-ncu: capture: %w", closeErr)
			}
			if len(expanded) > *maxBytes {
				return fmt.Errorf("import-ncu: expanded capture byte budget exceeded")
			}
			b = expanded
		}
		c, err := cudacapture.ReadLimits(bytes.NewReader(b), cudacapture.Limits{InputBytes: int64(*maxBytes), FileBytes: int64(*maxBytes), TotalBytes: int64(*maxBytes), Files: 64})
		if err != nil {
			return fmt.Errorf("import-ncu: capture: %w", err)
		}
		candidates = []nsightcompute.Candidate{{Capture: c, Artifact: &a, Workload: receipt.Workload, CompilerKey: receipt.CompilerKey}}
	}
	report, err := nsightcompute.Import(envelope, candidates)
	if err != nil {
		return err
	}
	summary := ncuSummary{SchemaVersion: 1, ToolVersion: envelope.ToolVersion, Format: envelope.Format, SourcePath: envelope.SourcePath, ReportDigest: envelope.ReportDigest, CSVDigest: envelope.CSVDigest, TotalRows: len(report.Rows), Truncated: len(report.Rows) > *limit}
	for _, row := range report.Rows {
		if row.Metric == nil {
			summary.Rules++
		} else if row.Metric.Value.Record().Number == nil {
			summary.Unread++
		} else {
			summary.Measured++
		}
	}
	summary.Rows = report.Rows[:min(*limit, len(report.Rows))]
	var output bytes.Buffer
	if *jsonOutput {
		if err := json.NewEncoder(&output).Encode(summary); err != nil {
			return err
		}
	} else {
		fmt.Fprintf(&output, "Nsight Compute %s; %d rows; external measured %d; unread %d; unsupported rules %d\n", summary.ToolVersion, summary.TotalRows, summary.Measured, summary.Unread, summary.Rules)
		fmt.Fprintf(&output, "report sha256:%s; CSV sha256:%s\n", summary.ReportDigest, summary.CSVDigest)
		for _, row := range summary.Rows {
			if row.Metric == nil {
				fmt.Fprintf(&output, "%q %q: unsupported rule\n", ncuShort(row.Entry), ncuShort(row.Section))
				continue
			}
			m := row.Metric
			r := m.Value.Record()
			class := string(r.Class)
			if class == "" {
				class = "unread"
			}
			fmt.Fprintf(&output, "%q %q %q=%q %q [%s] %q\n", ncuShort(row.Entry), ncuShort(row.Section), ncuShort(m.Name), ncuShort(m.RawValue), ncuShort(m.Unit), class, ncuShort(r.Reason))
		}
		if summary.Truncated {
			fmt.Fprintf(&output, "summary truncated to %d rows; raw export retained by importer\n", len(summary.Rows))
		}
	}
	if output.Len() > *maxBytes {
		return fmt.Errorf("import-ncu: output byte budget exceeded")
	}
	n, err := stdout.Write(output.Bytes())
	if err == nil && n != output.Len() {
		err = io.ErrShortWrite
	}
	return err
}

func ncuShort(s string) string {
	if len(s) > 128 {
		return s[:128] + "..."
	}
	return s
}
func readNCUFile(path string, limit int) ([]byte, error) {
	before, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("import-ncu: %w", err)
	}
	if !before.Mode().IsRegular() || before.Size() > int64(limit) {
		return nil, fmt.Errorf("import-ncu: input is not a bounded regular file")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("import-ncu: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !os.SameFile(before, info) || !info.Mode().IsRegular() || info.Size() > int64(limit) {
		return nil, fmt.Errorf("import-ncu: input is not a bounded regular file")
	}
	b, err := io.ReadAll(io.LimitReader(f, int64(limit)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > limit {
		return nil, fmt.Errorf("import-ncu: input byte budget exceeded")
	}
	return b, nil
}
func decodeNCUReceipt(b []byte, out *ncuJoinReceipt) error {
	d := json.NewDecoder(bytes.NewReader(b))
	if err := ncuJSONKeys(d, 0); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	var trailing any
	if err := d.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("trailing JSON")
	}
	return nil
}
func ncuJSONKeys(d *json.Decoder, depth int) error {
	if depth > 64 {
		return fmt.Errorf("JSON nesting budget exceeded")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			t, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := t.(string)
			if !ok || seen[key] || !ncuASCIIJSONKey(key) {
				return fmt.Errorf("duplicate or aliased JSON key")
			}
			seen[key] = true
			if err := ncuJSONKeys(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := ncuJSONKeys(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("invalid JSON delimiter")
	}
	_, err = d.Token()
	return err
}

func ncuASCIIJSONKey(key string) bool {
	if key == "" {
		return false
	}
	for i := 0; i < len(key); i++ {
		c := key[i]
		if c != '_' && !(c >= 'a' && c <= 'z') && !(c >= '0' && c <= '9') {
			return false
		}
	}
	return true
}
