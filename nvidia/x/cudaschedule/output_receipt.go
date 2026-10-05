package cudaschedule

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
)

const outputReceiptPrefix = "CUDASCHEDULE_OUTPUT:"
const outputReceiptLimit = 1 << 20

type outputReceipt struct {
	PID    int
	Output *cudaoutputcoverage.RecordedOutput
}

// WriteOutputReceipt writes completed output evidence to a recorded worker's
// stdout. The controller binds the receipt to its owned worker PID and loaded
// PTX receipt. This carries producer claims, not independent device attestation.
// Evidence must be immutable during the call. The wire format accepts at most
// 65536 values across all readbacks and references, 16 KiB of string metadata,
// and a one MiB JSON body.
func WriteOutputReceipt(w io.Writer, out *cudaoutputcoverage.RecordedOutput) error {
	if w == nil {
		return fmt.Errorf("output receipt requires a writer")
	}
	if err := boundedOutputReceipt(out); err != nil {
		return err
	}
	copy, err := snapshotOutput(out)
	if err != nil {
		return err
	}
	b, err := json.Marshal(outputReceipt{PID: os.Getpid(), Output: copy})
	if err != nil {
		return fmt.Errorf("encode output receipt: %w", err)
	}
	if len(b) > outputReceiptLimit {
		return fmt.Errorf("output receipt exceeds byte limit")
	}
	wire := outputReceiptPrefix + string(b) + "\n"
	n, err := io.WriteString(w, wire)
	if err == nil && n != len(wire) {
		err = io.ErrShortWrite
	}
	return err
}

func boundedOutputReceipt(out *cudaoutputcoverage.RecordedOutput) error {
	if out == nil || !validPTXDigest(out.PTXDigest) {
		return fmt.Errorf("output receipt requires evidence and PTX digest")
	}
	remaining := 16 << 10
	for _, s := range []string{
		out.PTXDigest, out.TwoFill.Contract.Buffer, string(out.TwoFill.Contract.Kind),
		out.TwoFill.SourceDigest, out.TwoFill.InputDigest, out.TwoFill.Target, out.TwoFill.MathMode,
		out.Policy.Target, out.Policy.DType, out.Policy.MathMode, out.Policy.SourceDigest,
		out.Policy.InputDigest, out.Policy.NaN, out.Policy.Infinity, out.Policy.Zero,
	} {
		if len(s) > remaining {
			return fmt.Errorf("output receipt exceeds metadata limit")
		}
		remaining -= len(s)
	}
	if len(out.Ordinary) > 65536 || len(out.TwoFill.Reference) > 65536 ||
		len(out.TwoFill.ReadbackA) > 65536 || len(out.TwoFill.ReadbackB) > 65536 ||
		len(out.TwoFill.Contract.Domain) > 65536 {
		return fmt.Errorf("output receipt exceeds value limits")
	}
	n := len(out.Ordinary) + len(out.TwoFill.Reference) + len(out.TwoFill.ReadbackA) + len(out.TwoFill.ReadbackB)
	if n > 65536 {
		return fmt.Errorf("output receipt exceeds total value limit")
	}
	return nil
}

func takeOutputReceipt(stdout string, worker int, ptx string) (string, *cudaoutputcoverage.RecordedOutput, error) {
	var clean strings.Builder
	var out *cudaoutputcoverage.RecordedOutput
	for rest := stdout; rest != ""; {
		line, tail, found := strings.Cut(rest, "\n")
		rest = tail
		if strings.HasPrefix(line, outputReceiptPrefix) {
			body := strings.TrimPrefix(line, outputReceiptPrefix)
			if out != nil || !found || len(body) > outputReceiptLimit {
				return stdout, nil, fmt.Errorf("invalid or duplicate output receipt")
			}
			var receipt outputReceipt
			if err := json.Unmarshal([]byte(body), &receipt); err != nil {
				return stdout, nil, fmt.Errorf("decode output receipt: %w", err)
			}
			// The writer emits canonical JSON. Exact encoding also rejects unknown
			// or duplicate fields instead of silently accepting the last value.
			canonical, err := json.Marshal(receipt)
			if err != nil || !bytes.Equal(canonical, []byte(body)) {
				return stdout, nil, fmt.Errorf("noncanonical output receipt")
			}
			if worker <= 1 || receipt.PID != worker {
				return stdout, nil, fmt.Errorf("output receipt worker pid mismatch")
			}
			if err := boundedOutputReceipt(receipt.Output); err != nil {
				return stdout, nil, err
			}
			if receipt.Output.PTXDigest != ptx {
				return stdout, nil, fmt.Errorf("output receipt loaded PTX mismatch")
			}
			out = receipt.Output
			continue
		}
		clean.WriteString(line)
		if found {
			clean.WriteByte('\n')
		}
	}
	return clean.String(), out, nil
}
