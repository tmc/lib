package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"

	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func traceReadBounded(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("trace evidence: %s exceeds %d bytes", path, limit)
	}
	return data, nil
}

func traceSourceHashes(s *traceSession) ([]string, error) {
	if s.Kind == "capture" {
		return []string{s.SHA256}, nil
	}
	if s.Kind != "comparison_archive" || s.Members[".cudaprof-bindings.json"] == nil {
		return nil, errors.New("input has no bound capture identities")
	}
	var a alignmentReceipt
	if err := json.Unmarshal(s.Members[".alignment.json"], &a); err != nil {
		return nil, err
	}
	return []string{a.Arms[0].SHA256, a.Arms[1].SHA256}, nil
}

// Shape qualification is recomputed from retained, hash-matched captures.
// A receipt's asserted pass bit alone never establishes evidence.
func checkTraceShape(s *traceSession, path string, limits traceInputLimits) (traceEvidence, error) {
	_, _, e, err := validatedTraceShape(s, path, limits)
	return e, err
}

// Return the exact validated receipt and decoded source snapshots so summaries
// never reopen a receipt after validation or trust its asserted gate fields.
func validatedTraceShape(s *traceSession, path string, limits traceInputLimits) (comparisonShapeReceipt, [2]*traceSession, traceEvidence, error) {
	var receipt comparisonShapeReceipt
	var sources [2]*traceSession
	unknown := traceEvidence{"unavailable", "shape comparison requires bound arms and original captures", "shape receipt"}
	hashes, err := traceSourceHashes(s)
	if err != nil || len(hashes) != 2 {
		return receipt, sources, unknown, nil
	}
	data, err := traceReadBounded(path, limits.InputBytes)
	if err != nil {
		return receipt, sources, unknown, err
	}
	if err := json.Unmarshal(data, &receipt); err != nil {
		return receipt, sources, unknown, fmt.Errorf("trace shape receipt: %w", err)
	}
	if receipt.Version != 1 || (receipt.Domain != "driver" && receipt.Domain != "runtime") {
		return receipt, sources, unknown, errors.New("trace shape receipt: unsupported schema or domain")
	}
	var records [2]timeline.Records
	for i, a := range receipt.Arms {
		if a.SHA256 != hashes[i] {
			return receipt, sources, unknown, errors.New("trace shape receipt: source identity differs from input")
		}
		source, err := loadTraceSessionLimits(a.Path, receipt.Domain, limits)
		if err != nil {
			return receipt, sources, unknown, err
		}
		if source.Kind != "capture" || source.SHA256 != hashes[i] {
			return receipt, sources, unknown, errors.New("trace shape receipt: original capture identity mismatch")
		}
		sources[i] = source
		if source.CollectorError != nil {
			return receipt, sources, traceEvidence{"failed", source.CollectorError.Error(), "bound original capture"}, nil
		}
		records[i] = source.Records
	}
	actual := shape.CompareScoped(records[0], records[1], receipt.Domain, receipt.Shape.RequestPairs)
	if !reflect.DeepEqual(actual, receipt.Shape) {
		return receipt, sources, unknown, errors.New("trace shape receipt: recomputed shape differs from supplied receipt")
	}
	receipt.Shape = actual
	state, reason := "passed", "shape recomputed from complete hash-bound original captures"
	if !actual.Pass {
		state, reason = "failed", fmt.Sprint(actual.Failures)
	}
	return receipt, sources, traceEvidence{state, reason, "bound shape receipt and original captures"}, nil
}

// Correctness accepts the supported campaign receipt only when retained
// ordinary and profiled stdout artifacts independently reproduce equality.
func checkTraceCorrectness(s *traceSession, path string, limits traceInputLimits) (traceEvidence, error) {
	_, e, err := validatedTraceCorrectness(s, path, limits)
	return e, err
}
func validatedTraceCorrectness(s *traceSession, path string, limits traceInputLimits) (campaignReport, traceEvidence, error) {
	var campaign campaignReport
	unknown := traceEvidence{"unavailable", "no bound independently retained output comparison", "campaign receipt"}
	hashes, err := traceSourceHashes(s)
	if err != nil {
		return campaign, unknown, nil
	}
	data, err := traceReadBounded(path, limits.InputBytes)
	if err != nil {
		return campaign, unknown, err
	}
	if err := json.Unmarshal(data, &campaign); err != nil {
		return campaign, unknown, fmt.Errorf("trace correctness receipt: %w", err)
	}
	if campaign.Version != 1 || len(campaign.ExpectedSHA256) != 64 || len(campaign.Blocks) == 0 {
		return campaign, unknown, errors.New("trace correctness receipt: unsupported or empty campaign")
	}
	matched := make(map[string]bool)
	for _, block := range campaign.Blocks {
		if len(block.Excluded) != 0 {
			continue
		}
		ordinary := map[string]bool{}
		profiled := map[string]bool{}
		for _, run := range block.Runs {
			if run.Failure != "" {
				return campaign, traceEvidence{"failed", run.Failure, "campaign run"}, nil
			}
			output, err := traceReadBounded(run.Stdout, limits.InputBytes)
			if err != nil {
				return campaign, unknown, err
			}
			digest := sha256.Sum256(output)
			hash := hex.EncodeToString(digest[:])
			if hash != campaign.ExpectedSHA256 || hash != run.OutputSHA256 {
				return campaign, traceEvidence{"failed", "retained stdout differs from campaign output equality", "stdout artifact hash"}, nil
			}
			if !run.Profiled {
				ordinary[run.Arm] = true
				continue
			}
			profiled[run.Arm] = true
			capture, err := traceReadBounded(run.Capture, limits.InputBytes)
			if err != nil {
				return campaign, unknown, err
			}
			digest = sha256.Sum256(capture)
			hash = hex.EncodeToString(digest[:])
			for _, wanted := range hashes {
				if hash == wanted {
					matched[wanted] = true
				}
			}
		}
		for _, arm := range qualifyArms {
			if !ordinary[arm] || !profiled[arm] {
				return campaign, unknown, errors.New("trace correctness receipt: incomplete per-arm output controls")
			}
		}
	}
	for _, hash := range hashes {
		if !matched[hash] {
			return campaign, unknown, nil
		}
	}
	return campaign, traceEvidence{"passed", "retained stdout reproduces equality across both arms and profiled/ordinary controls; input captures match campaign runs", "campaign and stdout/capture artifact hashes"}, nil
}
