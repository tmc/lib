package cudadiagdemo

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ExportArtifacts writes the full demonstration artifact chain to the designated directory.
func ExportArtifacts(dir string, result *DemonstrationResult, reqs []Request, logs []GroundTruthLog, captureBytes []byte) (map[string]string, error) {
	if dir == "" {
		return nil, nil
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("cudadiagdemo: failed to create artifact dir: %w", err)
	}

	paths := make(map[string]string)

	// 1. inputs.json
	inputsPath := filepath.Join(dir, "inputs.json")
	inputsData, err := json.MarshalIndent(reqs, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal inputs: %w", err)
	}
	if err := os.WriteFile(inputsPath, inputsData, 0644); err != nil {
		return nil, fmt.Errorf("write inputs: %w", err)
	}
	paths["inputs"] = inputsPath

	// 2. ground_truth.json
	gtPath := filepath.Join(dir, "ground_truth.json")
	gtData, err := json.MarshalIndent(logs, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal ground_truth: %w", err)
	}
	if err := os.WriteFile(gtPath, gtData, 0644); err != nil {
		return nil, fmt.Errorf("write ground_truth: %w", err)
	}
	paths["ground_truth"] = gtPath

	// 3. capture.tar
	if len(captureBytes) > 0 {
		capPath := filepath.Join(dir, "capture.tar")
		if err := os.WriteFile(capPath, captureBytes, 0644); err != nil {
			return nil, fmt.Errorf("write capture.tar: %w", err)
		}
		paths["capture"] = capPath
	}

	// 4. diagnosis.json
	diagPath := filepath.Join(dir, "diagnosis.json")
	diagData, err := json.MarshalIndent(result.HostDelayDiagnosis, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal diagnosis: %w", err)
	}
	if err := os.WriteFile(diagPath, diagData, 0644); err != nil {
		return nil, fmt.Errorf("write diagnosis: %w", err)
	}
	paths["diagnosis"] = diagPath

	// 5. trial_plan.json
	if result.TrialPlan != nil {
		planPath := filepath.Join(dir, "trial_plan.json")
		planData, err := json.MarshalIndent(result.TrialPlan, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("marshal trial_plan: %w", err)
		}
		if err := os.WriteFile(planPath, planData, 0644); err != nil {
			return nil, fmt.Errorf("write trial_plan: %w", err)
		}
		paths["trial_plan"] = planPath
	}

	// 6. execution_receipt.json
	if result.TrialExecution != nil {
		execPath := filepath.Join(dir, "execution_receipt.json")
		execData, err := json.MarshalIndent(result.TrialExecution, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("marshal execution_receipt: %w", err)
		}
		if err := os.WriteFile(execPath, execData, 0644); err != nil {
			return nil, fmt.Errorf("write execution_receipt: %w", err)
		}
		paths["execution_receipt"] = execPath
	}

	// 7. comparison_receipt.json
	if result.TrialComparison != nil {
		compPath := filepath.Join(dir, "comparison_receipt.json")
		compData, err := json.MarshalIndent(result.TrialComparison, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("marshal comparison_receipt: %w", err)
		}
		if err := os.WriteFile(compPath, compData, 0644); err != nil {
			return nil, fmt.Errorf("write comparison_receipt: %w", err)
		}
		paths["comparison_receipt"] = compPath
	}

	// 8. summary.json
	summaryPath := filepath.Join(dir, "summary.json")
	summaryData, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal summary: %w", err)
	}
	if err := os.WriteFile(summaryPath, summaryData, 0644); err != nil {
		return nil, fmt.Errorf("write summary: %w", err)
	}
	paths["summary"] = summaryPath

	return paths, nil
}
