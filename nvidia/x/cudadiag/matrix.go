package cudadiag

import (
	"fmt"
	"sort"
	"strings"
)

// ConfusionMatrix computes classification metrics across diagnostic trials,
// separating valid non-withheld mechanism predictions from withheld trials.
type ConfusionMatrix struct {
	TotalTrials     int                                     `json:"total_trials"`
	WithheldTrials  int                                     `json:"withheld_trials"`
	WithheldReasons map[string]int                          `json:"withheld_reasons"`
	Counts          map[MechanismType]map[MechanismType]int `json:"counts"`
}

// NewConfusionMatrix creates an initialized ConfusionMatrix.
func NewConfusionMatrix() *ConfusionMatrix {
	return &ConfusionMatrix{
		WithheldReasons: make(map[string]int),
		Counts:          make(map[MechanismType]map[MechanismType]int),
	}
}

// Record records one trial outcome. If predicted is MechanismWithheld, it increments
// the withheld count and records the specific withheld reason. Otherwise, it records
// the actual vs predicted mechanism classification.
func (cm *ConfusionMatrix) Record(actual, predicted MechanismType, withheldReason string) {
	cm.TotalTrials++
	if predicted == MechanismWithheld {
		cm.WithheldTrials++
		reason := withheldReason
		if reason == "" {
			reason = "unspecified"
		}
		cm.WithheldReasons[reason]++
		return
	}

	if cm.Counts[actual] == nil {
		cm.Counts[actual] = make(map[MechanismType]int)
	}
	cm.Counts[actual][predicted]++
}

// WithheldFraction returns the fraction of total trials that were withheld.
func (cm *ConfusionMatrix) WithheldFraction() float64 {
	if cm.TotalTrials == 0 {
		return 0.0
	}
	return float64(cm.WithheldTrials) / float64(cm.TotalTrials)
}

// Accuracy computes the classification accuracy across all non-withheld trials.
func (cm *ConfusionMatrix) Accuracy() float64 {
	validTrials := cm.TotalTrials - cm.WithheldTrials
	if validTrials <= 0 {
		return 0.0
	}

	correct := 0
	for actual, preds := range cm.Counts {
		correct += preds[actual]
	}
	return float64(correct) / float64(validTrials)
}

// FormatReport generates a human-readable summary of the confusion matrix,
// withheld breakdown, and overall accuracy.
func (cm *ConfusionMatrix) FormatReport() string {
	var sb strings.Builder
	sb.WriteString("=== Diagnostic Confusion Matrix ===\n")
	sb.WriteString(fmt.Sprintf("Total Trials: %d | Withheld Trials: %d (%.2f%%) | Evaluated Trials: %d\n",
		cm.TotalTrials, cm.WithheldTrials, cm.WithheldFraction()*100, cm.TotalTrials-cm.WithheldTrials))
	sb.WriteString(fmt.Sprintf("Evaluated Accuracy: %.2f%%\n\n", cm.Accuracy()*100))

	if cm.WithheldTrials > 0 {
		sb.WriteString("Withheld Reasons Breakdown:\n")
		var reasons []string
		for r := range cm.WithheldReasons {
			reasons = append(reasons, r)
		}
		sort.Strings(reasons)
		for _, r := range reasons {
			sb.WriteString(fmt.Sprintf("  - %s: %d\n", r, cm.WithheldReasons[r]))
		}
		sb.WriteString("\n")
	}

	mechanisms := []MechanismType{
		MechanismNoFault,
		MechanismHostSubmissionDelay,
		MechanismDeviceBottleneck,
		MechanismAmbiguousSimultaneous,
	}

	sb.WriteString("Classification Matrix (Rows: Actual, Columns: Predicted):\n")
	sb.WriteString(fmt.Sprintf("%-32s", "Actual \\ Predicted"))
	for _, m := range mechanisms {
		sb.WriteString(fmt.Sprintf("%-32s", m))
	}
	sb.WriteString("\n")

	for _, actual := range mechanisms {
		sb.WriteString(fmt.Sprintf("%-32s", actual))
		for _, pred := range mechanisms {
			count := 0
			if cm.Counts[actual] != nil {
				count = cm.Counts[actual][pred]
			}
			sb.WriteString(fmt.Sprintf("%-32d", count))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}
