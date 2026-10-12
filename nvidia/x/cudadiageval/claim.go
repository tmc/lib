package cudadiageval

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrIncompleteClaim indicates required claim fields are missing.
	ErrIncompleteClaim = errors.New("cudadiageval: claim missing required bounded fields")
)

// FormulateClaim checks the report's local no-fault receipt but cannot derive
// a comparative claim without independently verified paired evidence. Imported
// report assertions and named baselines do not supply that evidence.
func FormulateClaim(report *EvaluationReport) (*Claim, error) {
	if report == nil {
		return nil, errors.New("cudadiageval: nil evaluation report")
	}
	if report.Synthetic || !report.Passed || report.Protocol.Phase != PhaseQualification || !report.Protocol.Frozen {
		return nil, errors.New("cudadiageval: qualification evidence incomplete")
	}
	for _, gate := range report.Gates {
		if gate.Disposition != "passed" {
			return nil, errors.New("cudadiageval: qualification gate not passed")
		}
	}
	if err := validateNoFaultClaimReceipt(report); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("%w: independently verified paired candidate and comparator evidence required", ErrIncompleteClaim)
}

// This is the accepted no-fault guard alone. Passing it neither authenticates
// the report nor supplies candidate/comparator measurements.
func validateNoFaultClaimReceipt(report *EvaluationReport) error {
	if report == nil {
		return errors.New("cudadiageval: nil evaluation report")
	}
	d := report.Diagnostic
	noFaultDisp, noFaultReason := noFaultGate(d.NoFaultPassed, d.NoFaultWithheld, d.NoFaultMissing, d.NoFaultTotal, report.Protocol.Budgets.MinNoFaultRejectionPct)
	if noFaultDisp != "passed" {
		return fmt.Errorf("cudadiageval: no-fault qualification incomplete: %s", noFaultReason)
	}
	return nil
}

// Validate verifies that the claim conforms to the bounded five-part format
// required by sota-evaluation.md.
func (c *Claim) Validate() error {
	if c.Task == "" {
		return fmt.Errorf("%w: missing task", ErrIncompleteClaim)
	}
	if c.SpecificAdvantage == "" {
		return fmt.Errorf("%w: missing specific advantage", ErrIncompleteClaim)
	}
	if len(c.ComparedWith) == 0 {
		return fmt.Errorf("%w: missing compared_with alternatives", ErrIncompleteClaim)
	}
	if c.QualifiedOn == "" {
		return fmt.Errorf("%w: missing qualified_on hardware specification", ErrIncompleteClaim)
	}
	if c.Evidence == "" {
		return fmt.Errorf("%w: missing evidence receipts", ErrIncompleteClaim)
	}
	if len(c.Limits) == 0 {
		return fmt.Errorf("%w: missing limits and boundaries", ErrIncompleteClaim)
	}
	return nil
}

// Format renders the claim in the exact 5-part format specified in sota-evaluation.md.
func (c *Claim) Format() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Claim: %s for %s.\n", c.SpecificAdvantage, c.Task))

	sb.WriteString("Compared with:\n")
	for _, comp := range c.ComparedWith {
		sb.WriteString(fmt.Sprintf("  - %s\n", comp))
	}

	sb.WriteString(fmt.Sprintf("Qualified on: %s.\n", c.QualifiedOn))
	sb.WriteString(fmt.Sprintf("Evidence: %s.\n", c.Evidence))

	sb.WriteString("Limits:\n")
	for _, lim := range c.Limits {
		sb.WriteString(fmt.Sprintf("  - %s\n", lim))
	}

	return sb.String()
}

// JSON returns indented JSON representation of the Claim.
func (c *Claim) JSON() ([]byte, error) {
	return json.MarshalIndent(c, "", "  ")
}
