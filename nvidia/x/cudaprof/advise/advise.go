package advise

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"reflect"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

// WithEvidence returns a copy carrying the same classified profile share as
// item. It validates the supplied value without changing its quantity, source,
// denominator, or class. It cannot establish a missing population or replace
// withheld evidence; producers use WithShareEvidence to classify a share.
func (item Item) WithEvidence(value cudaevidence.Value) (Item, error) {
	validated, err := cudaevidence.New(value.Record())
	if err != nil {
		return Item{}, fmt.Errorf("advise evidence: %w", err)
	}
	next := item
	next.ClassifiedShare = &validated
	if err := next.ValidateEvidence(); err != nil {
		return Item{}, err
	}
	if item.ClassifiedShare == nil || !reflect.DeepEqual(item.ClassifiedShare.Record(), validated.Record()) {
		return Item{}, fmt.Errorf("advise evidence: supplied value changes classified profile share")
	}
	return next, nil
}

// CheckEvidenceClaim verifies that item carries valid classified evidence
// matching class and id without promoting Derived, Inferred, Modeled, or Unread
// evidence to Measured.
func (item Item) CheckEvidenceClaim(class cudaevidence.Class, id cudaevidence.Identity) error {
	if err := item.ValidateEvidence(); err != nil {
		return err
	}
	if item.ClassifiedShare == nil {
		return fmt.Errorf("advise evidence claim: missing classified evidence")
	}
	return item.ClassifiedShare.CheckClaim(class, id)
}

// ValidateEvidence checks that every item and nested pattern in r preserves
// evidence class boundaries, source identity binding, and numeric withholding
// for unmeasured or unread findings.
func (r *Report) ValidateEvidence() error {
	if r == nil {
		return nil
	}
	for i := range r.Items {
		if err := r.Items[i].ValidateEvidence(); err != nil {
			return fmt.Errorf("item %d: %w", i, err)
		}
	}
	return nil
}

// ValidateEvidence checks that item and its nested patterns carry only
// admissible classified evidence.
func (item Item) ValidateEvidence() error {
	if item.ClassifiedShare != nil {
		r := item.ClassifiedShare.Record()
		if _, err := cudaevidence.New(r); err != nil {
			return fmt.Errorf("advise evidence: %w", err)
		}
		if r.Unit != "percent" {
			return fmt.Errorf("advise evidence: classified profile share requires percent unit")
		}
		if r.Class == cudaevidence.Unread {
			if r.Number != nil || r.Uncertainty != nil {
				return fmt.Errorf("advise evidence: unread value carries numeric data")
			}
		} else {
			if item.EvidenceLevel == EvidenceUnmeasured {
				return fmt.Errorf("advise evidence: unmeasured finding must withhold numeric value (got class %q)", r.Class)
			}
			if item.EvidenceLevel != EvidenceObserved && item.EvidenceLevel != EvidenceSuspected {
				return fmt.Errorf("advise evidence: unsupported evidence level %q for class %q", item.EvidenceLevel, r.Class)
			}
			if item.Location.Kernel == "" || item.Location.Kernel != r.Source.Identity.Entry || item.Location.Module != "" {
				return fmt.Errorf("advise evidence: kernel identity mismatch for %q vs entry %q", item.Location.Kernel, r.Source.Identity.Entry)
			}
			if r.Class == cudaevidence.Measured && item.EvidenceLevel != EvidenceObserved {
				return fmt.Errorf("advise evidence: %q finding cannot claim measured evidence", item.EvidenceLevel)
			}
			if r.Number == nil || math.IsNaN(*r.Number) || math.IsInf(*r.Number, 0) || *r.Number < 0 || *r.Number > 100 {
				return fmt.Errorf("advise evidence: invalid percentage")
			}
			if r.Class == cudaevidence.Measured {
				return fmt.Errorf("advise evidence: profile share ratio cannot be measured")
			}
			if item.EvidenceLevel == EvidenceSuspected && r.Class == cudaevidence.Derived {
				return fmt.Errorf("advise evidence: suspected profile share cannot be derived")
			}
			if item.EvidenceLevel == EvidenceObserved && r.Class != cudaevidence.Derived || item.EvidenceLevel == EvidenceSuspected && r.Class != cudaevidence.Inferred {
				return fmt.Errorf("advise evidence: class %q does not describe this profile share", r.Class)
			}
		}
	}
	for i := range item.Patterns {
		if err := item.Patterns[i].ValidateEvidence(); err != nil {
			return fmt.Errorf("pattern %d: %w", i, err)
		}
	}
	return nil
}

type itemJSON Item

// MarshalJSON validates classified evidence boundaries before encoding item.
func (item Item) MarshalJSON() ([]byte, error) {
	if err := item.ValidateEvidence(); err != nil {
		return nil, err
	}
	return json.Marshal(itemJSON(item))
}

// UnmarshalJSON decodes one advice item and refuses fabricated measured
// provenance, identity mismatches, or numeric values on unmeasured findings.
func (item *Item) UnmarshalJSON(data []byte) error {
	if item == nil {
		return fmt.Errorf("decode advise item: nil destination")
	}
	d := json.NewDecoder(bytes.NewReader(data))
	if err := profileShareJSONKeys(d, 0); err != nil {
		return fmt.Errorf("decode advise item: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("decode advise item: trailing JSON")
	}
	var raw itemJSON
	d = json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&raw); err != nil {
		return fmt.Errorf("decode advise item: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("decode advise item: trailing JSON")
	}
	next := Item(raw)
	if err := next.ValidateEvidence(); err != nil {
		return fmt.Errorf("decode advise item: %w", err)
	}
	*item = next
	return nil
}
