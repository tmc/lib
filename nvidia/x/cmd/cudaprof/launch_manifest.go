package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

const (
	defaultLaunchManifestMaxBytes   = 64 << 10
	defaultLaunchManifestMaxRecords = 1024
)

// buildLaunchManifest joins already-retained host API and device activity
// records. It does not inspect callback pointers or infer a CUPTI parameter
// layout, so the result is explicitly metadata-only.
func buildLaunchManifest(result captureResult) json.RawMessage {
	m := newLaunchManifest(launchManifestOptions{Enabled: true})
	b, err := marshalBoundedLaunchManifest(m, manifestRecords(result))
	if err != nil {
		return launchManifestError(err)
	}
	return b
}

// launchManifestError reports why no manifest was produced. The reason is
// carried in the status record because a consumer that sees only a missing
// manifest cannot tell a bounded omission from a defect.
func launchManifestError(cause error) json.RawMessage {
	b, err := json.Marshal(struct {
		Schema       string `json:"schema"`
		Completeness string `json:"completeness"`
		Error        string `json:"error"`
	}{"cudaprof-launch-manifest/v1", "metadata-only", cause.Error()})
	if err != nil {
		return json.RawMessage(`{"schema":"cudaprof-launch-manifest/v1","completeness":"metadata-only","error":"marshal failed"}`)
	}
	return b
}

func manifestRecords(result captureResult) []launchManifestRecord {
	records := make([]launchManifestRecord, 0, len(result.APIRecords))
	for _, api := range result.APIRecords {
		r := launchManifestRecord{
			CallbackID:  api.CallbackID,
			Correlation: api.Correlation,
			Domain:      api.Domain,
			Function:    "UNREAD",
			Module:      "UNREAD",
			Parameters:  launchBytes{Status: "UNREAD"},
		}
		for _, activity := range result.Activities {
			if activity.Subtype != "kernel" || activity.Correlation != api.Correlation {
				continue
			}
			r.ActivityCorrelation = activity.Correlation
			r.Function = activity.Name
			r.Stream = activity.Stream
			r.Grid = activity.GridDim
			r.Block = activity.BlockDim
			r.SharedMemory = activity.DynamicSharedMemory
			break
		}
		// This metadata-only path cannot bind a compiler/workload identity.
		r.BlockXEvidence, _ = classifyLaunchBlockX(r, nil)
		records = append(records, r)
	}
	return records
}

// launchManifest is an opt-in, host-side description of launch callbacks and
// their matching activity records. It is provenance, not an operand dump.
// The zero value is disabled and produces no records.
type launchManifest struct {
	Schema         string                 `json:"schema"`
	CUPTIVersion   uint32                 `json:"cupti_version"`
	Completeness   string                 `json:"completeness"`
	MaxBytes       int                    `json:"max_bytes"`
	MaxRecords     int                    `json:"max_records"`
	Records        []launchManifestRecord `json:"records,omitempty"`
	OmittedRecords uint64                 `json:"omitted_records,omitempty"`
	Omission       string                 `json:"omission,omitempty"`
}

type launchManifestRecord struct {
	BlockXEvidence      cudaevidence.Value `json:"block_x_evidence"`
	CallbackID          uint32             `json:"callback_id"`
	ActivityCorrelation uint32             `json:"activity_correlation,omitempty"`
	Correlation         uint32             `json:"correlation"`
	Domain              string             `json:"domain"`
	Function            string             `json:"function"`
	Symbol              string             `json:"symbol,omitempty"`
	Module              string             `json:"module"`
	Stream              uint32             `json:"stream"`
	Grid                [3]uint32          `json:"grid"`
	Block               [3]uint32          `json:"block"`
	SharedMemory        uint32             `json:"shared_memory"`
	Parameters          launchBytes        `json:"parameters"`
}

type launchBytes struct {
	Status string `json:"status"`
	Hex    string `json:"hex,omitempty"`
}

type launchManifestOptions struct {
	Enabled      bool
	CUPTIVersion uint32
	MaxBytes     int
	MaxRecords   int
}

func newLaunchManifest(opts launchManifestOptions) *launchManifest {
	if !opts.Enabled {
		return nil
	}
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = defaultLaunchManifestMaxBytes
	}
	if opts.MaxRecords <= 0 {
		opts.MaxRecords = defaultLaunchManifestMaxRecords
	}
	return &launchManifest{
		Schema:       "cudaprof-launch-manifest/v1",
		CUPTIVersion: opts.CUPTIVersion,
		Completeness: "metadata-only",
		MaxBytes:     opts.MaxBytes,
		MaxRecords:   opts.MaxRecords,
	}
}

func (m *launchManifest) limits() {
	if m.MaxBytes <= 0 {
		m.MaxBytes = defaultLaunchManifestMaxBytes
	}
	if m.MaxRecords <= 0 {
		m.MaxRecords = defaultLaunchManifestMaxRecords
	}
}

// addLaunchManifestRecord adds a correlation record. Parameter bytes remain
// UNREAD unless a caller has supplied a version-qualified decoder; this
// package deliberately has no guessed CUPTI callback layouts.
func (m *launchManifest) addLaunchManifestRecord(r launchManifestRecord) {
	if m == nil {
		return
	}
	if r.Module == "" {
		r.Module = "UNREAD"
	}
	if r.Parameters.Status == "" {
		r.Parameters.Status = "UNREAD"
	}
	r.Parameters.Hex = ""
	m.Records = append(m.Records, r)
}

// qualifiedLaunchBytes is the only byte-emission choke point. A decoder must
// prove the exact CUPTI version and layout before calling it.
func qualifiedLaunchBytes(version, wantVersion uint32, decoded []byte, max int) (launchBytes, error) {
	if version == 0 || wantVersion == 0 || version != wantVersion {
		return launchBytes{Status: "UNREAD"}, nil
	}
	if max <= 0 || len(decoded) > max {
		return launchBytes{Status: "UNREAD"}, fmt.Errorf("launch parameters exceed bounded manifest size")
	}
	return launchBytes{Status: "DECODED", Hex: hex.EncodeToString(decoded)}, nil
}

func (m *launchManifest) marshalJSON() ([]byte, error) {
	if m == nil {
		return nil, nil
	}
	return json.Marshal(m)
}

func marshalBoundedLaunchManifest(m *launchManifest, records []launchManifestRecord) ([]byte, error) {
	if m == nil {
		return nil, nil
	}
	m.limits()
	for i, record := range records {
		if len(m.Records) >= m.MaxRecords {
			m.OmittedRecords = uint64(len(records) - i)
			m.Omission = "record_limit"
			break
		}
		m.Records = append(m.Records, record)
		b, err := m.marshalJSON()
		if err != nil {
			return nil, err
		}
		if len(b) > m.MaxBytes {
			m.Records = m.Records[:len(m.Records)-1]
			m.OmittedRecords = uint64(len(records) - i)
			m.Omission = "byte_limit"
			break
		}
	}
	// Omission and OmittedRecords are set after the size check that triggered
	// them, so they are not in the length that check measured. Recording an
	// omission can therefore push a manifest that just fit back over the
	// limit. Drop records until the marshaled form, those fields included,
	// fits.
	for {
		b, err := m.marshalJSON()
		if err != nil {
			return nil, err
		}
		if len(b) <= m.MaxBytes {
			return b, nil
		}
		if len(m.Records) == 0 {
			return nil, fmt.Errorf("bounded launch manifest envelope exceeds max_bytes=%d with no records", m.MaxBytes)
		}
		m.Records = m.Records[:len(m.Records)-1]
		m.OmittedRecords++
		m.Omission = "byte_limit"
	}
}
