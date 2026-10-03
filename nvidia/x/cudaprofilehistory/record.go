package cudaprofilehistory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Version identifies the record and snapshot format.
const Version = 1

// Verdict is a supplied observation, not an execution or performance estimate.
type Verdict string

const (
	// Unknown means this predicate has not been established.
	Unknown Verdict = "unknown"
	// Pass is a supplied successful predicate observation.
	Pass Verdict = "pass"
	// Fail is a supplied unsuccessful predicate observation.
	Fail Verdict = "fail"
)

// Correctness binds an exact output comparison to this record's input/image.
// Only digest-exact is supported. An absent result is Unknown, never Pass.
type Correctness struct {
	Verdict        Verdict `json:"verdict"`
	Oracle         string  `json:"oracle,omitempty"`
	Method         string  `json:"method,omitempty"`
	ExpectedDigest string  `json:"expected_digest,omitempty"`
	ActualDigest   string  `json:"actual_digest,omitempty"`
}

// Environment identifies the conditions and clock of a performance observation.
// Values are producer declarations; importing them does not verify hardware.
type Environment struct {
	Device string `json:"device"`
	Driver string `json:"driver"`
	CUDA   string `json:"cuda"`
	Clock  string `json:"clock"`
	Domain string `json:"domain"`
}

func (e Environment) complete() bool {
	return strings.TrimSpace(e.Device) != "" && strings.TrimSpace(e.Driver) != "" && strings.TrimSpace(e.CUDA) != "" && strings.TrimSpace(e.Clock) != "" && strings.TrimSpace(e.Domain) != ""
}

// Fact retains a typed value on the codegen, resource or performance axis.
// Qualified means the producer supplied native qualification. It does not
// result from importing a receipt or from an execution's Complete flag.
// For available timing facts, Samples counts retained timing observations,
// equal to Loss.Total-Loss.Dropped when loss coverage is known.
type Fact struct {
	Axis      string             `json:"axis"`
	Name      string             `json:"name"`
	Value     cudaevidence.Value `json:"value"`
	Qualified bool               `json:"qualified"`
	Samples   int                `json:"samples"`
}

// Record is constructor input and the versioned observation representation.
// Store.Append retains its own encoding, not mutable slices in this struct.
// Status is complete, unqualified, execution_failed or compile_failed.
// Compile failures have no Artifact or Identity and cannot carry numeric facts
// or correctness passes. Source/input/plan digests remain mandatory.
type Record struct {
	Version          int                        `json:"version"`
	Key              string                     `json:"key,omitempty"`
	SourceDigest     string                     `json:"source_digest"`
	InputDigest      string                     `json:"input_digest"`
	CompilerInputKey string                     `json:"compiler_input_key"`
	Compiler         cudacache.CompilerIdentity `json:"compiler"`
	CompilerKey      string                     `json:"compiler_key"`
	Workload         cudacache.WorkloadIdentity `json:"workload"`
	WorkloadKey      string                     `json:"workload_key"`
	NumericalMode    string                     `json:"numerical_mode"`
	Entry            string                     `json:"entry"`
	Target           string                     `json:"target"`
	PlanDigest       string                     `json:"plan_digest"`
	Receipt          []byte                     `json:"receipt"`
	ReceiptDigest    string                     `json:"receipt_digest"`
	Artifact         *cudakernel.Artifact       `json:"artifact"`
	Identity         cudaevidence.Identity      `json:"identity"`
	Environment      Environment                `json:"environment"`
	Status           string                     `json:"status"`
	Reason           string                     `json:"reason,omitempty"`
	Correctness      Correctness                `json:"correctness"`
	Facts            []Fact                     `json:"facts"`
}

func digest(data []byte) string { h := sha256.Sum256(data); return hex.EncodeToString(h[:]) }
func validDigest(s string) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == 32 && s == strings.ToLower(s)
}

// Validate checks content digests and consistency with existing cache keys.
func (r Record) Validate() error {
	if err := boundedValue(reflect.ValueOf(r), 64<<20); err != nil {
		return err
	}
	if r.Version != Version {
		return fmt.Errorf("history: unsupported version")
	}
	for _, s := range []string{r.SourceDigest, r.InputDigest, r.PlanDigest, r.ReceiptDigest} {
		if !validDigest(s) {
			return fmt.Errorf("history: invalid content digest")
		}
	}
	if len(r.Receipt) == 0 || digest(r.Receipt) != r.ReceiptDigest {
		return fmt.Errorf("history: receipt digest mismatch")
	}
	if r.CompilerInputKey == "" || r.Compiler.NVRTCVersion == "" || r.Compiler.DriverSONAME == "" || r.Compiler.TargetArch != r.Target || r.Entry == "" || r.Target == "" {
		return fmt.Errorf("history: incomplete compiler or entry identity")
	}
	if r.CompilerKey != cudacache.CompilerCacheKey(r.CompilerInputKey, r.Compiler) {
		return fmt.Errorf("history: compiler key mismatch")
	}
	if r.Workload.ProgramID == "" || r.NumericalMode == "" || r.Workload.NumericalMode != r.NumericalMode || r.WorkloadKey != r.Workload.Key() {
		return fmt.Errorf("history: workload identity mismatch")
	}
	for _, n := range r.Workload.ShapeDims {
		if n < 0 {
			return fmt.Errorf("history: negative workload dimension")
		}
	}
	switch r.Status {
	case "complete", "unqualified", "execution_failed", "compile_failed":
	default:
		return fmt.Errorf("history: invalid outcome")
	}
	if r.Status != "complete" && r.Reason == "" {
		return fmt.Errorf("history: noncomplete outcome needs reason")
	}
	if r.Artifact == nil {
		if r.Status != "compile_failed" || r.Identity != (cudaevidence.Identity{}) || r.Correctness.Verdict == Pass {
			return fmt.Errorf("history: missing artifact is not a compile failure")
		}
	} else {
		kind := r.Artifact.Kind
		valid := false
		switch r.Compiler.OutputKind {
		case nvrtcx.OutputAuto:
			valid = kind == cudakernel.KindPTX || kind == cudakernel.KindCUBIN
		case nvrtcx.OutputPTX:
			valid = kind == cudakernel.KindPTX
		case nvrtcx.OutputCUBIN:
			valid = kind == cudakernel.KindCUBIN
		case nvrtcx.OutputLTOIR:
			valid = kind == cudakernel.KindLTOIR
		case nvrtcx.OutputTileIR:
			valid = kind == cudakernel.KindTileIR
		case nvrtcx.OutputPTXTileIR:
			valid = kind == cudakernel.KindPTX || kind == cudakernel.KindTileIR
		}
		if !valid {
			return fmt.Errorf("history: compiler output and artifact format differ or are unsupported")
		}
		if r.Status == "compile_failed" {
			return fmt.Errorf("history: compile failure carries an image")
		}
		if err := r.Artifact.Validate(); err != nil {
			return fmt.Errorf("history artifact: %w", err)
		}
		id := cudaevidence.Identity{ArtifactDigest: r.Artifact.Digest, Entry: r.Artifact.EntryPoint, Target: r.Artifact.Target, CompilerKey: r.CompilerKey, WorkloadKey: r.WorkloadKey}
		if err := id.Validate(); err != nil {
			return err
		}
		if id != r.Identity || r.Entry != id.Entry || r.Target != id.Target {
			return fmt.Errorf("history: artifact identity mismatch")
		}
	}
	c := r.Correctness
	switch c.Verdict {
	case Unknown:
		if c.ActualDigest != "" || c.ExpectedDigest != "" || c.Method != "" || c.Oracle != "" {
			return fmt.Errorf("history: unknown correctness carries an oracle result")
		}
	case Pass, Fail:
		if r.Artifact == nil || c.Method != "digest-exact" || c.Oracle == "" || !validDigest(c.ExpectedDigest) || !validDigest(c.ActualDigest) {
			return fmt.Errorf("history: incomplete correctness evidence")
		}
		if (c.Verdict == Pass) != (c.ExpectedDigest == c.ActualDigest) {
			return fmt.Errorf("history: correctness verdict contradicts digests")
		}
	default:
		return fmt.Errorf("history: invalid correctness verdict")
	}
	names := map[string]bool{}
	for _, f := range r.Facts {
		if f.Name == "" || (f.Axis != "codegen" && f.Axis != "resource" && f.Axis != "performance") {
			return fmt.Errorf("history: invalid fact axis or name")
		}
		if f.Samples < 0 {
			return fmt.Errorf("history: negative sample count")
		}
		k := f.Axis + "/" + f.Name
		if names[k] {
			return fmt.Errorf("history: duplicate fact")
		}
		names[k] = true
		v := f.Value.Record()
		if _, err := cudaevidence.New(v); err != nil {
			return err
		}
		if f.Axis == "performance" && v.Class != cudaevidence.Unread && v.Loss != nil && uint64(f.Samples) != v.Loss.Total-v.Loss.Dropped {
			return fmt.Errorf("history: timing samples differ from retained loss population")
		}
		if v.Class != cudaevidence.Unread && (r.Artifact == nil || v.Source.Identity != r.Identity) {
			return fmt.Errorf("history: stale fact identity")
		}
		if v.Source.Identity != (cudaevidence.Identity{}) && v.Source.Identity != r.Identity {
			return fmt.Errorf("history: stale unread identity")
		}
		if r.Artifact == nil && (v.Class != cudaevidence.Unread || f.Qualified) {
			return fmt.Errorf("history: missing artifact carries usable facts")
		}
	}
	if r.Key != "" {
		key, err := r.observationKey()
		if err != nil {
			return err
		}
		if key != r.Key {
			return fmt.Errorf("history: observation key mismatch")
		}
	}
	return nil
}

func (r Record) observationKey() (string, error) {
	r.Key = ""
	data, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	return digest(data), nil
}

// ObservationKey validates a record and hashes its complete canonical JSON.
// This key is distinct from the artifact digest and compiler cache key.
func ObservationKey(r Record) (string, error) {
	if err := r.Validate(); err != nil {
		return "", err
	}
	return r.observationKey()
}
