package nvtelemetry

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"unicode/utf8"
)

// ProcEnvironState describes an untrusted environment observation.
type ProcEnvironState string

const (
	// ProcEnvironUnknown means no environment observation was made.
	ProcEnvironUnknown ProcEnvironState = ""
	// ProcEnvironAbsent means the file or supported value was absent.
	ProcEnvironAbsent ProcEnvironState = "absent"
	// ProcEnvironSelfReported means a bounded SLURM_JOB_ID value was observed.
	ProcEnvironSelfReported ProcEnvironState = "self_reported"
	// ProcEnvironInvalid means the complete input was malformed or too large.
	ProcEnvironInvalid ProcEnvironState = "invalid"
	// ProcEnvironPartial means the last environment pair was not terminated.
	ProcEnvironPartial ProcEnvironState = "partial"
	// ProcEnvironPermissionDenied means access to the observation was denied.
	ProcEnvironPermissionDenied ProcEnvironState = "permission_denied"
	// ProcEnvironReadError means I/O failed; no supported value is retained.
	ProcEnvironReadError ProcEnvironState = "read_error"
)

// ProcEnvironFacts contains only the supported self-reported environment value.
// SlurmJobID is nonempty only when State is ProcEnvironSelfReported. It is not
// a validated job identity, cgroup fact, scheduler proof or allocation authority.
// The zero value means no observation was made. No other environment value is
// retained, and these facts must not become metric labels or assigned owners.
type ProcEnvironFacts struct {
	State      ProcEnvironState
	SlurmJobID string
}

// ReadProcEnviron reads at most 64 KiB of NUL-separated environment pairs.
// Only SLURM_JOB_ID is retained, as a self-reported UTF-8 string of at most
// 256 bytes. Unknown pairs are ignored. Every pair must contain a nonempty
// name, and duplicate SLURM_JOB_ID entries refuse even if their values agree.
// Empty input or a missing/empty supported value is absent. Invalid, partial
// and error-bearing reads return their state and an error, with no value.
// Reading this file cannot prove an atomic process snapshot or job ownership.
func ReadProcEnviron(r io.Reader) (ProcEnvironFacts, error) {
	refuse := func(state ProcEnvironState, err error) (ProcEnvironFacts, error) {
		return ProcEnvironFacts{State: state}, err
	}
	if r == nil {
		return refuse(ProcEnvironInvalid, errors.New("proc environ: nil reader"))
	}
	b, err := io.ReadAll(io.LimitReader(r, 65537))
	if err != nil {
		state := ProcEnvironReadError
		if errors.Is(err, fs.ErrPermission) {
			state = ProcEnvironPermissionDenied
		}
		return refuse(state, fmt.Errorf("proc environ: read: %w", err))
	}
	if len(b) > 65536 {
		return refuse(ProcEnvironInvalid, errors.New("proc environ: input too large"))
	}
	if len(b) == 0 {
		return ProcEnvironFacts{State: ProcEnvironAbsent}, nil
	}
	if b[len(b)-1] != 0 {
		return refuse(ProcEnvironPartial, errors.New("proc environ: unterminated pair"))
	}
	var value string
	seen := false
	pairs := bytes.Split(b[:len(b)-1], []byte{0})
	if len(pairs) > entityMaxRecords {
		return refuse(ProcEnvironInvalid, errors.New("proc environ: too many pairs"))
	}
	for _, pair := range pairs {
		name, v, ok := bytes.Cut(pair, []byte{'='})
		if !ok || len(name) == 0 {
			return refuse(ProcEnvironInvalid, errors.New("proc environ: malformed pair"))
		}
		if !bytes.Equal(name, []byte("SLURM_JOB_ID")) {
			continue
		}
		if seen {
			return refuse(ProcEnvironInvalid, errors.New("proc environ: repeated slurm job id"))
		}
		seen = true
		if len(v) > 256 || !utf8.Valid(v) {
			return refuse(ProcEnvironInvalid, errors.New("proc environ: invalid slurm job id value"))
		}
		value = string(v)
	}
	if value == "" {
		return ProcEnvironFacts{State: ProcEnvironAbsent}, nil
	}
	return ProcEnvironFacts{State: ProcEnvironSelfReported, SlurmJobID: value}, nil
}

// readProcEnvironFile deliberately retains only a fixed observation state on
// failure. Optional environment failure never discards separately valid proc
// identity/cgroup facts or exposes arbitrary native/path error text.
func readProcEnvironFile(fsys fs.FS, name string) ProcEnvironFacts {
	f, err := fsys.Open(name)
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			return ProcEnvironFacts{State: ProcEnvironAbsent}
		case errors.Is(err, fs.ErrPermission):
			return ProcEnvironFacts{State: ProcEnvironPermissionDenied}
		default:
			return ProcEnvironFacts{State: ProcEnvironReadError}
		}
	}
	facts, _ := ReadProcEnviron(f)
	if err := f.Close(); err != nil {
		return ProcEnvironFacts{State: ProcEnvironReadError}
	}
	return facts
}
