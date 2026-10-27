package nvtelemetry_test

import (
	"errors"
	"io"
	"io/fs"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestReadProcEnviron(t *testing.T) {
	tests := []struct {
		name, input string
		state       nvtelemetry.ProcEnvironState
		job         string
		bad         bool
	}{
		{"empty", "", nvtelemetry.ProcEnvironAbsent, "", false},
		{"unrelated", "TOKEN=private\x00HOME=/private\x00", nvtelemetry.ProcEnvironAbsent, "", false},
		{"empty value", "SLURM_JOB_ID=\x00", nvtelemetry.ProcEnvironAbsent, "", false},
		{"self reported", "TOKEN=private\x00SLURM_JOB_ID=999\x00", nvtelemetry.ProcEnvironSelfReported, "999", false},
		{"untrusted nonnumeric", "SLURM_JOB_ID=made-up\x00", nvtelemetry.ProcEnvironSelfReported, "made-up", false},
		{"value equals", "SLURM_JOB_ID=a=b\x00", nvtelemetry.ProcEnvironSelfReported, "a=b", false},
		{"partial", "SLURM_JOB_ID=999", nvtelemetry.ProcEnvironPartial, "", true},
		{"partial unknown after value", "SLURM_JOB_ID=999\x00TOKEN=secret", nvtelemetry.ProcEnvironPartial, "", true},
		{"duplicate", "SLURM_JOB_ID=999\x00SLURM_JOB_ID=999\x00", nvtelemetry.ProcEnvironInvalid, "", true},
		{"conflict", "SLURM_JOB_ID=999\x00SLURM_JOB_ID=123\x00", nvtelemetry.ProcEnvironInvalid, "", true},
		{"malformed", "SLURM_JOB_ID=999\x00BROKEN\x00", nvtelemetry.ProcEnvironInvalid, "", true},
		{"empty name", "=private\x00", nvtelemetry.ProcEnvironInvalid, "", true},
		{"invalid utf8", "SLURM_JOB_ID=\xff\x00", nvtelemetry.ProcEnvironInvalid, "", true},
		{"oversize value", "SLURM_JOB_ID=" + strings.Repeat("x", 257) + "\x00", nvtelemetry.ProcEnvironInvalid, "", true},
		{"value boundary", "SLURM_JOB_ID=" + strings.Repeat("x", 256) + "\x00", nvtelemetry.ProcEnvironSelfReported, strings.Repeat("x", 256), false},
		{"input boundary", "X=" + strings.Repeat("x", 65533) + "\x00", nvtelemetry.ProcEnvironAbsent, "", false},
		{"oversize input", "X=" + strings.Repeat("x", 65534) + "\x00", nvtelemetry.ProcEnvironInvalid, "", true},
		{"too many pairs", strings.Repeat("X=\x00", 4097), nvtelemetry.ProcEnvironInvalid, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := nvtelemetry.ReadProcEnviron(strings.NewReader(tt.input))
			if got.State != tt.state || got.SlurmJobID != tt.job || (err != nil) != tt.bad {
				t.Fatalf("got %+v, %v; want %q %q error=%v", got, err, tt.state, tt.job, tt.bad)
			}
		})
	}
}

type environErrorReader struct{ err error }

func (r environErrorReader) Read(p []byte) (int, error) {
	return copy(p, "SLURM_JOB_ID=999\x00"), r.err
}

func TestReadProcEnvironIO(t *testing.T) {
	readErr := errors.New("incomplete environment read")
	for _, tt := range []struct {
		name  string
		r     io.Reader
		state nvtelemetry.ProcEnvironState
		err   error
	}{
		{"nil", nil, nvtelemetry.ProcEnvironInvalid, nil},
		{"partial read error", environErrorReader{readErr}, nvtelemetry.ProcEnvironReadError, readErr},
		{"permission read error", environErrorReader{fs.ErrPermission}, nvtelemetry.ProcEnvironPermissionDenied, fs.ErrPermission},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := nvtelemetry.ReadProcEnviron(tt.r)
			if err == nil || got.State != tt.state || got.SlurmJobID != "" || tt.err != nil && !errors.Is(err, tt.err) {
				t.Fatalf("got %+v, %v", got, err)
			}
		})
	}
	var zero nvtelemetry.ProcEnvironFacts
	if zero.State != nvtelemetry.ProcEnvironUnknown || zero.SlurmJobID != "" {
		t.Fatalf("zero observation %+v", zero)
	}
}
