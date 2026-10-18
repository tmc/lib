package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/tmc/lib/nvidia/x/cudadiageval"
)

type registeredPairInputs struct {
	registration, ledger, candidate, comparator, root string
}

// This command reports integrity and factual arithmetic. It has no independent
// acceptance provider, and does not construct a claim or execute receipt commands.
type registeredPairReport struct {
	Version     int               `json:"version"`
	Status      string            `json:"status"`
	Stage       string            `json:"stage"`
	Error       string            `json:"error,omitempty"`
	InputSHA256 map[string]string `json:"input_sha256"`
	// Keep every original attempt and reference, including failed attempts.
	DeclaredLedger *cudadiageval.QualificationLedger `json:"declared_ledger,omitempty"`
	// Registration and pair admission acquire separate bounded snapshots.
	Registration          *cudadiageval.RegistrationReport `json:"registration_snapshot,omitempty"`
	Pair                  *cudadiageval.PairIdentity       `json:"pair_identity,omitempty"`
	Results               []cudadiageval.MatchedResult     `json:"results,omitempty"`
	MeasurementAcceptance string                           `json:"measurement_acceptance"`
	IndependentAcceptance string                           `json:"independent_acceptance"`
	AcceptanceReason      string                           `json:"acceptance_reason"`
}

func cmdRegisteredPair(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("registered-pair", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var in registeredPairInputs
	fs.StringVar(&in.registration, "registration", "", "Frozen registration JSON")
	fs.StringVar(&in.ledger, "ledger", "", "Prerequisite qualification ledger JSON")
	fs.StringVar(&in.candidate, "candidate", "", "Candidate registered receipt envelope JSON")
	fs.StringVar(&in.comparator, "comparator", "", "Comparator registered receipt envelope JSON")
	fs.StringVar(&in.root, "root", "", "Artifact root; ancestors must be trusted")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if in.registration == "" || in.ledger == "" || in.candidate == "" || in.comparator == "" || in.root == "" || fs.NArg() != 0 {
		return errors.New("registered-pair requires -registration, -ledger, -candidate, -comparator and -root with no positional arguments")
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	return runRegisteredPair(ctx, in, stdout)
}

func runRegisteredPair(ctx context.Context, in registeredPairInputs, stdout io.Writer) error {
	report := registeredPairReport{
		Version: 1, Status: "refused", InputSHA256: make(map[string]string),
		MeasurementAcceptance: "unverified", IndependentAcceptance: "unavailable",
		AcceptanceReason: "independent acceptance provider not configured; integrity and factual results do not authenticate measurements",
	}
	read := func(stage, path string) ([]byte, error) {
		report.Stage = stage
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		b, err := readComparisonFile(path, 4<<20)
		if err != nil {
			return nil, err
		}
		report.InputSHA256[stage] = fmt.Sprintf("%x", sha256.Sum256(b))
		return b, ctx.Err()
	}
	evaluate := func() error {
		// Read the ledger first so later refusals retain safely decoded attempts.
		b, err := read("ledger", in.ledger)
		if err != nil {
			return err
		}
		ledger, err := cudadiageval.ReadQualificationLedger(bytes.NewReader(b))
		if err != nil {
			return err
		}
		report.DeclaredLedger = ledger
		b, err = read("registration", in.registration)
		if err != nil {
			return err
		}
		registration, err := cudadiageval.ReadRegistration(bytes.NewReader(b))
		if err != nil {
			return err
		}
		b, err = read("candidate", in.candidate)
		if err != nil {
			return err
		}
		candidate, err := cudadiageval.ReadRegisteredReceipt(bytes.NewReader(b))
		if err != nil {
			return err
		}
		b, err = read("comparator", in.comparator)
		if err != nil {
			return err
		}
		comparator, err := cudadiageval.ReadRegisteredReceipt(bytes.NewReader(b))
		if err != nil {
			return err
		}
		report.Stage = "registration-snapshot"
		snapshot, err := cudadiageval.VerifyRegistration(ctx, registration, *ledger, in.root)
		if err != nil {
			return err
		}
		report.Registration = &snapshot
		report.Stage = "pair-admission"
		pair, err := cudadiageval.ValidateRegisteredPair(ctx, registration, *ledger, candidate, comparator, in.root)
		if err != nil {
			return err
		}
		identity := pair.Identity()
		report.Pair = &identity
		report.Stage = "comparison"
		comparison, err := cudadiageval.CompareRegisteredPair(ctx, pair)
		if err != nil {
			return err
		}
		report.Results = comparison.Results()
		report.Status = "factual"
		return nil
	}
	err := evaluate()
	if err != nil {
		report.Error = err.Error()
	}
	if writeErr := writeComparisonJSON(stdout, report); writeErr != nil {
		return errors.Join(err, writeErr)
	}
	if err != nil {
		return fmt.Errorf("registered-pair %s: %w", report.Stage, err)
	}
	return nil
}
