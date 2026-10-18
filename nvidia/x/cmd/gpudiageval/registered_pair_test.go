package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiageval"
)

func registeredPairCommandFixture(t *testing.T) ([]string, string) {
	t.Helper()
	root := t.TempDir()
	if err := os.CopyFS(root, os.DirFS("testdata/registered-pair")); err != nil {
		t.Fatal(err)
	}
	return []string{"registered-pair", "-registration", filepath.Join(root, "registration.json"), "-ledger", filepath.Join(root, "ledger.json"), "-candidate", filepath.Join(root, "candidate-envelope.json"), "-comparator", filepath.Join(root, "comparator-envelope.json"), "-root", filepath.Join(root, "artifacts")}, root
}

func readPairCommandReport(t *testing.T, b []byte) registeredPairReport {
	t.Helper()
	var r registeredPairReport
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	if r.IndependentAcceptance != "unavailable" || r.MeasurementAcceptance != "unverified" || !strings.Contains(r.AcceptanceReason, "provider not configured") {
		t.Fatalf("acceptance upgraded: %+v", r)
	}
	return r
}

// Constructed wire data exercises the public CLI, never authentic acquisition.
func TestRegisteredPairCommandFactual(t *testing.T) {
	args, root := registeredPairCommandFixture(t)
	var out bytes.Buffer
	if err := run(args, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	first := append([]byte(nil), out.Bytes()...)
	r := readPairCommandReport(t, first)
	if r.Status != "factual" || r.Stage != "comparison" || r.Error != "" || r.Pair == nil || r.Pair.PairDigest == "" || len(r.Results) != 1 || r.Results[0].Status != "candidate_lower" || r.Results[0].Effect != 3 {
		t.Fatalf("factual result: %+v", r)
	}
	if len(r.InputSHA256) != 4 || r.Registration == nil || r.DeclaredLedger == nil {
		t.Fatalf("missing identities: %+v", r)
	}
	if len(r.DeclaredLedger.Cells) != 2 || len(r.DeclaredLedger.Cells[0].Attempts) != 1 || r.DeclaredLedger.Cells[1].Status != "unavailable" || r.DeclaredLedger.Cells[1].Gate != "native" {
		t.Fatalf("lost attempts/native HOLD: %+v", r.DeclaredLedger)
	}
	if !r.Registration.Cells[0].ReceiptConsistency || r.Registration.Cells[1].ReceiptConsistency || r.Registration.MeasurementAcceptance != "unverified" {
		t.Fatalf("consistency upgraded: %+v", r.Registration)
	}
	b, err := os.ReadFile(filepath.Join(root, "ledger.json"))
	if err != nil {
		t.Fatal(err)
	}
	original, err := cudadiageval.ReadQualificationLedger(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(r.DeclaredLedger)
	want, _ := json.Marshal(original)
	if !bytes.Equal(got, want) {
		t.Fatal("original ledger changed")
	}
	out.Reset()
	if err := run(args, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, out.Bytes()) {
		t.Fatal("nondeterministic report")
	}
}

func TestRegisteredPairCommandRefusals(t *testing.T) {
	for _, name := range []string{"registration", "candidate-version", "candidate-alias", "candidate-duplicate", "candidate-trailing", "wrong-arm", "corrupt-raw", "missing-raw", "symlink-raw", "oversized", "ledger-attempts"} {
		t.Run(name, func(t *testing.T) {
			args, root := registeredPairCommandFixture(t)
			path := filepath.Join(root, "candidate-envelope.json")
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			switch name {
			case "registration":
				path = filepath.Join(root, "registration.json")
				b = []byte("{}")
			case "candidate-version":
				b = bytes.Replace(b, []byte(`"version":1`), []byte(`"version":2`), 1)
			case "candidate-alias":
				b = bytes.Replace(b, []byte(`"arm":`), []byte(`"Arm":`), 1)
			case "candidate-duplicate":
				b = bytes.Replace(b, []byte(`"version":1`), []byte(`"version":1,"version":1`), 1)
			case "candidate-trailing":
				b = append(b, []byte(" {}")...)
			case "wrong-arm":
				b = bytes.Replace(b, []byte(`"arm":"candidate"`), []byte(`"arm":"comparator"`), 1)
			case "corrupt-raw":
				path = filepath.Join(root, "artifacts", "raw.txt")
				b = []byte("corrupt")
			case "missing-raw":
				if err := os.Remove(filepath.Join(root, "artifacts", "raw.txt")); err != nil {
					t.Fatal(err)
				}
			case "symlink-raw":
				raw := filepath.Join(root, "artifacts", "raw.txt")
				if err := os.Rename(raw, raw+".original"); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink("raw.txt.original", raw); err != nil {
					t.Fatal(err)
				}
			case "oversized":
				b = bytes.Repeat([]byte(" "), 4<<20+1)
			case "ledger-attempts":
				path = filepath.Join(root, "ledger.json")
				raw, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				l, err := cudadiageval.ReadQualificationLedger(bytes.NewReader(raw))
				if err != nil {
					t.Fatal(err)
				}
				prior := l.Cells[0].Attempts[0]
				prior.Path = "prior-failed.json"
				prior.Evidence.Status = "failed"
				l.Cells[0].Attempts = append([]cudadiageval.QualificationAttempt{prior}, l.Cells[0].Attempts...)
				var buf bytes.Buffer
				if err := cudadiageval.WriteQualificationLedger(&buf, *l); err != nil {
					t.Fatal(err)
				}
				b = buf.Bytes()
			}
			if name != "missing-raw" && name != "symlink-raw" {
				if err := os.WriteFile(path, b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			var out bytes.Buffer
			if err := run(args, &out, io.Discard); err == nil {
				t.Fatal("refusal accepted")
			}
			r := readPairCommandReport(t, out.Bytes())
			if r.Status != "refused" || r.Error == "" || len(r.Results) != 0 || r.Pair != nil || r.DeclaredLedger == nil {
				t.Fatalf("refusal lost: %+v", r)
			}
			if name == "ledger-attempts" && (len(r.DeclaredLedger.Cells[0].Attempts) != 2 || r.DeclaredLedger.Cells[0].Attempts[0].Evidence.Status != "failed") {
				t.Fatal("prior failed attempt lost")
			}
		})
	}
}

func TestRegisteredPairCommandFlags(t *testing.T) {
	for _, args := range [][]string{{"registered-pair"}, {"registered-pair", "-unknown"}, {"registered-pair", "-registration", "a", "-ledger", "b", "-candidate", "c", "-comparator", "d", "-root", "e", "extra"}} {
		if err := run(args, io.Discard, io.Discard); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	var out bytes.Buffer
	if err := run([]string{"help"}, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "registered-pair") || !strings.Contains(out.String(), "independent acceptance is unavailable") {
		t.Fatal("help omits factual scope")
	}
}

func TestRegisteredPairCommandCancellationAndWriters(t *testing.T) {
	args, _ := registeredPairCommandFixture(t)
	in := registeredPairInputs{registration: args[2], ledger: args[4], candidate: args[6], comparator: args[8], root: args[10]}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var out bytes.Buffer
	if err := runRegisteredPair(ctx, in, &out); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	r := readPairCommandReport(t, out.Bytes())
	if r.Status != "refused" || r.Stage != "ledger" {
		t.Fatalf("cancellation upgraded: %+v", r)
	}
	if err := run(args, compareShortWriter{}, io.Discard); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write: %v", err)
	}
	sentinel := errors.New("writer refused")
	if err := run(args, pairErrorWriter{sentinel}, io.Discard); !errors.Is(err, sentinel) {
		t.Fatalf("write error: %v", err)
	}
}

type pairErrorWriter struct{ err error }

func (w pairErrorWriter) Write(p []byte) (int, error) { return 0, w.err }
