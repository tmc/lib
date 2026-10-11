package main

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

type sidecarNativeReadback struct {
	Phase      string
	State      cudahotswap.State
	OutputBits []uint32
}

func sidecarSameVersion(a, b cudahotswap.State) bool {
	return a.Name == b.Name && a.Kind == b.Kind && a.Version == b.Version && a.SourceDigest == b.SourceDigest && a.CompilerKey == b.CompilerKey && a.ConfigRevision == b.ConfigRevision
}

func sidecarValidateNativeReport(index int, result *cudahotswap.PushResult, reportError, code string, row sidecarNativeReadback) error {
	if index < 0 || index > 2 || result == nil {
		return fmt.Errorf("sidecar native report missing or out of range")
	}
	wrong := index == 1
	outcomes := []cudahotswap.Outcome{cudahotswap.OutcomeOK, cudahotswap.OutcomeWrongAnswer, cudahotswap.OutcomeOK}
	if (reportError != "") != wrong || result.Report.Outcome != outcomes[index] || !result.Published || result.RolledBack != wrong {
		return fmt.Errorf("sidecar native report outcome disagrees")
	}
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(code)))
	if result.Report.Compile == nil || result.Report.Compile.SourceSHA256 != hash || result.State.SourceDigest != hash {
		return fmt.Errorf("sidecar native report source differs")
	}
	state := result.State
	if state.Name != "kernel" || state.Kind != "kernel" || state.Generation == 0 || state.Version == 0 || state.CompilerKey == "" || result.Report.Generation != state.Generation || result.Report.ConfigRevision != state.ConfigRevision {
		return fmt.Errorf("sidecar native candidate stamp differs")
	}
	if row.Phase != "candidate" || row.State != state {
		return fmt.Errorf("sidecar native readback differs from candidate report")
	}
	if !wrong && result.Selected != state {
		return fmt.Errorf("sidecar native successful selection differs")
	}
	return nil
}

func sidecarValidateRestored(prior, candidate, selected cudahotswap.State) error {
	if !sidecarSameVersion(prior, selected) || selected.Generation <= candidate.Generation || selected.Version == candidate.Version {
		return fmt.Errorf("sidecar native prior version was not restored")
	}
	return nil
}

func sidecarValidateNativeRows(rows []sidecarNativeReadback, reports []*cudahotswap.PushResult, initial cudahotswap.State, want []float32) error {
	if len(rows) != 6 || len(reports) != 3 {
		return fmt.Errorf("sidecar native row/report cardinality differs")
	}
	for _, report := range reports {
		if report == nil {
			return fmt.Errorf("sidecar native report missing")
		}
	}
	if initial.Name != "kernel" || initial.Kind != "kernel" || initial.Generation == 0 || initial.Version == 0 {
		return fmt.Errorf("sidecar native initial identity missing")
	}
	phases := []string{"setup-unbound", "initial-verified", "candidate", "candidate", "restored", "candidate"}
	states := []cudahotswap.State{{}, initial, reports[0].State, reports[1].State, reports[1].Selected, reports[2].State}
	for i, row := range rows {
		if row.Phase != phases[i] || row.State != states[i] || len(row.OutputBits) != len(want) {
			return fmt.Errorf("sidecar native row %d identity or extent differs", i)
		}
		out := make([]float32, len(row.OutputBits))
		for j, bits := range row.OutputBits {
			out[j] = math.Float32frombits(bits)
		}
		if i == 3 {
			for _, bits := range row.OutputBits {
				if bits != 0 {
					return fmt.Errorf("sidecar native wrong-store readback is not literal zero")
				}
			}
			if compareSidecar(want, out) == nil {
				return fmt.Errorf("sidecar native planted wrong stores agree with oracle")
			}
		} else if err := compareSidecar(want, out); err != nil {
			return fmt.Errorf("sidecar native row %d output: %w", i, err)
		}
	}
	return sidecarValidateRestored(reports[0].State, reports[1].State, reports[1].Selected)
}

// Synthetic values exercise the strict receipt validator only, not CUDA.
func sidecarNativeContractFixture() ([]sidecarNativeReadback, []*cudahotswap.PushResult, cudahotswap.State, []string) {
	codes := []string{"good", "wrong", "repair"}
	initial := cudahotswap.State{Name: "kernel", Kind: "kernel", Generation: 1, Version: 1, ConfigRevision: 2, CompilerKey: "initial", SourceDigest: "initial"}
	reports := make([]*cudahotswap.PushResult, 3)
	for i, code := range codes {
		hash := fmt.Sprintf("%x", sha256.Sum256([]byte(code)))
		state := cudahotswap.State{Name: "kernel", Kind: "kernel", Generation: uint64(i + 2), Version: uint64(i + 2), ConfigRevision: 3, SourceDigest: hash, CompilerKey: code}
		if i == 2 {
			state.Generation = 5
		}
		reports[i] = &cudahotswap.PushResult{State: state, Selected: state, Published: true, Report: cudahotswap.Report{Outcome: cudahotswap.OutcomeOK, Generation: state.Generation, ConfigRevision: state.ConfigRevision, Compile: &cudahotswap.CompileReport{SourceSHA256: hash}}}
	}
	reports[1].RolledBack = true
	reports[1].Report.Outcome = cudahotswap.OutcomeWrongAnswer
	reports[1].Selected = reports[0].State
	reports[1].Selected.Generation = 4
	reports[1].Selected.Label = "rollback"
	bits := []uint32{math.Float32bits(7)}
	rows := []sidecarNativeReadback{{"setup-unbound", cudahotswap.State{}, bits}, {"initial-verified", initial, bits}, {"candidate", reports[0].State, bits}, {"candidate", reports[1].State, []uint32{0}}, {"restored", reports[1].Selected, bits}, {"candidate", reports[2].State, bits}}
	return rows, reports, initial, codes
}
func TestSidecarNativeReceiptAssociations(t *testing.T) {
	rows, reports, initial, codes := sidecarNativeContractFixture()
	for i, r := range reports {
		text := ""
		if i == 1 {
			text = "wrong"
		}
		if err := sidecarValidateNativeReport(i, r, text, codes[i], rows[i+2+(i/2)]); err != nil {
			t.Fatal(err)
		}
	}
	if err := sidecarValidateNativeRows(rows, reports, initial, []float32{7}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"generation", "version", "source", "compiler", "row-phase", "missing-restored", "output", "wrong-zero", "selected", "report-generation"} {
		t.Run(name, func(t *testing.T) {
			r, rs, init, cs := sidecarNativeContractFixture()
			switch name {
			case "generation":
				r[2].State.Generation++
			case "version":
				r[2].State.Version++
			case "source":
				r[2].State.SourceDigest = "other"
			case "compiler":
				r[4].State.CompilerKey = "other"
			case "row-phase":
				r[2].Phase = "restored"
			case "missing-restored":
				r = append(r[:4], r[5:]...)
			case "output":
				r[4].OutputBits = []uint32{math.Float32bits(99)}
			case "wrong-zero":
				r[3].OutputBits = []uint32{math.Float32bits(float32(math.NaN()))}
			case "selected":
				rs[1].Selected.Version = rs[1].State.Version
			case "report-generation":
				rs[0].Report.Generation++
			}
			err := sidecarValidateNativeRows(r, rs, init, []float32{7})
			if name == "report-generation" {
				err = sidecarValidateNativeReport(0, rs[0], "", cs[0], r[2])
			}
			if err == nil {
				t.Fatal("altered evidence accepted")
			}
		})
	}
	if !reflect.DeepEqual(rows[4].State, reports[1].Selected) {
		t.Fatal("validator mutated raw receipt")
	}
}

func sidecarFinishNativeReceipt(capture, write, close func() error) error {
	return errors.Join(capture(), write(), close())
}
func TestSidecarNativeReceiptCloseErrors(t *testing.T) {
	failure := errors.New("actual close control")
	for _, failed := range []string{"capture", "write", "close"} {
		order := ""
		err := sidecarFinishNativeReceipt(func() error {
			order += "capture;"
			if failed == "capture" {
				return failure
			}
			return nil
		}, func() error {
			order += "write;"
			if failed == "write" {
				return failure
			}
			return nil
		}, func() error {
			order += "close;"
			if failed == "close" {
				return failure
			}
			return nil
		})
		if order != "capture;write;close;" || !errors.Is(err, failure) {
			t.Fatalf("terminal refusal lost %s: order=%q error=%v", failed, order, err)
		}
	}
}
func FuzzSidecarNativeRows(f *testing.F) {
	f.Add([]byte{1, 2, 3})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 256 {
			return
		}
		rows, reports, initial, _ := sidecarNativeContractFixture()
		for i, b := range data {
			switch i % 7 {
			case 0:
				rows[int(b)%6].State.Generation += uint64(b)
			case 1:
				rows[int(b)%6].Phase = string([]byte{b})
			case 2:
				rows[int(b)%6].OutputBits = []uint32{uint32(b) << 24}
			case 3:
				reports[int(b)%3].Selected.Version += uint64(b)
			case 4:
				reports[int(b)%3].Report.Generation += uint64(b)
			case 5:
				if b%2 == 0 {
					reports[int(b)%3] = nil
				}
			case 6:
				initial.Generation += uint64(b)
			}
			// Stop after a nil report to avoid the fuzzer itself dereferencing it.
			if reports[0] == nil || reports[1] == nil || reports[2] == nil {
				break
			}
		}
		_ = sidecarValidateNativeRows(rows, reports, initial, []float32{7})
	})
}

func TestSidecarNativeRestorationIdentity(t *testing.T) {
	_, reports, _, _ := sidecarNativeContractFixture()
	prior, candidate, selected := reports[0].State, reports[1].State, reports[1].Selected
	if err := sidecarValidateRestored(prior, candidate, selected); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"version", "source", "compiler", "config", "generation", "candidate"} {
		t.Run(name, func(t *testing.T) {
			s := selected
			switch name {
			case "version":
				s.Version += 99
			case "source":
				s.SourceDigest = "other"
			case "compiler":
				s.CompilerKey = "other"
			case "config":
				s.ConfigRevision++
			case "generation":
				s.Generation = candidate.Generation
			case "candidate":
				s.Version = candidate.Version
			}
			if err := sidecarValidateRestored(prior, candidate, s); err == nil {
				t.Fatal("altered restored version accepted")
			}
		})
	}
}
