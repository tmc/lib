package nvtelemetry

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func recoveryPreflightRoot(t *testing.T) string {
	t.Helper()
	p := t.TempDir()
	if err := os.Chmod(p, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(p, "owner.json"), []byte(`{"version":1,"schema_version":1,"owner":"preflight"}`), 0600); err != nil {
		t.Fatal(err)
	}
	return p
}

func recoveryWrite(t *testing.T, p, name string, b []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(p, name), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestRecoveryPreflightRefusalPreservesEvidence(t *testing.T) {
	for _, name := range []string{"oversized", "invalid JSON", "unknown envelope", "empty metrics", "mixed kinds", "too many records", "wrong owner", "missing owner", "unknown version", "foreign file", "temporary directory"} {
		t.Run(name, func(t *testing.T) {
			p := recoveryPreflightRoot(t)
			tmp := formatPendingTmpName(1)
			evidence := []byte("unpublished bytes")
			recoveryWrite(t, p, tmp, evidence)
			if err := os.Mkdir(filepath.Join(p, "lease"), 0700); err != nil {
				t.Fatal(err)
			}
			recoveryWrite(t, p, "lease/lock", []byte("retained lease"))
			recoveryWrite(t, p, formatSpoolName(1), []byte(deliveryFixture))
			switch name {
			case "oversized":
				f, err := os.OpenFile(filepath.Join(p, formatSpoolName(1)), os.O_RDWR, 0600)
				if err != nil {
					t.Fatal(err)
				}
				if err := f.Truncate(1<<30 + 1); err != nil {
					f.Close()
					t.Fatal(err)
				}
				if err := f.Close(); err != nil {
					t.Fatal(err)
				}
			case "invalid JSON":
				recoveryWrite(t, p, formatSpoolName(1), []byte("not a valid OTLP batch"))
			case "unknown envelope":
				recoveryWrite(t, p, formatSpoolName(1), []byte(`{"payload":[1]}`))
			case "empty metrics":
				recoveryWrite(t, p, formatSpoolName(1), []byte(`{"resourceMetrics":[]}`))
			case "mixed kinds":
				recoveryWrite(t, p, formatSpoolName(1), []byte(`{"resourceMetrics":[],"resourceLogs":[]}`))
			case "too many records":
				recoveryWrite(t, p, formatSpoolName(1), []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[`+strings.Repeat(`{},`, 4096)+`{}]}]}]}`))
			case "wrong owner":
				recoveryWrite(t, p, "owner.json", []byte(`{"version":1,"owner":"different"}`))
			case "missing owner":
				if err := os.Remove(filepath.Join(p, "owner.json")); err != nil {
					t.Fatal(err)
				}
			case "unknown version":
				recoveryWrite(t, p, "owner.json", []byte(`{"version":1,"schema_version":2,"owner":"preflight"}`))
			case "foreign file":
				recoveryWrite(t, p, "foreign.json", []byte("unrecognized"))
			case "temporary directory":
				if err := os.Mkdir(filepath.Join(p, formatPendingTmpName(2)), 0700); err != nil {
					t.Fatal(err)
				}
			}
			before, err := os.ReadDir(p)
			if err != nil {
				t.Fatal(err)
			}
			m, err := RecoverDurableDeliveryManifest(p, "preflight")
			if err == nil || m != (DurableDeliveryRecoveryManifest{}) {
				t.Fatalf("invalid recovery admitted: %+v %v", m, err)
			}
			after, err := os.ReadDir(p)
			if err != nil {
				t.Fatal(err)
			}
			var beforeNames, afterNames []string
			for _, e := range before {
				beforeNames = append(beforeNames, e.Name())
			}
			for _, e := range after {
				afterNames = append(afterNames, e.Name())
			}
			if !reflect.DeepEqual(beforeNames, afterNames) {
				t.Fatalf("refusal changed namespace: %v -> %v", beforeNames, afterNames)
			}
			for file, want := range map[string][]byte{tmp: evidence, "lease/lock": []byte("retained lease")} {
				got, err := os.ReadFile(filepath.Join(p, file))
				if err != nil || !bytes.Equal(got, want) {
					t.Fatalf("refusal changed %s: %q %v", file, got, err)
				}
			}
		})
	}
}

func TestRecoveryPreflightDoesNotInitializeOwner(t *testing.T) {
	p := t.TempDir()
	if err := os.Chmod(p, 0700); err != nil {
		t.Fatal(err)
	}
	if m, err := RecoverDurableDeliveryManifest(p, "preflight"); err == nil || m != (DurableDeliveryRecoveryManifest{}) {
		t.Fatalf("unowned empty directory admitted: %+v %v", m, err)
	}
	entries, err := os.ReadDir(p)
	if err != nil || len(entries) != 0 {
		t.Fatalf("unowned directory changed: %v %v", entries, err)
	}
}

func TestRecoveryPreflightPublicEnqueue(t *testing.T) {
	for _, logs := range []bool{false, true} {
		t.Run(fmt.Sprint(logs), func(t *testing.T) {
			p := recoveryPreflightRoot(t)
			root, err := os.OpenRoot(p)
			if err != nil {
				t.Fatal(err)
			}
			defer root.Close()
			payload := []byte(strings.Replace(deliveryFixture, `"resourceMetrics"`, `"callerExtension":1,"resourceMetrics"`, 1))
			if logs {
				payload = []byte(`{"resourceLogs":[{"scopeLogs":[{"logRecords":[{"body":{"stringValue":"constructed"}}]}]}]}`)
			}
			// No worker or transport is created: exercise the public admission and
			// its real on-disk commit path without network or vendor constructors.
			d := &DurableDelivery{root: root, isLogs: logs, done: make(chan struct{}), config: DurableDeliveryConfig{MaxSpoolBatches: 2, MaxSpoolBytes: 8 << 20, Delivery: DeliveryConfig{MaxBatchBytes: 4 << 20, MaxBatchDataPoints: 4096}}}
			if err := d.Enqueue(context.Background(), payload); err != nil {
				t.Fatal(err)
			}
			recoveryWrite(t, p, formatPendingTmpName(2), []byte("torn temporary request"))
			m, err := RecoverDurableDeliveryManifest(p, "preflight")
			if err != nil || m.PendingBatches != 1 || m.PendingBytes != int64(len(payload)) || m.RemovedTempFiles != 1 || m.RecoveredLease {
				t.Fatalf("manifest=%+v error=%v", m, err)
			}
			got, err := os.ReadFile(filepath.Join(p, formatSpoolName(1)))
			if err != nil || !bytes.Equal(got, payload) {
				t.Fatalf("committed payload changed: %q %v", got, err)
			}
			if _, err := os.Stat(filepath.Join(p, formatPendingTmpName(2))); !os.IsNotExist(err) {
				t.Fatalf("temporary not removed: %v", err)
			}
			var b bytes.Buffer
			if err := WriteDurableDeliveryRecoveryManifest(&b, m); err != nil {
				t.Fatal(err)
			}
			decoded, err := ReadDurableDeliveryRecoveryManifest(&b)
			if err != nil || decoded != m {
				t.Fatalf("manifest roundtrip=%+v %v", decoded, err)
			}
		})
	}
}

func TestRecoveryPreflightLiveLease(t *testing.T) {
	p := recoveryPreflightRoot(t)
	if err := os.Mkdir(filepath.Join(p, "lease"), 0700); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(p, "lease/lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := lockFileExclusive(f); err != nil {
		t.Fatal(err)
	}
	defer unlockFile(f)
	recoveryWrite(t, p, formatPendingTmpName(1), []byte("unpublished"))
	if _, err := RecoverDurableDeliveryManifest(p, "preflight"); err == nil {
		t.Fatal("live writer recovered")
	}
	if b, err := os.ReadFile(filepath.Join(p, formatPendingTmpName(1))); err != nil || string(b) != "unpublished" {
		t.Fatalf("live recovery changed temp: %q %v", b, err)
	}
}
