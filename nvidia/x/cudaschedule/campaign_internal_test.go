package cudaschedule

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCampaignVariantRootUsesProvidedParent(t *testing.T) {
	parent := t.TempDir()
	root, err := campaignVariantRoot(parent)
	if err != nil {
		t.Fatal(err)
	}
	rel, err := filepath.Rel(parent, root)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		t.Fatalf("campaign root %q is outside parent %q: rel=%q err=%v", root, parent, rel, err)
	}
	if info, err := os.Stat(root); err != nil || !info.IsDir() {
		t.Fatalf("campaign root stat=%v err=%v", info, err)
	}
}

func TestTakeLoadedPTXDigestStripsReceiptFromOutput(t *testing.T) {
	const digest = "0123456789abcdef"
	stdout := "PASS\n" + loadedPTXDigestPrefix + digest + "\n"
	gotOutput, gotDigest := takeLoadedPTXDigest(stdout)
	if gotOutput != "PASS\n" || gotDigest != digest {
		t.Fatalf("takeLoadedPTXDigest() = (%q, %q), want (%q, %q)", gotOutput, gotDigest, "PASS\n", digest)
	}
	duplicate := strings.Join([]string{loadedPTXDigestPrefix + digest, loadedPTXDigestPrefix + digest}, "\n")
	_, gotDigest = takeLoadedPTXDigest(duplicate)
	if gotDigest == "" || gotDigest == digest {
		t.Fatalf("duplicate receipt digest = %q, want invalid marker", gotDigest)
	}
}

func TestTakeOracleLiveness(t *testing.T) {
	for _, tc := range []struct {
		name, receipt string
		pid, modules  int
		accesses      uint64
	}{
		{"valid", "CUDASCHEDULE_ORACLE_LIVENESS:123 2 7\n", 123, 2, 7},
		{"missing", "", 0, 0, 0},
		{"invalid pid", "CUDASCHEDULE_ORACLE_LIVENESS:bad 2 7\n", 0, 0, 0},
		{"invalid modules", "CUDASCHEDULE_ORACLE_LIVENESS:123 bad 7\n", 0, 0, 0},
		{"invalid accesses", "CUDASCHEDULE_ORACLE_LIVENESS:123 2 bad\n", 0, 0, 0},
		{"extra field", "CUDASCHEDULE_ORACLE_LIVENESS:123 2 7 9\n", 0, 0, 0},
		{"duplicate", "CUDASCHEDULE_ORACLE_LIVENESS:123 2 7\nCUDASCHEDULE_ORACLE_LIVENESS:123 2 7\n", 0, 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, pid, modules, accesses := takeOracleLiveness("PASS\n" + tc.receipt)
			if pid != tc.pid || modules != tc.modules || accesses != tc.accesses {
				t.Fatalf("receipt=(%d,%d,%d) want(%d,%d,%d)", pid, modules, accesses, tc.pid, tc.modules, tc.accesses)
			}
		})
	}
}

func TestDeviceNodeAvailable(t *testing.T) {
	for _, tt := range []struct {
		name string
		goos string
		err  error
		want bool
	}{
		{name: "linux device present", goos: "linux", want: true},
		{name: "linux device absent", goos: "linux", err: errors.New("not found")},
		{name: "non-linux", goos: "darwin", want: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			got := deviceNodeAvailable(tt.goos, func(path string) error {
				calls++
				if path != "/dev/nvidiactl" {
					t.Fatalf("stat path = %q, want /dev/nvidiactl", path)
				}
				return tt.err
			})
			if got != tt.want {
				t.Fatalf("deviceNodeAvailable = %t, want %t", got, tt.want)
			}
			if tt.goos == "linux" && calls != 1 {
				t.Fatalf("Linux probe calls = %d, want 1", calls)
			}
			if tt.goos != "linux" && calls != 0 {
				t.Fatalf("non-Linux probe called stat %d times, want 0", calls)
			}
		})
	}
}

func TestExecuteRunRecordsSignal(t *testing.T) {
	root, err := campaignVariantRoot(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	rec, err := executeRun(context.Background(), CampaignOptions{
		Executable: "/bin/sh",
		Args:       []string{"-c", "kill -TERM $$", "signal-test"},
	}, 1, ".version 8.7\n", false, "signal-test", time.Second, root)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Signal != "terminated" {
		t.Fatalf("signal = %q, want terminated; record %+v", rec.Signal, rec)
	}
	if !rec.Exited || rec.ExitCode != -1 {
		t.Fatalf("signal exit status = exited %t code %d, want true and -1", rec.Exited, rec.ExitCode)
	}
}
