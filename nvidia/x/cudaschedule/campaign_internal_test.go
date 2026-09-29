package cudaschedule

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"
)

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
	root, err := campaignVariantRoot()
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(root)
	rec, err := executeRun(context.Background(), CampaignOptions{
		Executable: "/bin/sh",
		Args:       []string{"-c", "kill -TERM $$", "signal-test"},
	}, 1, ".version 8.7\n", false, time.Second, root)
	if err != nil {
		t.Fatal(err)
	}
	if rec.Signal != "terminated" {
		t.Fatalf("signal = %q, want terminated; record %+v", rec.Signal, rec)
	}
}
