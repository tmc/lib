package cudaschedule

import (
	"context"
	"os"
	"testing"
	"time"
)

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
