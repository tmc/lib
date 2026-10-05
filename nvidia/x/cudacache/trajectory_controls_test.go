package cudacache

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestTrajectoryQuarantineRetainsOversizeAndFailure(t *testing.T) {
	for _, failed := range []bool{false, true} {
		name := "retained"
		if failed {
			name = "quarantine-error"
		}
		t.Run(name, func(t *testing.T) {
			c, err := New(Config{RootDir: t.TempDir()})
			if err != nil {
				t.Fatal(err)
			}
			r := cachedTrajectoryFixture()
			ctx := context.Background()
			if err := c.PutTrustedTrajectory(ctx, r); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(c.receiptsDir, "trajectories", r.SHA256+".json")
			data := make([]byte, 128<<10)
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			if failed {
				c.quarantineDir = filepath.Join(c.rootDir, "not-a-directory")
				if err := os.WriteFile(c.quarantineDir, []byte("keep"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := c.GetTrajectory(ctx, r.SHA256); !errors.Is(err, ErrCorruptReceipt) {
				t.Fatal(err)
			}
			if failed {
				got, err := os.ReadFile(path)
				if err != nil || len(got) != len(data) {
					t.Fatal("failed quarantine lost original", err, len(got))
				}
				return
			}
			entries, err := os.ReadDir(c.quarantineDir)
			if err != nil || len(entries) != 1 {
				t.Fatal(err, entries)
			}
			got, err := os.ReadFile(filepath.Join(c.quarantineDir, entries[0].Name()))
			if err != nil || len(got) != len(data) {
				t.Fatal("over-limit original truncated", err, len(got))
			}
		})
	}
}

func TestTrajectoryZeroCacheAndSymlinkRefusal(t *testing.T) {
	var zero Cache
	r := cachedTrajectoryFixture()
	ctx := context.Background()
	if err := zero.PutTrustedTrajectory(ctx, r); err == nil {
		t.Fatal("zero cache published")
	}
	if _, err := zero.GetTrajectory(ctx, r.SHA256); err == nil {
		t.Fatal("zero cache read cwd")
	}
	c, err := New(Config{RootDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.PutTrustedTrajectory(ctx, r); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(c.receiptsDir, "trajectories", r.SHA256+".json")
	outside := filepath.Join(t.TempDir(), "outside.json")
	if err := os.Rename(path, outside); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, path); err != nil {
		t.Fatal(err)
	}
	if _, err := c.GetTrajectory(ctx, r.SHA256); !errors.Is(err, ErrCorruptReceipt) {
		t.Fatal("symlink admitted", err)
	}
	if _, err := os.Stat(outside); err != nil {
		t.Fatal("outside file damaged", err)
	}
}
