package cudaflightrecorder

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestDiskRetentionRejectsEscapingNames(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "captures")
	manager, err := NewSnapshotDiskManager(dir, 100)
	if err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside")
	if err := os.WriteFile(outside, []byte("preserve"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"../outside", outside, "nested/file", "nested\\file", ".", "..", ""} {
		if _, err := manager.SaveSnapshot(name, []byte("corrupt")); err == nil {
			t.Fatalf("accepted %q", name)
		}
	}
	data, err := os.ReadFile(outside)
	if err != nil || string(data) != "preserve" {
		t.Fatalf("outside changed: %q %v", data, err)
	}
}

func TestDiskRetentionDoesNotPruneUnownedFiles(t *testing.T) {
	dir := t.TempDir()
	sentinel := filepath.Join(dir, "unrelated")
	if err := os.WriteFile(sentinel, []byte("preserve"), 0600); err != nil {
		t.Fatal(err)
	}
	manager, err := NewSnapshotDiskManager(dir, 10)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.SaveSnapshot("capture.tar", []byte("more")); !errors.Is(err, ErrDiskBudgetExceeded) {
		t.Fatalf("err=%v", err)
	}
	data, err := os.ReadFile(sentinel)
	if err != nil || string(data) != "preserve" {
		t.Fatalf("unowned file changed: %q %v", data, err)
	}
}

func TestDiskRetentionProtectsExistingDestination(t *testing.T) {
	dir := t.TempDir()
	manager, err := NewSnapshotDiskManager(dir, 100)
	if err != nil {
		t.Fatal(err)
	}
	path, err := manager.SaveSnapshot("capture.tar", []byte("original"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.SaveSnapshot("capture.tar", []byte("replacement")); err == nil {
		t.Fatal("existing destination overwritten")
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "original" {
		t.Fatalf("existing file changed: %q %v", data, err)
	}
}

func TestDiskRetentionDoesNotPruneReplacedFile(t *testing.T) {
	dir := t.TempDir()
	manager, err := NewSnapshotDiskManager(dir, 10)
	if err != nil {
		t.Fatal(err)
	}
	path, err := manager.SaveSnapshot("capture.tar", []byte("original"))
	if err != nil {
		t.Fatal(err)
	}
	replacement := filepath.Join(t.TempDir(), "replacement")
	if err := os.WriteFile(replacement, []byte("unowned!"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.SaveSnapshot("new.tar", []byte("new")); !errors.Is(err, ErrDiskBudgetExceeded) {
		t.Fatalf("err=%v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "unowned!" {
		t.Fatalf("replacement changed: %q %v", data, err)
	}
}

func TestDiskRetentionRejectsSymlink(t *testing.T) {
	dir := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.WriteFile(outside, []byte("preserve"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "capture.tar")); err != nil {
		t.Skip(err)
	}
	manager, err := NewSnapshotDiskManager(dir, 100)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.SaveSnapshot("capture.tar", []byte("corrupt")); err == nil {
		t.Fatal("symlink followed")
	}
	if _, err := manager.SaveSnapshot("different.tar", []byte("data")); err == nil {
		t.Fatal("nonregular directory entry admitted")
	}
	data, err := os.ReadFile(outside)
	if err != nil || string(data) != "preserve" {
		t.Fatalf("outside changed: %q %v", data, err)
	}
}
