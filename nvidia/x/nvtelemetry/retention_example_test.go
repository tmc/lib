package nvtelemetry_test

import (
	"bytes"
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
	"os"
	"path/filepath"
)

func ExampleOpenRetention() {
	_, err := nvtelemetry.OpenRetention("unused", nvtelemetry.RetentionConfig{})
	fmt.Println(err)
	// Output: invalid retention configuration
}
func ExampleRecoverRetention() {
	dir, err := os.MkdirTemp("", "retention-example-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	stream, err := os.ReadFile("testdata/canonical_v2.jsonl")
	if err != nil {
		fmt.Println(err)
		return
	}
	cfg := nvtelemetry.RetentionConfig{Owner: "example", MaxFiles: 1, MaxBytes: int64(len(stream)), MaxStreamBytes: int64(len(stream))}
	path := filepath.Join(dir, "owned")
	writer, err := nvtelemetry.OpenRetention(path, cfg)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("live recovery refused:", nvtelemetry.RecoverRetention(path, cfg.Owner) != nil)
	if err := writer.Close(); err != nil {
		fmt.Println(err)
		return
	}
	// Model one unpublished artifact left by an interrupted writer. The writer
	// above has actually stopped; recovery never infers death from a PID.
	pending := filepath.Join(path, "pending.jsonl")
	if err := os.WriteFile(pending, []byte("interrupted bytes"), 0600); err != nil {
		fmt.Println(err)
		return
	}
	if err := nvtelemetry.RecoverRetention(path, cfg.Owner); err != nil {
		fmt.Println(err)
		return
	}
	_, err = nvtelemetry.OpenRetention(path, cfg)
	fmt.Println("pending blocks reopen:", err != nil)
	// Preserve this single artifact outside the managed directory. The caller
	// owns the archive budget; retention does not create an accumulating archive.
	if err := os.Rename(pending, filepath.Join(dir, "preserved.jsonl")); err != nil {
		fmt.Println(err)
		return
	}
	writer, err = nvtelemetry.OpenRetention(path, cfg)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := writer.Store(1, bytes.NewReader(stream)); err != nil {
		writer.Close()
		fmt.Println(err)
		return
	}
	fmt.Println("stored:", writer.Status().Files)
	fmt.Println("closed:", writer.Close() == nil)
	// Output:
	// live recovery refused: true
	// pending blocks reopen: true
	// stored: 1
	// closed: true
}

func ExampleRetention_Store() {
	var retained nvtelemetry.Retention
	fmt.Println(retained.Store(1, nil))
	// Output: retention is closed
}

func ExampleRetention_Close() {
	var retained nvtelemetry.Retention
	fmt.Println(retained.Close())
	// Output: <nil>
}

func ExampleRetention_Status() {
	var retained nvtelemetry.Retention
	status := retained.Status()
	fmt.Println(status.Closed, status.Ready, status.CountsAvailable)
	// Output: true false false
}
