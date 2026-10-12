package ncclinspector_test

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func ExampleFileDecoder_Next_truncated() {
	b, err := os.ReadFile("testdata/stats.jsonl")
	if err != nil {
		panic(err)
	}
	dir, err := os.MkdirTemp("", "inspector")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "capture.jsonl")
	if err = os.WriteFile(path, b, 0600); err != nil {
		panic(err)
	}
	d, err := ncclinspector.OpenFile(path, ncclinspector.Limits{}, nil)
	if err != nil {
		panic(err)
	}
	defer d.Close()
	if _, err = d.Next(); err != nil {
		panic(err)
	}
	// A rotation that truncates the file in place is not more data to wait for.
	if err = os.Truncate(path, 0); err != nil {
		panic(err)
	}
	_, err = d.Next()
	fmt.Println(errors.Is(err, ncclinspector.ErrFileTruncated), errors.Is(err, io.EOF))
	// Output: true false
}

func ExampleFileDecoder_Next_replaced() {
	b, err := os.ReadFile("testdata/stats.jsonl")
	if err != nil {
		panic(err)
	}
	dir, err := os.MkdirTemp("", "inspector")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "capture.jsonl")
	if err = os.WriteFile(path, b, 0600); err != nil {
		panic(err)
	}
	d, err := ncclinspector.OpenFile(path, ncclinspector.Limits{}, nil)
	if err != nil {
		panic(err)
	}
	defer d.Close()
	if _, err = d.Next(); err != nil {
		panic(err)
	}
	// A rotation that renames the file and creates a new one starts a new epoch.
	if err = os.Rename(path, path+".1"); err != nil {
		panic(err)
	}
	if err = os.WriteFile(path, b, 0600); err != nil {
		panic(err)
	}
	_, err = d.Next()
	fmt.Println(errors.Is(err, ncclinspector.ErrFileReplaced), errors.Is(err, io.EOF))
	// Output: true false
}
