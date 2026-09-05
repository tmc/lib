package nvrtcx

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestCacheProcessPublication(t *testing.T) {
	if dir := os.Getenv("NVRTC_CACHE_WRITER_DIR"); dir != "" {
		cacheDirOverride = dir
		marker := os.Getenv("NVRTC_CACHE_WRITER_MARKER")
		for i := 0; i < 40; i++ {
			value := fmt.Sprintf("%s-%d", marker, i)
			cm := &CompiledModule{output: OutputPTXTileIR, ptx: value, tileir: []byte(value), log: value, lowered: map[string]string{"kernel": value}}
			if err := storeCache("shared", cm); err != nil {
				t.Fatal(err)
			}
		}
		return
	}
	dir := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	type child struct {
		cmd    *exec.Cmd
		output bytes.Buffer
	}
	children := make([]child, 2)
	done := make(chan error, len(children))
	for i := range children {
		c := &children[i]
		c.cmd = exec.CommandContext(ctx, exe, "-test.run=^TestCacheProcessPublication$")
		c.cmd.Env = append(os.Environ(), "NVRTC_CACHE_WRITER_DIR="+dir, fmt.Sprintf("NVRTC_CACHE_WRITER_MARKER=writer%d", i))
		c.cmd.Stdout, c.cmd.Stderr = &c.output, &c.output
		if err := c.cmd.Start(); err != nil {
			cancel()
			t.Fatal(err)
		}
		go func() { done <- c.cmd.Wait() }()
	}
	reads, completed := 0, 0
	for completed < len(children) {
		select {
		case err := <-done:
			completed++
			if err != nil {
				t.Errorf("writer: %v", err)
			}
		default:
		}
		data, err := os.ReadFile(filepath.Join(dir, "shared.cache"))
		if os.IsNotExist(err) {
			time.Sleep(time.Millisecond)
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		cm, err := decodeCacheEntry("shared", data)
		if err != nil {
			t.Fatal(err)
		}
		if cm.ptx != string(cm.tileir) || cm.ptx != cm.log || cm.ptx != cm.lowered["kernel"] {
			t.Fatalf("mixed publication: %#v", cm)
		}
		reads++
	}
	for i := range children {
		if t.Failed() {
			t.Log(children[i].output.String())
		}
	}
	if reads == 0 {
		t.Fatal("no cache records observed")
	}
	t.Logf("two processes published 80 records; validated %d reads", reads)
}
