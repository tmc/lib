//go:build unix

package cudahotswap

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

// This host example blocks a fake acquisition, controls the clock, and reads
// the owner file from a scratch directory instead of cudalock.DefaultDir.
func ExampleService_Waiting() {
	s, err := NewService(&serviceLoader{kernel: &serviceKernel{raw: 1}})
	if err != nil {
		panic(err)
	}
	dir, err := os.MkdirTemp("", "cudahotswap-waiting")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	holder := cudalock.Owner{PID: 4242, Host: "gpuhost", Label: "other-job", At: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)}
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(holder.String()), 0o644); err != nil {
		panic(err)
	}
	clock := &waitClock{now: time.Date(2026, 1, 2, 5, 4, 5, 0, time.UTC)}
	s.waits.now = clock.Now
	s.waits.observe = func() (cudalock.Owner, string) { return observeOwner(dir) }
	entered, release := make(chan struct{}), make(chan struct{})
	s.rawAcquire = func(_ context.Context, name string) (io.Closer, error) {
		if name == "kernel" {
			close(entered)
			<-release
		}
		return io.NopCloser(strings.NewReader("")), nil
	}

	done := make(chan error, 1)
	go func() {
		_, err := s.Register(context.Background(), "kernel", Source{Code: "source", Entry: "entry"}, "original")
		done <- err
	}()
	<-entered
	clock.Advance(2 * time.Second)
	for _, w := range s.Waiting() {
		fmt.Println(w.Request, w.Name, w.Label, w.Waited, w.Holder.PID, w.Holder.Label, w.HolderUnavailable == "")
	}
	close(release)
	fmt.Println(<-done, len(s.Waiting()))
	fmt.Println(s.Close(context.Background()))
	// Output:
	// 1 kernel cudahotswap:kernel 2s 4242 other-job true
	// <nil> 0
	// <nil>
}
