package cudahotswap

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

func ExampleWatchPaths() {
	p := WatchPaths{SourceFile: "kernel.cu", Directories: []string{"include"}}
	fmt.Println(p.SourceFile, len(p.Directories))
	// Output: kernel.cu 1
}

func ExampleWatchConfig() {
	c := WatchConfig{Poll: true}
	fmt.Println(c.Poll, c.Debounce)
	// Output: true 0s
}

func ExampleService_SetWatchPaths() {
	s := settingsExampleService() // Explicit host loader and lease.
	defer s.Close(context.Background())
	dir, err := os.MkdirTemp("", "hotswap-example-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "kernel.cu")
	if err := os.WriteFile(path, []byte("source"), 0600); err != nil {
		fmt.Println(err)
		return
	}
	state, err := s.SetWatchPaths("kernel", &WatchPaths{SourceFile: path}, "watch")
	fmt.Println(state.Generation, state.ConfigRevision, err)
	// Output: 1 1 <nil>
}

func ExampleService_WatchPaths() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	p, err := s.WatchPaths("kernel")
	fmt.Println(p == nil, err)
	// Output: true <nil>
}

func ExampleService_Watch() {
	s := settingsExampleService()
	defer s.Close(context.Background())
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	fmt.Println(s.Watch(ctx, WatchConfig{Poll: true}, nil))
	// Output: context canceled
}
