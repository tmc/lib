package cupticheckpoint

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
)

var (
	libOnce sync.Once
	libErr  error

	fnCheckpointSave    func(*cupti.Checkpoint) int32
	fnCheckpointRestore func(*cupti.Checkpoint) int32
	fnCheckpointFree    func(*cupti.Checkpoint) int32
)

func candidatePaths() []string {
	paths := []string{
		"libcheckpoint.so",
		"libcheckpoint.so.1",
	}
	if cudaPath := os.Getenv("CUDA_PATH"); cudaPath != "" {
		paths = append(paths,
			filepath.Join(cudaPath, "lib64", "libcheckpoint.so"),
			filepath.Join(cudaPath, "targets", "sbsa-linux", "lib", "libcheckpoint.so"),
			filepath.Join(cudaPath, "targets", "x86_64-linux", "lib", "libcheckpoint.so"),
		)
	}
	for _, base := range []string{"/usr/local/cuda", "/usr/local/cuda-13.4", "/usr/local/cuda-13.0", "/usr/local/cuda-12.8", "/usr/local/cuda-12.6"} {
		paths = append(paths,
			filepath.Join(base, "targets", "sbsa-linux", "lib", "libcheckpoint.so"),
			filepath.Join(base, "targets", "x86_64-linux", "lib", "libcheckpoint.so"),
			filepath.Join(base, "lib64", "libcheckpoint.so"),
		)
	}
	return paths
}

func loadLib() error {
	libOnce.Do(func() {
		if runtime.GOOS != "linux" {
			libErr = fmt.Errorf("cupticheckpoint: native checkpoint requires Linux")
			return
		}
		var handle uintptr
		var lastErr error
		for _, path := range candidatePaths() {
			h, err := openDynamicLibrary(path)
			if err == nil && h != 0 {
				handle = h
				break
			}
			lastErr = err
		}
		if handle == 0 {
			libErr = fmt.Errorf("cupticheckpoint: load libcheckpoint.so: %w", lastErr)
			return
		}

		bind := func(fn interface{}, name string) {
			ptr, err := lookupLibrarySymbol(handle, name)
			if err != nil || ptr == 0 {
				libErr = fmt.Errorf("cupticheckpoint: symbol %s: %w", name, err)
				return
			}
			purego.RegisterFunc(fn, ptr)
		}

		bind(&fnCheckpointSave, "cuptiCheckpointSave")
		bind(&fnCheckpointRestore, "cuptiCheckpointRestore")
		bind(&fnCheckpointFree, "cuptiCheckpointFree")
	})
	return libErr
}

func toError(res int32) error {
	if res == 0 {
		return nil
	}
	return fmt.Errorf("cupticheckpoint: CUPTI result %d", res)
}
