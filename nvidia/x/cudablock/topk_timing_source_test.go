package cudablock

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This executes the actual candidate source with 128 host threads and a
// reusable barrier. It exercises shared reductions, not GPU scheduling.
func TestTopKParallelSourceEmulation(t *testing.T) {
	compiler, err := exec.LookPath("clang++")
	if err != nil {
		t.Skip("parallel source emulation requires clang++")
	}
	shim := strings.Replace(topkHostShim, "struct { unsigned int x; } blockIdx, threadIdx;", "struct { unsigned int x; } blockIdx;\nstruct ThreadIndex { unsigned int x; };\nthread_local ThreadIndex threadIdx;", 1)
	shim += `
#include <thread>
#include <mutex>
#include <condition_variable>
#define __shared__ static
std::mutex barrier_mu;
std::condition_variable barrier_cv;
int barrier_count=0, barrier_generation=0;
void __syncthreads() {
    std::unique_lock<std::mutex> lock(barrier_mu);
    int generation=barrier_generation;
    if (++barrier_count==128) {
        barrier_count=0; ++barrier_generation; barrier_cv.notify_all();
    } else {
        barrier_cv.wait(lock,[&] { return generation!=barrier_generation; });
    }
}
`
	main := strings.Replace(topkHostMain, "topk_fp32(in.data(),lengths.data(),offsets.data(),indices.data(),out_values.data(),counts.data(),status.data(),stride,width,k,rows,values);", `std::vector<std::thread> threads;
        for (int tid=0;tid<128;++tid) threads.emplace_back([&,tid] {
            threadIdx.x=tid;
            topk_fp32(in.data(),lengths.data(),offsets.data(),indices.data(),out_values.data(),counts.data(),status.data(),stride,width,k,rows,values);
        });
        for (auto& thread:threads) thread.join();`, 1)
	dir := t.TempDir()
	path, bin := filepath.Join(dir, "parallel.cc"), filepath.Join(dir, "parallel-host")
	if err := os.WriteFile(path, []byte(shim+topkParallelSource+main), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-pthread", "-O0", path, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("parallel host build: %v\n%s", err, out)
	}
	for _, f := range topkFixtures() {
		t.Run(f.name, func(t *testing.T) {
			if err := topkCompare(f.c, topkRunHost(t, bin, f), topkSortReference(f)); err != nil {
				t.Fatal(err)
			}
		})
	}
}
