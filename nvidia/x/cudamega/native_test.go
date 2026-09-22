package cudamega_test

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// megaTarget returns the architecture the program fixture is compiled for.
// CUDA_MEGA_NATIVE_ARCH names another; sm_121a is the arch-specific target
// the coordinator authorized and the one the goldens under testdata were
// generated for. sm_121 is the family, not this target, and ptxas rejects
// sm_121a before CUDA 12.9.
func megaTarget() string {
	if target := os.Getenv("CUDA_MEGA_NATIVE_ARCH"); target != "" {
		return target
	}
	return goldenArch
}

// nativeMain emits a host harness for one fixture. It is generated rather
// than written once because the two authorized fixtures differ in task count,
// grid and thread width, and a harness hard-coded to the first of them would
// silently pass the second by checking the wrong words.
//
// Every failure has its own exit status so that a nonzero exit says what went
// wrong. Status 9 in particular is the abort buffer coming back nonzero: the
// in-kernel poll bound raises [cudamega.AbortSpinExhausted] there, and a run
// that hit the bound must never be read as a run that finished. Statuses 10-12
// belong to the epoch-2³² launch, which is what makes the 64-bit epoch
// parameter observable: a 32-bit parameter takes that epoch to zero and the
// run comes back with [cudamega.AbortBadExtent] as status 11.
func nativeMain(f programFixture) string {
	tasks := len(f.graph.Tasks())
	return fmt.Sprintf(`
#include <cstdio>
int main() {
  const unsigned n = 32, nt = %[1]du, threads = %[2]du;
  const size_t words = (size_t)nt*n, sb = words*sizeof(unsigned);
  unsigned h[n]; for (unsigned i=0;i<n;i++) h[i]=i;
  unsigned *in=0,*sp=0,*so=0,*completed=0,*abort=0;
  if (cudaMalloc(&in,sizeof h)||cudaMalloc(&sp,sb)||cudaMalloc(&so,sb)||
      cudaMalloc(&completed,nt*sizeof(unsigned))||cudaMalloc(&abort,sizeof(unsigned))) return 2;
  cudaMemcpy(in,h,sizeof h,cudaMemcpyHostToDevice);
  cudaMemset(sp,0,sb); cudaMemset(so,0,sb);
  cudaMemset(completed,0,nt*sizeof(unsigned)); cudaMemset(abort,0,sizeof(unsigned));
  int blocks=0, sms=0, dev=0;
  if (cudaGetDevice(&dev)!=cudaSuccess||cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev)!=cudaSuccess||sms<1) return 3;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,dag_worker,threads,0)!=cudaSuccess||blocks<1) return 3;
  // Residency is blocks-per-SM times SM count. Comparing the per-SM figure
  // against the whole grid is a different quantity: it refuses every grid
  // wider than one SM's residency, and on these one- and two-CTA fixtures it
  // passes without ever reaching the admission it claims to check.
  if ((unsigned long long)blocks*(unsigned long long)sms < grid_ctas) {
    printf("residency %%d/SM * %%d SMs < grid_ctas %%u\n", blocks, sms, grid_ctas); return 3; }
  dag_worker<<<grid_ctas,threads>>>(in,sp,completed,abort,n,1ull);
  if (cudaDeviceSynchronize()!=cudaSuccess) { printf("persistent: %%s\n", cudaGetErrorString(cudaGetLastError())); return 4; }
  unsigned ac=0; cudaMemcpy(&ac,abort,sizeof ac,cudaMemcpyDeviceToHost);
  if (ac) { printf("persistent run aborted with code %%u\n", ac); return 9; }
  for (unsigned t=0;t<nt;t++) ordinary_task<<<1,threads>>>(in,so,n,t);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 5;
  unsigned *a=(unsigned*)malloc(sb), *b=(unsigned*)malloc(sb);
  cudaMemcpy(a,sp,sb,cudaMemcpyDeviceToHost); cudaMemcpy(b,so,sb,cudaMemcpyDeviceToHost);
  for (size_t i=0;i<words;i++) if (a[i]!=b[i]) {
    printf("task %%zu element %%zu: persistent %%u ordinary %%u\n", i/n, i%%n, a[i], b[i]); return 6; }
  cudaMemset(sp,0xa5,sb); cudaMemset(completed,0,nt*sizeof(unsigned));
  unsigned one=1; cudaMemcpy(abort,&one,sizeof one,cudaMemcpyHostToDevice);
  dag_worker<<<grid_ctas,threads>>>(in,sp,completed,abort,n,2ull);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 7;
  cudaMemcpy(a,sp,sb,cudaMemcpyDeviceToHost);
  for (size_t i=0;i<words;i++) if (a[i]!=0xa5a5a5a5u) {
    printf("aborted launch wrote task %%zu element %%zu\n", i/n, i%%n); return 8; }
  // Epoch 2^32 exactly. 2^32+1 is the value that looks like the test and is
  // not: it truncates to 1, which a 32-bit parameter accepts, so a run at
  // 2^32+1 passes whether the parameter is 32 or 64 bits wide. 2^32 truncates
  // to 0, the unpublished value the worker refuses with AbortBadExtent, and
  // is therefore the smallest epoch that tells the two widths apart.
  cudaMemset(sp,0,sb); cudaMemset(completed,0,nt*sizeof(unsigned)); cudaMemset(abort,0,sizeof(unsigned));
  dag_worker<<<grid_ctas,threads>>>(in,sp,completed,abort,n,1ull<<32);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 10;
  cudaMemcpy(&ac,abort,sizeof ac,cudaMemcpyDeviceToHost);
  if (ac) { printf("launch at epoch 2^32 aborted with code %%u\n", ac); return 11; }
  cudaMemcpy(a,sp,sb,cudaMemcpyDeviceToHost);
  for (size_t i=0;i<words;i++) if (a[i]!=b[i]) {
    printf("epoch 2^32: task %%zu element %%zu: persistent %%u ordinary %%u\n", i/n, i%%n, a[i], b[i]); return 12; }
  printf("OK n=%%u tasks=%%u ctas=%%u threads=%%u\n", n, nt, grid_ctas, threads);
  free(a); free(b);
  cudaFree(in);cudaFree(sp);cudaFree(so);cudaFree(completed);cudaFree(abort);
  return 0;
}
`, tasks, f.grid.ThreadsPerCTA)
}

// TestNativePersistentReference launches the persistent grid on the authorized
// device and requires it to agree with the ordinary-launch reference.
//
// The fixtures run in the order cta-1cta, device-2cta: a one-CTA grid that
// hangs is the cheapest version of the failure, so the cheap one goes first
// and a hang there stops the rest.
//
// The context timeout below is a harness convenience for a host process that
// will not exit, not a recovery mechanism. It cannot stop a resident CTA that
// will not yield; the poll bound compiled into the kernel is what does that.
func TestNativePersistentReference(t *testing.T) {
	if os.Getenv("CUDA_MEGA_NATIVE") != "1" {
		t.Skip("native qualification opt-in: CUDA_MEGA_NATIVE=1")
	}
	// See cudashared/native_test.go: the lock exports the holder's pid, not "1".
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDA_MEGA_NATIVE=1 requires running under the GPU lock")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := megaTarget()
	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudamega-native-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	// Subtests, not a loop body, so the cheap fixture can be run and banked
	// on its own: -run '/cta-1cta' launches the one-CTA grid and nothing
	// else. Ordering here also means a hang on cta-1cta never reaches
	// device-2cta.
	for _, f := range []programFixture{ctaFixture(t), deviceFixture(t)} {
		t.Run(f.name, func(t *testing.T) {
			src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid)
			if err != nil {
				t.Fatal(err)
			}
			cu := filepath.Join(dir, f.name+".cu")
			exe := filepath.Join(dir, f.name)
			if err := os.WriteFile(cu, []byte(src+nativeMain(f)), 0600); err != nil {
				t.Fatal(err)
			}
			if out, err := exec.Command(nvcc, "-arch="+target, cu, "-o", exe).CombinedOutput(); err != nil {
				t.Fatalf("compile: %v\n%s", err, out)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()
			out, err := exec.CommandContext(ctx, exe).CombinedOutput()
			if err != nil {
				t.Fatalf("%v\n%s", err, out)
			}
			t.Logf("%s on %s: %s", f.name, target, out)
		})
	}
}

func TestCompileProgramSource(t *testing.T) {
	if os.Getenv("CUDA_MEGA_COMPILE") != "1" {
		t.Skip("compile qualification opt-in: CUDA_MEGA_COMPILE=1")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := megaTarget()
	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudamega-compile-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	for _, f := range []programFixture{ctaFixture(t), deviceFixture(t)} {
		src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid)
		if err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}
		cu := filepath.Join(dir, f.name+".cu")
		if err := os.WriteFile(cu, []byte(src), 0600); err != nil {
			t.Fatal(err)
		}
		if out, err := exec.Command(nvcc, "-arch="+target, "-c", cu, "-o", filepath.Join(dir, f.name+".o")).CombinedOutput(); err != nil {
			t.Fatalf("%s: compile: %v\n%s", f.name, err, out)
		}
	}
}

// TestNativeSpinBoundReleasesAGridThatWouldHang launches a grid that is
// guaranteed to deadlock without the bound, and requires it to come back with
// AbortSpinExhausted instead of never coming back.
//
// The fixture is deliberately broken: the producer's publication is deleted
// from the generated source, so its completion word never reaches grid_ctas
// and the consumer's wait can only end at the bound. That is the whole point.
// Nothing else in this package tests the bound against a real hang -- the
// correct fixtures never spin long enough to reach it -- so without this the
// bound is installed and unexercised.
//
// It carries its own opt-in beyond CUDA_MEGA_NATIVE because it is the one
// launch here whose failure mode is a resident grid on a device that drives a
// display. Run it last, over ssh, with a second session already open.
func TestNativeSpinBoundReleasesAGridThatWouldHang(t *testing.T) {
	if os.Getenv("CUDA_MEGA_SPINCAP") != "1" {
		t.Skip("deliberate-deadlock probe opt-in: CUDA_MEGA_SPINCAP=1")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDA_MEGA_SPINCAP=1 requires running under the GPU lock")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := megaTarget()

	f := ctaFixture(t)
	f.grid.SpinLimit = 1 << 20 // reached in well under a second; the default is not a deadline
	src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid)
	if err != nil {
		t.Fatal(err)
	}
	const publish = "    if (threadIdx.x == 0) atomicAdd(completed + 0, 1u); // publish task 0\n"
	if !strings.Contains(src, publish) {
		t.Fatalf("producer publication not found in generated source; the probe would test nothing:\n%s", src)
	}
	src = strings.Replace(src, publish, "    // publication deleted: the consumer below can only exit at the bound\n", 1)

	src += fmt.Sprintf(`
#include <cstdio>
int main() {
  const unsigned n = 32, nt = 2u, threads = %du;
  unsigned h[n]; for (unsigned i=0;i<n;i++) h[i]=i;
  unsigned *in=0,*sp=0,*completed=0,*abort=0;
  if (cudaMalloc(&in,sizeof h)||cudaMalloc(&sp,(size_t)nt*n*sizeof(unsigned))||
      cudaMalloc(&completed,nt*sizeof(unsigned))||cudaMalloc(&abort,sizeof(unsigned))) return 2;
  cudaMemcpy(in,h,sizeof h,cudaMemcpyHostToDevice);
  cudaMemset(sp,0,(size_t)nt*n*sizeof(unsigned));
  cudaMemset(completed,0,nt*sizeof(unsigned)); cudaMemset(abort,0,sizeof(unsigned));
  dag_worker<<<grid_ctas,threads>>>(in,sp,completed,abort,n,1ull);
  if (cudaDeviceSynchronize()!=cudaSuccess) { printf("sync: %%s\n", cudaGetErrorString(cudaGetLastError())); return 4; }
  unsigned ac=0; cudaMemcpy(&ac,abort,sizeof ac,cudaMemcpyDeviceToHost);
  printf("abort_code=%%u\n", ac);
  if (ac != %du) return 9;
  cudaFree(in);cudaFree(sp);cudaFree(completed);cudaFree(abort);
  return 0;
}
`, f.grid.ThreadsPerCTA, cudamega.AbortSpinExhausted)

	home, _ := os.UserHomeDir()
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudamega-spincap-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	cu, exe := filepath.Join(dir, "spincap.cu"), filepath.Join(dir, "spincap")
	if err := os.WriteFile(cu, []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(nvcc, "-arch="+target, cu, "-o", exe).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, exe).CombinedOutput()
	if err != nil {
		t.Fatalf("spin-bound probe: %v\n%s", err, out)
	}
	t.Logf("spin-bound probe on %s: %s", target, out)
}
