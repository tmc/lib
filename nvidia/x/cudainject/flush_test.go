package cudainject

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	assets "github.com/tmc/lib/nvidia/internal/cudainject"
)

// This runs the unchanged worker and stop function with a CPU CUPTI-call
// counter. It proves worker sequencing, not native provider delivery.
func TestExistingPeriodicFlushWorker(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("pthread worker")
	}
	compiler, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("C compiler unavailable")
	}
	source, err := assets.Sources.ReadFile("native/cupti_agent.c")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	start := strings.Index(text, "static void *flush_worker(void *arg) {")
	end := strings.Index(text, "static const char *cupti_library_path(void) {")
	if start < 0 || end <= start {
		t.Fatal("existing worker boundaries not found")
	}
	adapter := `#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <errno.h>
static long flush_interval_ms=2;
static int flush_stop,flush_started;
static pthread_t flusher;
static pthread_mutex_t flush_mu=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t flush_cond=PTHREAD_COND_INITIALIZER;
static _Atomic int shutdown_started,journal_ready,producer_failed,calls;
static int cuptiActivityFlushAll(int flags) { (void)flags;atomic_fetch_add(&calls,1);return 0; }
static void pause_ms(long ms) {struct timespec d={ms/1000,(ms%1000)*1000000L};nanosleep(&d,0);}
`
	adapter += text[start:end]
	adapter += `
int main(void) {
 if (pthread_create(&flusher,0,flush_worker,0)!=0) return 10;
 flush_started=1;
 pause_ms(20);
 if (atomic_load(&calls)!=0) return 11;
 atomic_store(&journal_ready,1);
 for(int i=0;i<100 && atomic_load(&calls)<2;i++) pause_ms(2);
 if (atomic_load(&calls)<2) return 12;
 atomic_store(&shutdown_started,1);
 pause_ms(10);
 int n=atomic_load(&calls);
 pause_ms(20);
 if(atomic_load(&calls)!=n) return 13;
 stop_flush_worker();
 pause_ms(10);
 if(atomic_load(&calls)!=n || flush_started || atomic_load(&producer_failed)) return 14;
 return 0;
}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "worker.c")
	binary := filepath.Join(dir, "worker")
	if err := os.WriteFile(path, []byte(adapter), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if b, err := exec.CommandContext(ctx, compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("worker compile: %v %s", err, b)
	}
	if b, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("worker sequencing: %v %s", err, b)
	}
}
