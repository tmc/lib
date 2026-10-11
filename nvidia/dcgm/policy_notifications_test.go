package dcgm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/ebitengine/purego"
)

const notificationStub = `
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
typedef struct { uint32_t version; int32_t condition; uint64_t val[2]; uint32_t gpu; } response;
typedef struct { uint32_t version; int32_t condition,mode,isolation,action,validation,response; uint64_t parms[14]; } policy;
_Static_assert(sizeof(response)==32 && offsetof(response,gpu)==24,"response ABI");
_Static_assert(sizeof(policy)==144,"policy ABI");
typedef int (*callback)(response*,uint64_t);
int dcgmPolicyGet(uintptr_t h,uintptr_t g,int32_t count,policy *p,uintptr_t status) {
 if(count<1||count>32)return -1;
 for(int i=0;i<count;i++){if(p[i].version!=0x1000090)return -12;p[i].condition=i+1;}
 if(h==55)p[0].version=0;
 return 0;
}
static callback cb; static uint64_t token; static int busy;
int dcgmPolicyRegister_v2(uintptr_t h,uintptr_t g,int32_t c,callback f,uint64_t u) {
 cb=f; token=u; busy=(h==2); if(h==99) return -1; return 0;
}
int dcgmPolicyUnregister(uintptr_t h,uintptr_t g,int32_t c) {
 if(busy) { busy=0; return -34; } return 0;
}
int fixture_short(void) {
 long size=sysconf(_SC_PAGESIZE);
 char *p=mmap(0,size*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 if(p==MAP_FAILED)return -1;
 mprotect(p+size,size,PROT_NONE);
 uint32_t *header=(uint32_t*)(p+size-4);*header=0x1000018;
 cb((response*)header,token);munmap(p,size*2);return 0;
}
static void *emit(void *unused) {
 response r={0x2000020,1,{17,23},7};
 for(int i=0;i<1000;i++) if(cb) cb(&r,token);
 return 0;
}
int fixture_emit(uintptr_t h) {
 if(h==3) { pthread_t threads[4]; for(int i=0;i<4;i++) pthread_create(&threads[i],0,emit,0);
 for(int i=0;i<4;i++) pthread_join(threads[i],0); }
 else { response r={0x2000020,1,{17,23},7}; if(cb) cb(&r,token); }
 return 0;
}
`

func TestPolicyNotificationsStub(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("host shared-library fixture uses a Unix C compiler")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("C compiler unavailable")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "policy.c")
	if err := os.WriteFile(source, []byte(notificationStub), 0600); err != nil {
		t.Fatal(err)
	}
	library := filepath.Join(dir, "policy.so")
	args := []string{"-shared", "-fPIC", "-pthread", "-o", library, source}
	if runtime.GOOS == "darwin" {
		args[0] = "-dynamiclib"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if output, err := exec.CommandContext(ctx, cc, args...).CombinedOutput(); err != nil {
		t.Fatalf("compile stub: %v\n%s", err, output)
	}
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestPolicyNotificationsStubChild$", "-test.count=1")
	cmd.Env = append(os.Environ(), "DCGM_NOTIFICATIONS_STUB=1", "DCGM_LIB_PATH="+library)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("stub child: %v\n%s", err, output)
	}
}

func TestPolicyNotificationsStubChild(t *testing.T) {
	if os.Getenv("DCGM_NOTIFICATIONS_STUB") != "1" {
		t.Skip("fixture subprocess")
	}
	if err := Load(os.Getenv("DCGM_LIB_PATH")); err != nil {
		t.Fatal(err)
	}
	policies, err := ReadPolicies(1, 1, 32, 0)
	if err != nil || len(policies) != 32 || policies[31].Condition != 32 {
		t.Fatal(policies, err)
	}
	if _, err := ReadPolicies(1, 1, 33, 0); err == nil {
		t.Fatal("unbounded query")
	}
	if _, err := ReadPolicies(55, 1, 1, 0); err == nil {
		t.Fatal("response layout contradiction")
	}
	var emit func(Handle) Result
	purego.RegisterLibFunc(&emit, lib, "fixture_emit")
	failed, err := NewPolicyNotifications(99, 1, PolicyCondDbe, 1)
	if err == nil || failed == nil {
		t.Fatal("failed native registration lost owner")
	}
	emit(1)
	if r := <-failed.Notifications(); r.GpuId != 7 {
		t.Fatal("registration side effects lost", r)
	}
	if err := failed.Close(); err != nil {
		t.Fatal(err)
	}
	s, err := NewPolicyNotifications(2, 1, PolicyCondDbe, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewPolicyNotifications(2, 1, PolicyCondDbe, 1); err == nil {
		t.Fatal("overlap accepted")
	}
	emit(1)
	r := <-s.Notifications()
	if r.Version != 0x2000020 || r.GpuId != 7 || r.Val != [2]uint64{17, 23} {
		t.Fatal(r)
	}
	var short func() int32
	purego.RegisterLibFunc(&short, lib, "fixture_short")
	if short() != 0 {
		t.Fatal("guard fixture failed")
	}
	if s.Dropped() != 1 {
		t.Fatal("short unknown layout not refused")
	}
	emit(3)
	if s.Dropped() == 0 {
		t.Fatal("loss not counted")
	}
	if err := s.Close(); !errors.Is(err, ErrInUse) {
		t.Fatal(err)
	}
	emit(1)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	emit(3)
	for range s.Notifications() {
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	if err := Load(os.Getenv("DCGM_LIB_PATH")); err != nil {
		t.Fatal(err)
	}
	purego.RegisterLibFunc(&emit, lib, "fixture_emit")
	s, err = NewPolicyNotifications(1, 1, PolicyCondDbe, 1)
	if err != nil {
		t.Fatal(err)
	}
	emit(1)
	if r := <-s.Notifications(); r.GpuId != 7 {
		t.Fatal(r)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func ExampleNewPolicyNotifications() {
	_, err := NewPolicyNotifications(0, 0, 0, 0)
	fmt.Println(err != nil)
	// Output: true
}

func ExamplePolicyNotifications_Dropped() {
	var subscription *PolicyNotifications
	fmt.Println(subscription.Dropped())
	// Output: 0
}

func ExamplePolicyNotifications_Notifications() {
	var subscription *PolicyNotifications
	fmt.Println(subscription.Notifications() == nil)
	// Output: true
}

func ExamplePolicyNotifications_Close() {
	var subscription *PolicyNotifications
	fmt.Println(subscription.Close() != nil)
	// Output: true
}

func ExampleReadPolicies() {
	_, err := ReadPolicies(0, 0, 0, 0)
	fmt.Println(err != nil)
	// Output: true
}
