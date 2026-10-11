package dcgm

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
)

const policyStub = `
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
typedef struct { uint32_t version; int32_t condition; uint64_t val[2]; uint32_t gpu; } response;
typedef struct { uint32_t version; int32_t condition,mode,isolation,action,validation,response; uint64_t parms[14]; } policy;
_Static_assert(sizeof(response)==32 && offsetof(response,gpu)==24,"response ABI");
_Static_assert(sizeof(policy)==144,"policy ABI");
typedef int (*callback)(response*,uint64_t);
typedef struct { uint32_t version; int32_t condition; uint64_t val[2]; } response_v1;
_Static_assert(sizeof(response_v1)==24,"response v1 ABI");
typedef int (*callback_v1)(void*);
int dcgmPolicyRegister(uintptr_t h,uintptr_t g,int32_t c,callback_v1 begin,callback_v1 finish) {
 response_v1 r={0x1000018,2,{31,41}};
 if(begin(&r)!=11 || finish(&r)!=22) return -1;
 return 0;
}
static callback cb; static uint64_t token; static int busy;
int dcgmPolicyRegister_v2(uintptr_t h,uintptr_t g,int32_t c,callback f,uint64_t u) {
 if(h==99) return -1; cb=f; token=u; busy=(h==2); return 0;
}
int dcgmPolicyUnregister(uintptr_t h,uintptr_t g,int32_t c) {
 if(busy) { busy=0; return -34; } return 0;
}
static void *emit(void *unused) {
 response r={0x2000020,1,{17,23},7};
 for(int i=0;i<1000;i++) if(cb) cb(&r,token);
 return 0;
}
int dcgmPolicyTrigger(uintptr_t h) {
 if(h==3) { pthread_t threads[4]; for(int i=0;i<4;i++) pthread_create(&threads[i],0,emit,0);
 for(int i=0;i<4;i++) pthread_join(threads[i],0); }
 else { response r={0x2000020,1,{17,23},7}; if(cb) cb(&r,token); }
 return 0;
}
int dcgmShutdown(void) { response r={0x2000020,1,{17,23},7}; if(cb) cb(&r,token); return 0; }
int dcgmStopEmbedded(uintptr_t h) { return dcgmShutdown(); }
int dcgmDisconnect(uintptr_t h) { return dcgmShutdown(); }
int dcgmPolicyGet(uintptr_t h,uintptr_t g,int32_t count,policy *p,uintptr_t s) {
 if(count<1||count>32) return -1;
 for(int i=0;i<count;i++) { if(p[i].version!=0x1000090) return -12; p[i].condition=i+1; }
 return 0;
}
int dcgmPolicySet(uintptr_t h,uintptr_t g,policy *p,uintptr_t s) {
 return p && p->version==0x1000090 ? 0 : -12;
}
`

func TestPolicyStub(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("host shared-library fixture uses a Unix C compiler")
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("C compiler unavailable")
	}
	dir := t.TempDir()
	source := filepath.Join(dir, "policy.c")
	if err := os.WriteFile(source, []byte(policyStub), 0600); err != nil {
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
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestPolicyStubChild$", "-test.count=1")
	cmd.Env = append(os.Environ(), "DCGM_POLICY_STUB=1", "DCGM_LIB_PATH="+library)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("stub child: %v\n%s", err, output)
	}
}

func TestPolicyStubChild(t *testing.T) {
	if os.Getenv("DCGM_POLICY_STUB") != "1" {
		t.Skip("isolated stub process only")
	}
	var begin, finish PolicyCallbackResponseV1
	b := PolicyCallbackV1(purego.NewCallback(func(r *PolicyCallbackResponseV1) int32 { begin = *r; return 11 }))
	f := PolicyCallbackV1(purego.NewCallback(func(r *PolicyCallbackResponseV1) int32 { finish = *r; return 22 }))
	if err := PolicyRegisterV1(1, 1, PolicyCondPci, b, f); err != nil {
		t.Fatal(err)
	}
	if begin != finish || begin.Version != PolicyCallbackResponseVersion1 || begin.Condition != PolicyCondPci || begin.Val != [2]uint64{31, 41} {
		t.Fatalf("legacy callbacks: %+v %+v", begin, finish)
	}
	s, err := NewPolicyNotifications(1, 1, PolicyCondDbe, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewPolicyNotifications(1, 1, PolicyCondDbe, 1); err == nil {
		t.Fatal("overlapping registration admitted")
	}
	if err := PolicyTrigger(1); err != nil {
		t.Fatal(err)
	}
	if err := PolicyTrigger(1); err != nil {
		t.Fatal(err)
	}
	if s.Dropped() != 1 {
		t.Fatalf("dropped=%d want 1", s.Dropped())
	}
	r := <-s.Notifications()
	if r.Version != PolicyCallbackResponseVersion2 || r.Condition != PolicyCondDbe || r.GpuId != 7 || r.Val != [2]uint64{17, 23} {
		t.Fatalf("response=%+v", r)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := PolicyTrigger(1); err != nil {
		t.Fatal(err)
	} // Stub intentionally retains callback after unregister.
	if _, ok := <-s.Notifications(); ok {
		t.Fatal("closed subscription received late callback")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	busy, err := NewPolicyNotifications(2, 1, PolicyCondDbe, 1)
	if err != nil {
		t.Fatal(err)
	}
	if busy.token <= s.token {
		t.Fatal("callback token reused")
	}
	if err := busy.Close(); !errors.Is(err, ErrInUse) {
		t.Fatalf("unregister error lost: %v", err)
	}
	if err := PolicyTrigger(1); err != nil {
		t.Fatal(err)
	}
	if _, ok := <-busy.Notifications(); !ok {
		t.Fatal("failed unregister closed channel")
	}
	if err := busy.Close(); err != nil {
		t.Fatal(err)
	}
	failed, err := NewPolicyNotifications(99, 1, PolicyCondDbe, 1)
	if err == nil || failed == nil {
		t.Fatal("register failure lost cleanup owner")
	}
	if err := failed.Close(); err != nil {
		t.Fatal(err)
	}
	concurrent, err := NewPolicyNotifications(3, 1, PolicyCondDbe, 1)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- PolicyTrigger(3) }()
	if err := concurrent.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	for range concurrent.Notifications() { // In-flight responses may already be buffered.
	}
	if err := Disconnect(3); err != nil {
		t.Fatal(err)
	}
	if err := StopEmbedded(3); err != nil {
		t.Fatal(err)
	}
	if err := Shutdown(); err != nil {
		t.Fatal(err)
	}
	if _, ok := <-concurrent.Notifications(); ok {
		t.Fatal("shutdown late callback reached closed subscription")
	}
	policies, err := ReadPolicies(1, 1, 32, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(policies) != 32 || policies[31].Condition != 32 {
		t.Fatalf("policies=%+v", policies)
	}
	if err := PolicySet(1, 1, &Policy{Version: PolicyVersion1}, 0); err != nil {
		t.Fatal(err)
	}
	if err := PolicySet(1, 1, &Policy{}, 0); !errors.Is(err, ErrVerMismatch) {
		t.Fatalf("wrong version: %v", err)
	}
	var subscriptions []*PolicyNotifications
	for group := 1; group <= 64; group++ {
		s, err := NewPolicyNotifications(7, GpuGrp(group), PolicyCondDbe, 1)
		if err != nil {
			t.Fatal(err)
		}
		subscriptions = append(subscriptions, s)
	}
	if _, err := NewPolicyNotifications(7, 65, PolicyCondDbe, 1); err == nil {
		t.Fatal("subscription limit admitted")
	}
	for _, s := range subscriptions {
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
	}
	// Legacy symbol lookup must not modify the current lazy loader's published
	// symbol map while another caller reads it.
	beginOnly := PolicyCallbackV1(purego.NewCallback(func(*PolicyCallbackResponseV1) int32 { return 11 }))
	finishOnly := PolicyCallbackV1(purego.NewCallback(func(*PolicyCallbackResponseV1) int32 { return 22 }))
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				if err := PolicyRegisterV1(1, 1, PolicyCondPci, beginOnly, finishOnly); err != nil {
					t.Error(err)
				}
				if err := PolicyTrigger(1); err != nil {
					t.Error(err)
				}
			}
		})
	}
	wg.Wait()
}

func TestPolicyBounds(t *testing.T) {
	var absent *PolicyNotifications
	if absent.Notifications() != nil || absent.Dropped() != 0 || absent.Close() == nil {
		t.Fatal("nil subscription contract")
	}
	if unsafe.Sizeof(PolicyCallbackResponseV1{}) != 24 || unsafe.Offsetof(PolicyCallbackResponseV1{}.Val) != 8 {
		t.Fatal("legacy layout drift")
	}
	for _, capacity := range []int{-1, 0, 4097} {
		if _, err := NewPolicyNotifications(1, 1, PolicyCondDbe, capacity); err == nil {
			t.Errorf("capacity %d admitted", capacity)
		}
	}
	for _, condition := range []PolicyCondition{0, 128, -1} {
		if _, err := NewPolicyNotifications(1, 1, condition, 1); err == nil {
			t.Errorf("condition %d admitted", condition)
		}
	}
	for _, count := range []int{-1, 0, 33} {
		if _, err := ReadPolicies(1, 1, count, 0); err == nil {
			t.Errorf("count %d admitted", count)
		}
	}
	if got := PolicyBool(true); got.Tag != 0 || got.Val[0] != 1 {
		t.Fatalf("boolean=%+v", got)
	}
	if got := PolicyNumber(1 << 63); got.Tag != 1 || got.Val[0] != 1<<63 {
		t.Fatalf("number=%+v", got)
	}
}

func TestPolicyTrampolineTokenIsolation(t *testing.T) {
	s := &PolicyNotifications{queue: make(chan PolicyCallbackResponse, 1), token: 1 << 63}
	policyCallbackLookup.Store(s.token, s)
	defer policyCallbackLookup.Delete(s.token)
	r := PolicyCallbackResponse{Version: PolicyCallbackResponseVersion2, GpuId: 7}
	policyNotificationCallback(&r, s.token+1)
	if len(s.queue) != 0 {
		t.Fatal("unknown token admitted")
	}
	policyNotificationCallback(&r, s.token)
	r.GpuId = 9
	if got := <-s.queue; got.GpuId != 7 || got.Version != PolicyCallbackResponseVersion2 {
		t.Fatal("callback payload not copied exactly")
	}
	s.mu.Lock()
	policyNotificationCallback(&r, s.token)
	s.mu.Unlock()
	if s.Dropped() != 1 {
		t.Fatal("contended callback not counted")
	}
	policyNotificationCallback(nil, s.token)
	if s.Dropped() != 2 || len(s.queue) != 0 {
		t.Fatal("nil callback response not counted")
	}
	s.retire()
	if _, ok := policyCallbackLookup.Load(s.token); ok {
		t.Fatal("retired subscription state retained")
	}
	policyNotificationCallback(&r, s.token)
	if _, ok := <-s.queue; ok {
		t.Fatal("retired token admitted")
	}
}
