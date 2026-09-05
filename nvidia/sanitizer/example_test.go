package sanitizer_test

import (
	"errors"
	"fmt"
	"log"
	"sync/atomic"
	"unsafe"

	"github.com/tmc/lib/nvidia/sanitizer"
)

// The examples in this file are compiled but not run. None of them has an
// Output comment, because every one needs libsanitizer-public.so, an NVIDIA
// GPU and a CUDA driver, none of which the test machine is required to have.

// Example shows the whole shape of a subscription: load the library, register
// a callback before any CUDA call happens, enable the domains of interest, and
// unsubscribe at the end.
func Example() {
	if err := sanitizer.LoadFromToolkit(); err != nil {
		log.Fatal(err)
	}

	sub, err := sanitizer.SubscribeFunc(onCallback, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer sub.Unsubscribe()

	if err := sub.EnableDomain(true, sanitizer.CbDomainResource); err != nil {
		log.Fatal(err)
	}

	// Application CUDA work goes here. Anything the process did before
	// SubscribeFunc returned produced no callbacks and is not replayed.
}

// allocated counts device allocations the sanitizer reported. It is a
// package-level variable, and the callback touches nothing else, because the
// callback runs on a thread this package did not create.
var allocated atomic.Int64

// onCallback is invoked by the sanitizer for every enabled callback point.
//
// cbdata points into memory the library owns. It is valid only for the
// duration of this call, so anything worth keeping is copied out here.
func onCallback(userdata uintptr, domain sanitizer.CallbackDomain, cbid uint32, cbdata unsafe.Pointer) {
	if domain != sanitizer.CbDomainResource {
		return
	}
	if sanitizer.CallbackIdResource(cbid) != sanitizer.CbidResourceDeviceMemoryAlloc {
		return
	}
	// Copy the struct out of library memory before returning.
	data := *(*sanitizer.ResourceMemoryData)(cbdata)
	allocated.Add(int64(data.Size))
}

// ExampleLoad shows pointing the loader at an explicit path, which is what to
// do when the toolkit is somewhere SearchPaths does not look.
func ExampleLoad() {
	err := sanitizer.Load("/opt/nvidia/cuda-13.3/compute-sanitizer/libsanitizer-public.so")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("loaded from", sanitizer.Path())
}

// ExampleSearchPaths shows the candidate list LoadFromToolkit walks. The
// library is not on the loader's default search path, so this list is usually
// what a "not found" failure comes down to.
func ExampleSearchPaths() {
	for _, p := range sanitizer.SearchPaths() {
		fmt.Println(p)
	}
}

// ExampleSubscribeFunc_secondSubscriber shows that the one-subscriber-per-process
// rule is enforced by the library, not by this package: a second subscribe
// returns the library's own ErrMaxLimitReached.
func ExampleSubscribeFunc_secondSubscriber() {
	first, err := sanitizer.SubscribeFunc(onCallback, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer first.Unsubscribe()

	_, err = sanitizer.SubscribeFunc(onCallback, 0)
	if errors.Is(err, sanitizer.ErrMaxLimitReached) {
		fmt.Println("a subscriber already exists in this process")
	}
}

// ExampleHasSubscribe shows probing for a symbol before using it, so a binding
// missing from an older install is a branch rather than an error at the call.
func ExampleHasSubscribe() {
	if !sanitizer.HasSubscribe() {
		fmt.Println("this install predates sanitizerSubscribe")
		return
	}
	fmt.Println("sanitizerSubscribe is available")
}
