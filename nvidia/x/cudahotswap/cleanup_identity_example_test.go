package cudahotswap

import (
	"errors"
	"fmt"
	"sync/atomic"
)

// A real caller obtains the error from Register or Push. This host fixture
// supplies two occurrences of one uncomparable cleanup owner without a device.
func exampleIdentityError() (error, func() error, *atomic.Int32) {
	calls := new(atomic.Int32)
	owner := identityCloser{token: []byte{1}, calls: calls}
	r := (&Service{}).retainCleanup(nil)
	r.adopt(owner)
	r.adopt(owner)
	return r.Close(), r.Close, calls
}

func ExampleCleanupIdentityError() {
	err, retry, calls := exampleIdentityError()
	var identity *CleanupIdentityError
	if errors.As(err, &identity) {
		// The loader's caller knows these two occurrences are aliases.
		if err := identity.ResolveAliases([]int{0, 1}); err != nil {
			fmt.Println(err)
			return
		}
	}
	fmt.Println(retry())
	fmt.Println(calls.Load())
	// Output:
	// <nil>
	// 1
}

func ExampleCleanupIdentityError_Owners() {
	err, _, calls := exampleIdentityError()
	var identity *CleanupIdentityError
	if errors.As(err, &identity) {
		fmt.Println(len(identity.Owners()), calls.Load())
	}
	// Output: 2 0
}

func ExampleCleanupIdentityError_ResolveAliases() {
	err, retry, calls := exampleIdentityError()
	var identity *CleanupIdentityError
	if errors.As(err, &identity) {
		fmt.Println(identity.ResolveAliases([]int{0, 1}))
		fmt.Println(calls.Load()) // Resolution does not close anything.
		fmt.Println(retry())
		fmt.Println(calls.Load())
	}
	// Output:
	// <nil>
	// 0
	// <nil>
	// 1
}
