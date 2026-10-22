package dsv4expertcache

import (
	"bytes"
	"errors"
	"fmt"
)

func ExampleReserveMetalPrefill() {
	cache, reserve, err := ReserveMetalPrefill(1000, 10, 20, 3)
	fmt.Println(cache, reserve, err)
	// Output: 600 400 <nil>
}

func ExampleMetalPrefillEnabled() {
	// Scalar calls and quality mode always use the ordinary schedule.
	fmt.Println(MetalPrefillEnabled(1, 256, 6, 16, 10, 256, false))
	fmt.Println(MetalPrefillEnabled(64, 256, 6, 16, 10, 256, true))
	// Output:
	// false
	// false
}

func ExampleCompactMetalRoutes() {
	unique, remap, err := CompactMetalRoutes([]int32{7, 2, 7, 3, 2}, 8)
	fmt.Println(unique, remap, err)
	// Output: [7 2 3] [0 1 0 2 1] <nil>
}

func ExampleMetalHotness_NoteToken() {
	var hotness MetalHotness
	fmt.Println(hotness.NoteToken(0))
	fmt.Println(hotness.NoteToken(80))
	// Output:
	// <nil>
	// invalid Metal hotness layer
}

func ExampleMetalHotness_Acquire() {
	var id Identity
	id.Model[0], id.Quant[0], id.Layout[0] = 1, 2, 3
	cache, err := New(id, 2, 6, 12)
	if err != nil {
		fmt.Println(err)
		return
	}
	var hotness MetalHotness
	lease, err := hotness.Acquire(cache, []Key{{id, 0, 1}, {id, 0, 0}, {id, 0, 1}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(lease.Slots())
	fmt.Println(lease.Abort(), cache.Close())
	// Output:
	// [0 1 0]
	// <nil> <nil>
}

func ExampleMetalPinFailure_Error() {
	failure := &MetalPinFailure{Err: errors.New("memory limit")}
	fmt.Println(failure.Error())
	// Output: pin Metal expert slot: memory limit
}

func ExampleMetalPinFailure_Unwrap() {
	cause := errors.New("memory limit")
	failure := &MetalPinFailure{Err: cause}
	fmt.Println(errors.Is(failure, cause))
	// Output: true
}

// These examples use a host-backed transport so lease ownership can be shown
// without allocating a Metal device. Production uses NewMetalUpload.
func exampleMetalTransport() (*MetalUpload, *Source, *Cache) {
	u, _ := newMetalUpload(new(fakeMetalUpload), 2, [3]int{2, 2, 2}, 12)
	tr := Triplet{Tensor{0, 4, 2, 2}, Tensor{4, 4, 2, 2}, Tensor{8, 4, 2, 2}}
	s, _ := NewSource(bytes.NewReader([]byte("abcdefghijkl")), 12, identity(), map[int]Triplet{0: tr})
	c, _ := New(identity(), 2, 6, 12)
	return u, s, c
}

func ExampleMetalUpload_Select() {
	u, source, cache := exampleMetalTransport()
	selection, err := u.Select(source, cache, 0, []int{1, 0, 1}, false)
	if err != nil {
		fmt.Println(err)
		return
	}
	slots, err := selection.Slots()
	fmt.Println(slots, err)
	fmt.Println(selection.Release(metalWriteCompletion{}), u.Close(), cache.Close())
	// Output:
	// [0 1 0] <nil>
	// <nil> <nil> <nil>
}

func ExampleMetalUpload_SelectRoutes() {
	u, source, cache := exampleMetalTransport()
	selection, experts, err := u.SelectRoutes(source, cache, 0, []int{1, 0, 1})
	fmt.Println(experts, err)
	if err == nil {
		fmt.Println(selection.Release(metalWriteCompletion{}))
	}
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// [1 0] <nil>
	// <nil>
	// <nil> <nil>
}

func ExampleMetalUpload_Stage() {
	u, source, cache := exampleMetalTransport()
	lease, _ := cache.Acquire([]Key{key(1)})
	fmt.Println(u.Stage(source, lease, false))
	fmt.Println(lease.Release(metalWriteCompletion{}), u.Close(), cache.Close())
	// Output:
	// <nil>
	// <nil> <nil> <nil>
}

func ExampleMetalUpload_Submit() {
	var upload *MetalUpload
	// Writes outside an owned staging lease are refused.
	_, err := upload.Submit(Upload{}, 0, []byte("ab"))
	fmt.Println(err)
	// Output: invalid Metal expert write
}

func ExampleMetalUpload_PinBackoff() {
	u, source, cache := exampleMetalTransport()
	selection, _, _ := u.SelectRoutes(source, cache, 0, []int{0, 1})
	_ = selection.Release(metalWriteCompletion{})
	capacity, err := u.PinBackoff(cache, &MetalPinFailure{LockedSlots: 1})
	fmt.Println(capacity, err)
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// 1 <nil>
	// <nil> <nil>
}

func ExampleMetalUpload_SeedPriority() {
	u, source, cache := exampleMetalTransport()
	selection, _ := u.Select(source, cache, 0, []int{1}, false)
	_ = selection.Release(metalWriteCompletion{})
	fmt.Println(u.SeedPriority(0, 1, 32))
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// <nil>
	// <nil> <nil>
}

func ExampleMetalUpload_NoteToken() {
	u, _, cache := exampleMetalTransport()
	fmt.Println(u.NoteToken(0))
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// <nil>
	// <nil> <nil>
}

func ExampleMetalUpload_NoteMappedRoutes() {
	u, source, cache := exampleMetalTransport()
	fmt.Println(u.NoteMappedRoutes(source, 0, []int32{1, 0, 1}))
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// <nil>
	// <nil> <nil>
}

func ExampleMetalUpload_Close() {
	u, _, cache := exampleMetalTransport()
	fmt.Println(u.Close(), u.Close(), cache.Close())
	// Output: <nil> <nil> <nil>
}

func ExampleMetalSelection_Ready() {
	u, source, cache := exampleMetalTransport()
	selection, _ := u.Select(source, cache, 0, []int{1}, false)
	fmt.Println(selection.Ready())
	_ = selection.Release(metalWriteCompletion{})
	fmt.Println(selection.Ready())
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// true
	// false
	// <nil> <nil>
}

func ExampleMetalSelection_Discard() {
	var selection *MetalSelection
	fmt.Println(selection.Discard())
	// Output: <nil>
}

func ExampleMetalSelection_Slots() {
	u, source, cache := exampleMetalTransport()
	selection, _ := u.Select(source, cache, 0, []int{1, 1}, false)
	slots, err := selection.Slots()
	fmt.Println(slots, err)
	_ = selection.Release(metalWriteCompletion{})
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// [0 0] <nil>
	// <nil> <nil>
}

func ExampleMetalSelection_Release() {
	u, source, cache := exampleMetalTransport()
	selection, _ := u.Select(source, cache, 0, []int{1}, false)
	fmt.Println(u.Close())
	fmt.Println(selection.Release(metalWriteCompletion{}))
	fmt.Println(u.Close(), cache.Close())
	// Output:
	// Metal consumer lease active
	// <nil>
	// <nil> <nil>
}
