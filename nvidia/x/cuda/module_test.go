package cuda

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"testing"
)

func TestClosedModule(t *testing.T) {
	m := new(Module)
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Function("kernel"); err == nil {
		t.Fatal("lookup on closed module")
	}
	k := &Kernel{module: m}
	if err := k.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, nil); err == nil {
		t.Fatal("launch on closed module")
	}
}

func ExampleOwnModule() {
	_, err := OwnModule(nil, 0)
	fmt.Println(err)
	// Output: cuda: nil device or module
}

func ExampleModule_Close() {
	var m Module
	fmt.Println(m.Close())
	// Output: <nil>
}

func ExampleModule_Function() {
	var m Module
	_, err := m.Function("kernel")
	fmt.Println(err)
	// Output: cuda: closed module
}

func ExampleKernel_Launch() {
	var k Kernel
	fmt.Println(k.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, nil))
	// Output: cuda: closed module
}

func TestWithGlobalClosed(t *testing.T) {
	for _, m := range []*Module{nil, {}} {
		called := false
		err := m.WithGlobal("value", func(driver.DevicePtr, uint) error { called = true; return nil })
		if err == nil || called {
			t.Fatalf("WithGlobal = %v, called %v", err, called)
		}
	}
	m := new(Module)
	if err := m.WithGlobal("", func(driver.DevicePtr, uint) error { return nil }); err == nil {
		t.Fatal("empty name accepted")
	}
	if err := m.WithGlobal("value", nil); err == nil {
		t.Fatal("nil callback accepted")
	}
}

func ExampleModule_WithGlobal() {
	var m Module
	err := m.WithGlobal("scale", func(ptr driver.DevicePtr, size uint) error {
		fmt.Println(size)
		return nil
	})
	fmt.Println(err)
	// Output: cuda: closed module
}
