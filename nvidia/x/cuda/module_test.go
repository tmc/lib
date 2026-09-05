package cuda

import (
	"fmt"
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
