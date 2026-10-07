package cudalaunch

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudashared"
	"testing"
)

func TestSharedPlanRequiresOwner(t *testing.T) {
	for _, l := range []*Launcher{nil, Kernel(0), OwnedKernel(nil)} {
		if _, err := l.PrepareSharedPlan(nil, 0); err == nil {
			t.Fatal("unowned plan accepted")
		}
	}
}
func TestSharedPlanOverrideRefused(t *testing.T) {
	l := Kernel(0).WithAttrs(driver.LaunchAttribute{Id: driver.LaunchAttributeSharedMemoryMode})
	if err := l.sharedPlanMode(); err == nil {
		t.Fatal("override accepted")
	}
}
func TestSharedPlanBytesCannotChange(t *testing.T) {
	p, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 64, DynamicMax: 64, TotalMax: 64, Source: "host fixture"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = p.Add(cudashared.Request{Name: "dynamic", Size: 16, Alignment: 1, Dynamic: true}); err != nil {
		t.Fatal(err)
	}
	l := Kernel(0)
	l.sharedPlan = p
	l.shmem = 17
	if err = l.checkSharedPlan(); err == nil {
		t.Fatal("changed dynamic bytes accepted")
	}
}
func ExampleLauncher_PrepareSharedPlan() {
	_, err := OwnedKernel(nil).PrepareSharedPlan(nil, 0)
	fmt.Println(err)
	// Output: cudalaunch: shared plan requires an owned kernel
}
