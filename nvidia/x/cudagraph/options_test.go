package cudagraph

import "testing"

func TestInstantiateOptions(t *testing.T) {
	for _, tt := range []struct {
		name    string
		options InstantiateOptions
		want    uint64
		invalid bool
	}{
		{"default", InstantiateOptions{}, 0, false},
		{"auto-free", InstantiateOptions{AutoFreeOnLaunch: true}, 1, false},
		{"device", InstantiateOptions{DeviceLaunch: true}, 4, false},
		{"priority", InstantiateOptions{UseNodePriority: true}, 8, false},
		{"device priority", InstantiateOptions{DeviceLaunch: true, UseNodePriority: true}, 12, false},
		{"auto-free priority", InstantiateOptions{AutoFreeOnLaunch: true, UseNodePriority: true}, 9, false},
		{"incompatible", InstantiateOptions{AutoFreeOnLaunch: true, DeviceLaunch: true}, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.options.flags()
			if got != tt.want || (err != nil) != tt.invalid {
				t.Fatalf("flags = %d, error = %v", got, err)
			}
		})
	}
}

func TestInstantiateOptionsRejectBeforeNative(t *testing.T) {
	g := &Graph{raw: 1} // A sentinel handle must never reach the driver.
	if _, err := g.InstantiateWithOptions(InstantiateOptions{DeviceLaunch: true, AutoFreeOnLaunch: true}); err == nil {
		t.Fatal("accepted incompatible flags")
	}
	var closed *Graph
	if _, err := closed.InstantiateWithOptions(InstantiateOptions{}); err == nil {
		t.Fatal("accepted closed graph")
	}
}
