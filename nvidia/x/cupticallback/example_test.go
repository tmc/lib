package cupticallback_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cupticallback"
)

func ExampleDispatcher() {
	d := &cupticallback.Dispatcher{
		OnDriverApi: func(cbid uint32, data *cupticallback.ApiData) {
			fmt.Println("driver api:", data.FunctionName)
		},
		OnResource: func(cbid uint32, data *cupticallback.ResourceData) {
			fmt.Println("resource cbid:", cbid)
		},
	}

	d.HandleCallback(&cupticallback.Event{
		Domain: cupti.CbDomainDriverApi,
		Api: &cupticallback.ApiData{
			FunctionName: "cuMemAlloc_v2",
		},
	})
	d.HandleCallback(&cupticallback.Event{
		Domain: cupti.CbDomainResource,
		ID:     1,
		Res: &cupticallback.ResourceData{
			ResourceHandle: 42,
		},
	})

	// Output:
	// driver api: cuMemAlloc_v2
	// resource cbid: 1
}
