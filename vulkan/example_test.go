package vulkan_test

import (
	"fmt"

	"github.com/tmc/lib/vulkan"
)

func ExampleMakeAPIVersion() {
	v := vulkan.MakeAPIVersion(0, 1, 3, 268)
	fmt.Println(vulkan.APIVersionMajor(v), vulkan.APIVersionMinor(v), vulkan.APIVersionPatch(v))
	// Output: 1 3 268
}

func ExampleInstanceCreateInfo() {
	info := vulkan.InstanceCreateInfo{
		ApplicationInfo: &vulkan.ApplicationInfo{
			ApplicationName: "example",
			APIVersion:      vulkan.APIVersion10,
		},
		EnabledLayerNames:     []string{"VK_LAYER_KHRONOS_validation"},
		EnabledExtensionNames: []string{"VK_KHR_surface"},
	}
	fmt.Println(info.ApplicationInfo.ApplicationName)
	fmt.Println(len(info.EnabledLayerNames), len(info.EnabledExtensionNames))
	// Output:
	// example
	// 1 1
}
