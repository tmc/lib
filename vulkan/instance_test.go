package vulkan

import "testing"

func TestInstanceCreateInfoDoesNotExposeSType(t *testing.T) {
	info := InstanceCreateInfo{
		ApplicationInfo:       &ApplicationInfo{ApplicationName: "test", APIVersion: APIVersion10},
		EnabledLayerNames:     []string{"VK_LAYER_KHRONOS_validation"},
		EnabledExtensionNames: []string{"VK_KHR_surface"},
	}
	raw, keep := vulkanizeInstanceCreateInfo(&info)
	if raw.SType == 0 {
		t.Fatal("missing instance create sType")
	}
	if raw.PApplicationInfo == nil {
		t.Fatal("missing application info")
	}
	if raw.PApplicationInfo.APIVersion != APIVersion10 {
		t.Fatal("application info did not default API version")
	}
	if raw.EnabledLayerCount != 1 || raw.PpEnabledLayerNames == nil {
		t.Fatal("layer names were not converted")
	}
	if raw.EnabledExtensionCount != 1 || raw.PpEnabledExtensionNames == nil {
		t.Fatal("extension names were not converted")
	}
	if len(keep.layerNames) != 1 || len(keep.extensionNames) != 1 {
		t.Fatal("converted strings are not kept alive")
	}
}
