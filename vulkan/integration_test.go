package vulkan

import (
	"os"
	"testing"
)

func TestIntegrationLoader(t *testing.T) {
	if os.Getenv("VULKAN_TEST_LOADER") != "1" {
		t.Skip("set VULKAN_TEST_LOADER=1 to load the platform Vulkan loader")
	}
	if err := Load(); err != nil {
		t.Fatal(err)
	}
}

func TestIntegrationHardware(t *testing.T) {
	if os.Getenv("VULKAN_TEST_HARDWARE") != "1" {
		t.Skip("set VULKAN_TEST_HARDWARE=1 on a host with a Vulkan loader and compatible GPU")
	}
	inst, err := CreateInstance(&InstanceCreateInfo{ApplicationInfo: &ApplicationInfo{
		ApplicationName: "tmc-lib-vulkan-test",
		APIVersion:      APIVersion10,
	}})
	if err != nil {
		t.Fatal(err)
	}
	defer inst.Close()
	if _, err := inst.EnumeratePhysicalDevices(); err != nil {
		t.Fatal(err)
	}
}
