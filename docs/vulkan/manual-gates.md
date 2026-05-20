# Manual Gates: vulkan

Default `go test` does not load Vulkan and does not require a GPU.

## Loader Gate

Command:

```sh
VULKAN_TEST_LOADER=1 go test -run '^TestIntegrationLoader$' -count=1 ./vulkan
```

This gate loads the platform Vulkan loader using the generated pure-Go loader.

## Hardware Gate

Command:

```sh
VULKAN_TEST_HARDWARE=1 go test -run '^TestIntegrationHardware$' -count=1 ./vulkan
```

This gate creates a Vulkan instance and enumerates physical devices. It requires
a host with a Vulkan loader and compatible driver. It does not require a display
server or WSI surface in this first slice.
