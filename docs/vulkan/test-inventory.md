# Test Inventory: vulkan

Cheap tests:

- `TestResultError` checks Result error behavior.
- `TestAPIVersion` checks Vulkan API version encoding helpers.
- `TestInstanceCreateInfoDoesNotExposeSType` checks public create-info marshalling without loading Vulkan.
- `ExampleMakeAPIVersion` and `ExampleInstanceCreateInfo` provide runnable public examples.
- `internal/abi` layout tests check 64-bit struct sizes and offsets.
- `internal/loader` tests check platform loader-name selection without loading a library.
- `contrib/vulkan` tests check registry selection, pointer-depth parsing, derived `sType` values, and generated ABI signatures.

Manual tests:

- `TestIntegrationLoader` runs only with `VULKAN_TEST_LOADER=1`.
- `TestIntegrationHardware` runs only with `VULKAN_TEST_HARDWARE=1`.
