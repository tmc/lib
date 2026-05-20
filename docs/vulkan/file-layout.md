# File Layout: vulkan

- `doc.go`: package documentation.
- `errors.go`: generated VkResult values and error handling.
- `version.go`: Vulkan API version helpers.
- `types.go`: public first-slice handles and create-info types.
- `instance.go`: loader, instance creation, close, and physical-device enumeration.
- `example_test.go`: runnable examples for version helpers and create-info construction.
- `integration_test.go`: manual loader and hardware gates.
- `result_test.go`: cheap result and version tests.
- `instance_test.go`: cheap create-info marshalling tests.
- `internal/abi`: registry-derived raw ABI structs, handles, function variables, and layout tests.
- `internal/loader`: purego/syscall dynamic-loader boundary.
- `manual-gates.md`: explicit runtime gates.
