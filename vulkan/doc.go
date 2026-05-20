// Package vulkan provides generated pure-Go bindings for a small Vulkan
// loader and discovery surface.
//
// This first slice is intentionally narrow. It can load the platform Vulkan
// loader, create and destroy an instance, and enumerate physical devices. It
// does not expose logical devices, queues, command buffers, window-system
// integration, pNext chains, unions, or callbacks yet.
//
// Default tests do not load Vulkan or require a GPU. Runtime loader and GPU
// checks are explicit manual gates documented in manual-gates.md.
package vulkan
