package vulkan

import (
	"errors"
	"fmt"
)

var ErrClosed = errors.New("vulkan: closed")

// Result is a Vulkan VkResult value.
type Result int32

const (
	ErrorUnknown              Result = -13
	ErrorFragmentedPool       Result = -12
	ErrorFormatNotSupported   Result = -11
	ErrorTooManyObjects       Result = -10
	ErrorIncompatibleDriver   Result = -9
	ErrorFeatureNotPresent    Result = -8
	ErrorExtensionNotPresent  Result = -7
	ErrorLayerNotPresent      Result = -6
	ErrorMemoryMapFailed      Result = -5
	ErrorDeviceLost           Result = -4
	ErrorInitializationFailed Result = -3
	ErrorOutOfDeviceMemory    Result = -2
	ErrorOutOfHostMemory      Result = -1
	Success                   Result = 0
	NotReady                  Result = 1
	Timeout                   Result = 2
	EventSet                  Result = 3
	EventReset                Result = 4
	Incomplete                Result = 5
)

// Error returns the Vulkan result name.
func (r Result) Error() string {
	switch r {
	case ErrorUnknown:
		return "VK_ERROR_UNKNOWN"
	case ErrorFragmentedPool:
		return "VK_ERROR_FRAGMENTED_POOL"
	case ErrorFormatNotSupported:
		return "VK_ERROR_FORMAT_NOT_SUPPORTED"
	case ErrorTooManyObjects:
		return "VK_ERROR_TOO_MANY_OBJECTS"
	case ErrorIncompatibleDriver:
		return "VK_ERROR_INCOMPATIBLE_DRIVER"
	case ErrorFeatureNotPresent:
		return "VK_ERROR_FEATURE_NOT_PRESENT"
	case ErrorExtensionNotPresent:
		return "VK_ERROR_EXTENSION_NOT_PRESENT"
	case ErrorLayerNotPresent:
		return "VK_ERROR_LAYER_NOT_PRESENT"
	case ErrorMemoryMapFailed:
		return "VK_ERROR_MEMORY_MAP_FAILED"
	case ErrorDeviceLost:
		return "VK_ERROR_DEVICE_LOST"
	case ErrorInitializationFailed:
		return "VK_ERROR_INITIALIZATION_FAILED"
	case ErrorOutOfDeviceMemory:
		return "VK_ERROR_OUT_OF_DEVICE_MEMORY"
	case ErrorOutOfHostMemory:
		return "VK_ERROR_OUT_OF_HOST_MEMORY"
	case Success:
		return "VK_SUCCESS"
	case NotReady:
		return "VK_NOT_READY"
	case Timeout:
		return "VK_TIMEOUT"
	case EventSet:
		return "VK_EVENT_SET"
	case EventReset:
		return "VK_EVENT_RESET"
	case Incomplete:
		return "VK_INCOMPLETE"
	default:
		return fmt.Sprintf("VkResult(%d)", int32(r))
	}
}

// Err returns nil for non-error status results and r otherwise.
func (r Result) Err() error {
	if r.failed() {
		return r
	}
	return nil
}

func (r Result) failed() bool {
	return r < 0
}
