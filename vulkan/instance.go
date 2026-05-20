package vulkan

import (
	"runtime"
	"sync"
	"unsafe"

	"github.com/tmc/lib/vulkan/internal/abi"
	"github.com/tmc/lib/vulkan/internal/loader"
)

var loadState struct {
	mu     sync.Mutex
	loaded bool
	err    error
}

// Load loads the platform Vulkan loader and registers the first-slice commands.
func Load() error {
	loadState.mu.Lock()
	defer loadState.mu.Unlock()
	if loadState.loaded {
		return nil
	}
	lib, err := loader.Open("")
	if err != nil {
		loadState.err = err
		return err
	}
	if err := abi.Register(lib); err != nil {
		loadState.err = err
		return err
	}
	loadState.loaded = true
	loadState.err = nil
	return nil
}

// CreateInstance creates a Vulkan instance.
func CreateInstance(info *InstanceCreateInfo) (*Instance, error) {
	if err := Load(); err != nil {
		return nil, err
	}
	var raw abi.Instance
	ci, keepAlive := vulkanizeInstanceCreateInfo(info)
	res := Result(abi.CreateInstance(&ci, nil, &raw))
	runtime.KeepAlive(keepAlive)
	if res.failed() {
		return nil, res
	}
	return &Instance{handle: uintptr(raw)}, nil
}

// Close destroys the Vulkan instance.
func (i *Instance) Close() error {
	if i == nil || i.closed || i.handle == 0 {
		return nil
	}
	abi.DestroyInstance(abi.Instance(i.handle), nil)
	i.closed = true
	i.handle = 0
	return nil
}

// EnumeratePhysicalDevices returns the physical devices visible to i.
func (i *Instance) EnumeratePhysicalDevices() ([]PhysicalDevice, error) {
	if i == nil || i.closed || i.handle == 0 {
		return nil, ErrClosed
	}
	for {
		var n uint32
		res := Result(abi.EnumeratePhysicalDevices(abi.Instance(i.handle), &n, nil))
		if res.failed() {
			return nil, res
		}
		if n == 0 {
			return nil, nil
		}
		raw := make([]abi.PhysicalDevice, n)
		res = Result(abi.EnumeratePhysicalDevices(abi.Instance(i.handle), &n, unsafe.Pointer(unsafe.SliceData(raw))))
		if res == Incomplete {
			continue
		}
		if res.failed() {
			return nil, res
		}
		out := make([]PhysicalDevice, n)
		for j := range out {
			out[j] = PhysicalDevice{handle: uintptr(raw[j])}
		}
		return out, nil
	}
}

type createInfoKeepAlive struct {
	appName        []byte
	engineName     []byte
	layerNames     [][]byte
	extensionNames [][]byte
	layerPtrs      []unsafe.Pointer
	extensionPtrs  []unsafe.Pointer
	app            abi.ApplicationInfo
}

func vulkanizeInstanceCreateInfo(info *InstanceCreateInfo) (abi.InstanceCreateInfo, *createInfoKeepAlive) {
	keep := &createInfoKeepAlive{}
	var out abi.InstanceCreateInfo
	out.SType = abi.StructureTypeInstanceCreateInfo
	if info == nil {
		return out, keep
	}
	out.Flags = abi.InstanceCreateFlags(info.Flags)
	if app := info.ApplicationInfo; app != nil {
		keep.appName = cString(app.ApplicationName)
		keep.engineName = cString(app.EngineName)
		keep.app = abi.ApplicationInfo{
			SType:              abi.StructureTypeApplicationInfo,
			PApplicationName:   unsafe.SliceData(keep.appName),
			ApplicationVersion: app.ApplicationVersion,
			PEngineName:        unsafe.SliceData(keep.engineName),
			EngineVersion:      app.EngineVersion,
			APIVersion:         defaultAPIVersion(app.APIVersion),
		}
		out.PApplicationInfo = &keep.app
	}
	keep.layerNames, keep.layerPtrs = cStringPointers(info.EnabledLayerNames)
	if len(keep.layerPtrs) > 0 {
		out.EnabledLayerCount = uint32(len(keep.layerPtrs))
		out.PpEnabledLayerNames = unsafe.Pointer(unsafe.SliceData(keep.layerPtrs))
	}
	keep.extensionNames, keep.extensionPtrs = cStringPointers(info.EnabledExtensionNames)
	if len(keep.extensionPtrs) > 0 {
		out.EnabledExtensionCount = uint32(len(keep.extensionPtrs))
		out.PpEnabledExtensionNames = unsafe.Pointer(unsafe.SliceData(keep.extensionPtrs))
	}
	return out, keep
}

func cString(s string) []byte {
	if s == "" {
		return []byte{0}
	}
	b := make([]byte, 0, len(s)+1)
	b = append(b, s...)
	return append(b, 0)
}

func cStringPointers(values []string) ([][]byte, []unsafe.Pointer) {
	if len(values) == 0 {
		return nil, nil
	}
	bytes := make([][]byte, len(values))
	ptrs := make([]unsafe.Pointer, len(values))
	for i, v := range values {
		bytes[i] = cString(v)
		ptrs[i] = unsafe.Pointer(unsafe.SliceData(bytes[i]))
	}
	return bytes, ptrs
}

func defaultAPIVersion(v uint32) uint32 {
	if v == 0 {
		return APIVersion10
	}
	return v
}
