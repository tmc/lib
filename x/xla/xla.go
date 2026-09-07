package xla

import (
	"fmt"

	"github.com/tmc/lib/openxla"
)

// Engine owns a PJRT client. Close its buffers, executables and models before
// closing the engine. Its zero value is closed.
type Engine struct {
	plugin *openxla.Plugin
	client *openxla.ClientWrapper
}

// NewEngine loads a PJRT plugin and initializes the client.
func NewEngine(pluginPath string) (*Engine, error) {
	plugin, err := openxla.OpenPlugin(pluginPath)
	if err != nil {
		return nil, fmt.Errorf("xla: failed to open plugin: %w", err)
	}

	client, err := plugin.NewClient()
	if err != nil {
		return nil, fmt.Errorf("xla: failed to initialize client: %w", err)
	}

	return &Engine{
		plugin: plugin,
		client: client,
	}, nil
}

// NewEngineFromAPI initializes a new client using an already loaded PJRT API.
// The caller must keep the API table and its library loaded until all engine
// resources have been closed. Close destroys the client, not the API or library.
func NewEngineFromAPI(api *openxla.Api) (*Engine, error) {
	client, err := openxla.NewClientFromApi(api)
	if err != nil {
		return nil, fmt.Errorf("xla: initialize client: %w", err)
	}
	return &Engine{client: client}, nil
}

// Close destroys the client and shuts down the engine.
func (e *Engine) Close() error {
	if e == nil || e.client == nil {
		return nil
	}
	resource := e.client
	e.client = nil
	if err := resource.Destroy(); err != nil {
		return fmt.Errorf("xla: close engine: %w", err)
	}
	return nil
}

// Devices returns the locally addressable devices for this engine.
func (e *Engine) Devices() ([]*openxla.Device, error) {
	if e == nil || e.client == nil {
		return nil, fmt.Errorf("xla: engine is closed")
	}
	devices, err := e.client.AddressableDevices()
	if err != nil {
		return nil, fmt.Errorf("xla: failed to get devices: %w", err)
	}
	return devices, nil
}
