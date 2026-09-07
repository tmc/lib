// Package xla provides a high-level, pure Go API for executing XLA programs.
package xla

import (
	"fmt"

	"github.com/tmc/lib/openxla"
)

// Engine manages the PJRT plugin lifecycle and client initialization.
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

// Close destroys the client and shuts down the engine.
func (e *Engine) Close() error {
	if e.client != nil {
		if err := e.client.Destroy(); err != nil {
			return fmt.Errorf("xla: failed to close engine: %w", err)
		}
		e.client = nil
	}
	return nil
}

// Devices returns the locally addressable devices for this engine.
func (e *Engine) Devices() ([]*openxla.Device, error) {
	devices, err := e.client.AddressableDevices()
	if err != nil {
		return nil, fmt.Errorf("xla: failed to get devices: %w", err)
	}
	return devices, nil
}
