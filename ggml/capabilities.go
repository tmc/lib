package ggml

type DeviceKind int

const (
	DeviceUnknown DeviceKind = iota
	DeviceHost
	DeviceCUDA
	DeviceROCM
	DeviceNetwork
	DeviceApple
)

type ProviderInfo struct {
	Name      string
	Version   string
	Runtime   string
	Transport string
}

type HealthSnapshot struct {
	Ready     bool
	Message   string
	Retryable bool
}

type Capabilities struct {
	DeviceKinds []DeviceKind
	Provider    ProviderInfo
	Health      HealthSnapshot
}
