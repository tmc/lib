package gorccl

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
	Collectives  []string
	PointToPoint bool
	DeviceKinds  []DeviceKind
	Provider     ProviderInfo
	Health       HealthSnapshot
}
