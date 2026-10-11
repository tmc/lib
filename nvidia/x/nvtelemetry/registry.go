package nvtelemetry

import "fmt"

// MetricKind represents the aggregation and temporal semantics of a metric.
type MetricKind = string

const (
	KindGauge   MetricKind = "gauge"
	KindCounter MetricKind = "counter"
	KindBitmask MetricKind = "bitmask"
	KindEvent   MetricKind = "event"
)

// DefaultMetricRegistryVersion is the current version of the metric definitions dictionary.
const DefaultMetricRegistryVersion = "2026.10-v2"

// metricKinds maps canonical metric names to their MetricKind classification.
var metricKinds = map[string]MetricKind{
	MetricGPUUtilization:      KindGauge,
	MetricMemoryUtilization:   KindGauge,
	MetricMemoryUsedBytes:     KindGauge,
	MetricMemoryTotalBytes:    KindGauge,
	MetricPowerUsageMW:        KindGauge,
	MetricTotalEnergyMJ:       KindCounter,
	MetricTemperatureC:        KindGauge,
	MetricSMClockMHz:          KindGauge,
	MetricMemoryClockMHz:      KindGauge,
	MetricClocksEventReasons:  KindBitmask,
	MetricSMActivity:          KindGauge,
	MetricTensorActivity:      KindGauge,
	MetricDRAMActivity:        KindGauge,
	MetricPCIeRxBytesPerSec:   KindGauge,
	MetricPCIeTxBytesPerSec:   KindGauge,
	MetricNVLinkRxBytesPerSec: KindGauge,
	MetricNVLinkTxBytesPerSec: KindGauge,
	MetricThrottleReasons:     KindBitmask,
	MetricPCIERxBytes:         KindCounter,
	MetricPCIETxBytes:         KindCounter,
	MetricXIDEvent:            KindEvent,
	"imex_domain_status":      KindEvent,
	"imex_daemon_status":      KindEvent,
}

// LookupMetricKind returns the kind classification for a metric name, and whether it was found.
func LookupMetricKind(name string) (kind string, found bool) {
	k, ok := metricKinds[name]
	return k, ok
}

// ValidateMetricAveraging verifies whether a metric may be summarized using numeric averaging.
// Bitmask and event metrics are rejected because numeric averages across discrete bit patterns
// or error event codes are mathematically invalid.
func ValidateMetricAveraging(name string) error {
	kind, found := LookupMetricKind(name)
	if found && (kind == KindBitmask || kind == KindEvent) {
		return fmt.Errorf("%w: metric %q of kind %q cannot be averaged", ErrBitmaskAveragingRefused, name, kind)
	}
	return nil
}
