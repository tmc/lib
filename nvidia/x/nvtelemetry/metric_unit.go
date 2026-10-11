package nvtelemetry

// LookupMetricUnit returns the canonical wire unit for name, when specified.
// An unknown result must not be interpreted as permission to change units.
func LookupMetricUnit(name string) (unit string, found bool) {
	unit, found = expectedMetricUnits[name]
	return unit, found
}
