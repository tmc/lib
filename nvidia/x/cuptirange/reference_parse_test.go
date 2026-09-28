package cuptirange_test

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

var metricNumber = regexp.MustCompile(`^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$`)

func parseMetricValue(output, metric string) (float64, error) {
	var value float64
	rows := 0
	for _, line := range strings.Split(output, "\n") {
		fields := strings.Fields(line)
		for i, field := range fields {
			if field != metric {
				continue
			}
			rows++
			if i+1 == len(fields) || !metricNumber.MatchString(fields[i+1]) {
				return 0, fmt.Errorf("metric %q row has no value", metric)
			}
			parsed, err := strconv.ParseFloat(fields[i+1], 64)
			if err != nil {
				return 0, fmt.Errorf("parse metric %q value: %w", metric, err)
			}
			value = parsed
		}
	}
	if rows == 0 {
		return 0, fmt.Errorf("metric %q not found", metric)
	}
	if rows != 1 {
		return 0, fmt.Errorf("metric %q has %d range values; select one range", metric, rows)
	}
	return value, nil
}

func referenceMetricRound(value float64) float64 {
	rounded, _ := strconv.ParseFloat(fmt.Sprintf("%.6g", value), 64)
	return rounded
}

func TestParseMetricValueSampleRows(t *testing.T) {
	const output = `
                               Range Name                                                                                          Metric Name                                                                                          Metric Value
----------------------------------------------------------------------------------------------------------------------------------------------------------------
VectorAddSubtract                           smsp__warps_launched.avg                                                                                                1.23457e+06
`
	got, err := parseMetricValue(output, "smsp__warps_launched.avg")
	if err != nil || got != 1.23457e6 {
		t.Fatalf("parseMetricValue = %g, %v", got, err)
	}
	if referenceMetricRound(1234567) != got {
		t.Fatalf("reference rounding mismatch: rounded=%g parsed=%g", referenceMetricRound(1234567), got)
	}
}

func TestParseMetricValueRejectsMissingAndMultipleRows(t *testing.T) {
	for _, output := range []string{
		"VectorAdd smsp__warps_launched.avg\n123.0\n",
		"RangeA smsp__warps_launched.avg 1.0\nRangeB smsp__warps_launched.avg 2.0\n",
	} {
		if _, err := parseMetricValue(output, "smsp__warps_launched.avg"); err == nil {
			t.Fatalf("parseMetricValue accepted ambiguous or missing-value output: %q", output)
		}
	}
}
