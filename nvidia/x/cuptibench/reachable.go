package cuptibench

import (
	"errors"
	"fmt"
	"sort"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// MetricInfo describes one metric a device offers. It is a definition, not a
// measurement: CUPTI listing a metric does not prove that collecting it is
// permitted, that it schedules with other metrics, or that it fits a pass
// budget. Additive follows the rule in additiveMetric, so a listed metric that
// is not additive must not be summed across ranges.
type MetricInfo struct {
	Name        string
	Base        string
	Unit        string
	Description string
	Type        cupti.MetricType
	Additive    bool
}

// reachableBounds keeps the native inventory queries bounded. A chip offering
// more than this fails rather than returning a partial inventory that would
// read as a complete one.
const (
	maxMetricNames = 1 << 16
	maxMetricBytes = 512
)

// Reachable lists every metric dev's chip offers, sorted by name. It expands
// each base metric into the submetrics CUPTI reports for it rather than
// appending rollup suffixes, because not every base supports the same ones.
//
// The result is an inventory for choosing metrics, not a collectable set:
// asking for all of them at once would exceed any sane pass budget, since each
// scheduling group costs a replay pass. Pass the names you want to Config, or
// name them through Resolve, and let New report the pass count.
func Reachable(dev *cuda.Device) ([]MetricInfo, error) {
	if dev == nil {
		return nil, errors.New("cuptibench: nil device")
	}
	var ctx *cuda.Context
	if err := dev.WithContext(func() error {
		c, e := dev.PrimaryContext()
		if e != nil {
			return e
		}
		ctx = c
		return nil
	}); err != nil {
		return nil, err
	}
	var out []MetricInfo
	err := ctx.Run(func() error {
		chip, err := cuptisass.DeviceChipName(uint32(dev.Ordinal()))
		if err != nil {
			return err
		}
		availability, err := cuptirange.CounterAvailability(dev, 1<<20)
		if err != nil {
			return err
		}
		catalog, err := cuptirange.NewCatalog(chip, availability)
		if err != nil {
			return err
		}
		defer func() {
			if cleanup := catalog.Close(); cleanup != nil && err == nil {
				err = cleanup
			}
		}()
		out, err = inventory(catalog)
		return err
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// lister is the part of cuptirange.Catalog inventory needs, so the walk can be
// tested without a device.
type lister interface {
	BaseMetrics(kind cupti.MetricType, maxNames, maxNameBytes int) ([]string, error)
	SubMetrics(name string, kind cupti.MetricType, maxNames, maxNameBytes int) ([]string, error)
	Properties(name string, maxStringBytes int) (cuptirange.MetricProperties, error)
}

func inventory(catalog lister) ([]MetricInfo, error) {
	var out []MetricInfo
	seen := make(map[string]bool)
	for kind := cupti.MetricTypeCounter; kind < cupti.MetricTypeCount; kind++ {
		base, err := catalog.BaseMetrics(kind, maxMetricNames, maxMetricBytes)
		if err != nil {
			return nil, fmt.Errorf("cuptibench: list base metrics for type %d: %w", kind, err)
		}
		for _, b := range base {
			subs, err := catalog.SubMetrics(b, kind, maxMetricNames, maxMetricBytes)
			if err != nil {
				return nil, fmt.Errorf("cuptibench: list submetrics of %q: %w", b, err)
			}
			for _, sub := range subs {
				name := qualify(b, sub)
				if seen[name] {
					continue
				}
				seen[name] = true
				prop, err := catalog.Properties(name, 1<<20)
				if err != nil {
					return nil, fmt.Errorf("cuptibench: metric properties for %q: %w", name, err)
				}
				unit := prop.Dimension
				if unit == "" {
					unit = "unknown"
				}
				out = append(out, MetricInfo{
					Name: name, Base: b, Unit: unit, Description: prop.Description,
					Type: prop.Type, Additive: additiveMetric(name, unit, prop.Type),
				})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// qualify joins a base metric and one of its submetrics. CUPTI reports
// submetrics either fully qualified or as the trailing rollup alone, so this
// uses what was returned instead of assuming one form.
func qualify(base, sub string) string {
	if sub == "" {
		return base
	}
	if len(sub) > len(base) && sub[:len(base)] == base {
		return sub
	}
	if sub[0] == '.' {
		return base + sub
	}
	return base + "." + sub
}
