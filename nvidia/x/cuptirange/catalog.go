package cuptirange

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Catalog owns a native metric-discovery host without a metric configuration.
// It cannot capture or replay workloads. Its zero value is not usable.
// A Catalog must not be copied, and Close must succeed before discarding it.
type Catalog struct {
	plan    *Plan
	version uint32
}

// NewCatalog opens the Range Profiler's metric catalog for chip and an optional
// counter availability image. No metric names or configuration are required.
// Versions 24, 26, 28, 130001, 130101, 130200, 130202, 130301 and 130400 have
// measured host layouts; others fail closed.
// A nonnil Catalog accompanying an error retains cleanup ownership: retry Close.
func NewCatalog(chip string, availability []byte) (*Catalog, error) {
	if chip == "" || strings.ContainsRune(chip, 0) {
		return nil, errors.New("cuptirange: invalid chip")
	}
	version, err := cupti.GetVersion()
	if err != nil {
		return nil, fmt.Errorf("cuptirange: query CUPTI version: %w", err)
	}
	return newCatalog(chip, availability, version, hostAPI{initialize: cupti.ProfilerHostInitialize, deinitialize: cupti.ProfilerHostDeinitialize})
}

func newCatalog(chip string, availability []byte, version uint32, api hostAPI) (*Catalog, error) {
	if chip == "" || strings.ContainsRune(chip, 0) {
		return nil, errors.New("cuptirange: invalid chip")
	}
	if !qualifiedLayout(version) {
		return nil, fmt.Errorf("cuptirange: unqualified catalog version %d", version)
	}
	p := &Plan{api: api, chip: append([]byte(chip), 0), availability: append([]byte(nil), availability...)}
	c := &Catalog{plan: p, version: version}
	if err := p.initializeHost(version); err != nil {
		if cleanup := p.Close(); cleanup != nil {
			return c, errors.Join(err, cleanup)
		}
		return nil, err
	}
	return c, nil
}

func (p *Plan) initializeHost(version uint32) error {
	if unsafe.Sizeof(uintptr(0)) != 8 || !qualifiedLayout(version) {
		return fmt.Errorf("cuptirange: unqualified host initialization layout for CUPTI %d", version)
	}
	p.pin.Pin(&p.chip[0])
	v := cupti.ProfilerHostInitializeParams{ProfilerType: cupti.ProfilerTypeRangeProfiler, PChipName: uintptr(unsafe.Pointer(&p.chip[0]))}
	v.StructSize = uint(unsafe.Offsetof(v.PHostObject) + unsafe.Sizeof(v.PHostObject))
	if singlePassInitialize(version) {
		v.StructSize = uint(unsafe.Offsetof(v.PSinglePassMetricSetName) + unsafe.Sizeof(v.PSinglePassMetricSetName))
	}
	if len(p.availability) > 0 {
		p.pin.Pin(&p.availability[0])
		v.PCounterAvailabilityImage = uintptr(unsafe.Pointer(&p.availability[0]))
	}
	err := p.api.initialize(&v)
	p.host = v.PHostObject
	if err != nil {
		return fmt.Errorf("cuptirange: initialize host: %w", err)
	}
	if p.host == 0 {
		return errors.New("cuptirange: empty host object")
	}
	return nil
}

// Close releases the host and its retained inputs. A failed close is retryable.
func (c *Catalog) Close() error {
	if c == nil {
		return nil
	}
	return c.plan.Close()
}

// BaseMetrics queries owned native names; see Plan.BaseMetrics for bounds and
// the distinction between discovery and collection support.
func (c *Catalog) BaseMetrics(kind cupti.MetricType, maxNames, maxNameBytes int) ([]string, error) {
	if c == nil {
		return nil, errors.New("cuptirange: nil catalog")
	}
	return c.plan.BaseMetrics(kind, maxNames, maxNameBytes)
}

// SubMetrics queries native submetric names; see Plan.SubMetrics.
func (c *Catalog) SubMetrics(name string, kind cupti.MetricType, maxNames, maxNameBytes int) ([]string, error) {
	if c == nil {
		return nil, errors.New("cuptirange: nil catalog")
	}
	return c.plan.SubMetrics(name, kind, maxNames, maxNameBytes)
}

// Properties queries native metric definitions using the catalog's host ABI.
// See Plan.Properties for bounds and availability semantics.
func (c *Catalog) Properties(name string, maxStringBytes int) (MetricProperties, error) {
	if c == nil {
		return MetricProperties{}, errors.New("cuptirange: nil catalog")
	}
	return c.plan.properties(name, maxStringBytes, c.version, cupti.ProfilerHostGetMetricProperties)
}
