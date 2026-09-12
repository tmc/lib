package cuptipcsampling

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func testSamplingOptions() (samplingOptions, ReasonInventory) {
	return samplingOptions{period: 10, workerSleepMS: 20, mode: cupti.PcSamplingCollectionModeContinuous, scratchBytes: 1024, hardwareBytes: 2048, maxNativeBytes: 3072, reasons: []uint32{9, 77}}, ReasonInventory{CUPTIVersion: 130001, Reasons: []StallReason{{Index: 9, Name: "reason9"}, {Index: 77, Name: "reason77"}}}
}

func TestSamplingConfig(t *testing.T) {
	b := testSamplingBuffer(t)
	o, inventory := testSamplingOptions()
	c, err := newSamplingConfig(o, inventory, b)
	if err != nil {
		t.Fatal(err)
	}
	defer c.release()
	o.reasons[0] = 999
	if c.reasons[0] != 9 {
		t.Fatal("borrowed selected reasons")
	}
	err = c.apply(5, func(p *cupti.PCSamplingConfigurationInfoParams) error {
		if p.Size != 40 || p.Ctx != 5 || p.NumAttributes != 9 {
			t.Fatalf("params=%+v", p)
		}
		want := []uint64{10, 2, 1024, 2048, 1, 1, 1, c.attrs[7].AttributeData[0], 20}
		for i := range c.attrs {
			if c.attrs[i].AttributeData[0] != want[i] || c.attrs[i].AttributeStatus != -1 {
				t.Fatalf("attribute %d=%+v", i, c.attrs[i])
			}
			c.attrs[i].AttributeStatus = 0
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.apply(5, func(*cupti.PCSamplingConfigurationInfoParams) error { t.Fatal("reapplied configuration"); return nil }); err == nil {
		t.Fatal("repeat accepted")
	}
}

func TestSamplingConfigRejectsOptions(t *testing.T) {
	for _, change := range []func(*samplingOptions){
		func(o *samplingOptions) { o.period = 4 }, func(o *samplingOptions) { o.period = 32 },
		func(o *samplingOptions) { o.workerSleepMS = 0 }, func(o *samplingOptions) { o.maxNativeBytes-- },
		func(o *samplingOptions) { o.hardwareBytes = ^uint64(0) }, func(o *samplingOptions) { o.scratchBytes = 0 },
		func(o *samplingOptions) { o.reasons = []uint32{9, 9} }, func(o *samplingOptions) { o.reasons = []uint32{8} },
		func(o *samplingOptions) { o.mode = cupti.PcSamplingCollectionModeKernelSerialized },
	} {
		o, inventory := testSamplingOptions()
		change(&o)
		if c, err := newSamplingConfig(o, inventory, testSamplingBuffer(t)); err == nil {
			c.release()
			t.Fatalf("accepted %+v", o)
		}
	}
}

func TestSamplingConfigFailure(t *testing.T) {
	failure := errors.New("native failure")
	for _, name := range []string{"native", "unwritten status", "status", "pointer", "attribute", "reasons"} {
		t.Run(name, func(t *testing.T) {
			o, inventory := testSamplingOptions()
			c, err := newSamplingConfig(o, inventory, testSamplingBuffer(t))
			if err != nil {
				t.Fatal(err)
			}
			defer c.release()
			err = c.apply(5, func(p *cupti.PCSamplingConfigurationInfoParams) error {
				if name == "native" {
					return failure
				}
				if name == "unwritten status" {
					return nil
				}
				for i := range c.attrs {
					c.attrs[i].AttributeStatus = 0
				}
				switch name {
				case "status":
					c.attrs[2].AttributeStatus = 1
				case "pointer":
					p.PPCSamplingConfigurationInfo++
				case "attribute":
					c.attrs[0].AttributeData[0]++
				case "reasons":
					c.reasons[0]++
				}
				return nil
			})
			if err == nil || !c.attempted || c.released {
				t.Fatalf("failed configuration not retained: %v", err)
			}
			if name == "native" && !errors.Is(err, failure) {
				t.Fatal("lost native error")
			}
		})
	}
}
