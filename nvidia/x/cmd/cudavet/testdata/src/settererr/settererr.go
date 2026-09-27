package settererr

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func dropped(f driver.Function) {
	f.SetAttribute(0, 256<<20) // want `the error of SetAttribute is discarded; a rejected value leaves the old setting in place`
}

func limit() {
	driver.CtxSetLimit(driver.LimitPersistingL2CacheSize, 99<<20) // want `the error of CtxSetLimit is discarded; a rejected value leaves the old setting in place`
}

func checked(f driver.Function) error {
	if err := f.SetAttribute(0, 256<<20); err != nil {
		return err
	}
	return driver.CtxSetLimit(driver.LimitPersistingL2CacheSize, 99<<20)
}

func blank(f driver.Function) {
	_ = f.SetAttribute(0, 256<<20)
}

// Reset is not a setter: its Set is not followed by an upper-case letter.
func reset(d driver.Device) {
	d.PrimaryCtxReset()
}

// cudair is not a binding; its SetOutputs checks Go values.
func graph(n *cudair.Node) {
	cudair.NewGraph().SetOutputs(n)
}
