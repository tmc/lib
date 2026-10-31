package advise

import "github.com/tmc/lib/nvidia/x/cudaevidence"

const noRuntimeAttributionReason = "runtime attribution is not admitted for this finding"

// withoutRuntimeAttribution removes every numeric runtime channel from a
// finding whose evidence cannot support it. The value copy leaves the caller's
// fields and immutable evidence value unchanged.
func (item Item) withoutRuntimeAttribution(source *cudaevidence.Source) Item {
	if !item.NoRuntimeAttribution {
		return item
	}
	item.DeviceTimeNanos = 0
	item.HostTimeNanos = 0
	item.ShareOfProfile = 0
	item.Gap = nil
	item.terms = nil
	item.magnitude = ""
	src := cudaevidence.Source{Producer: "static-analysis", Locator: "compiler-artifacts"}
	if item.ClassifiedShare != nil {
		src = item.ClassifiedShare.Record().Source
	}
	if source != nil {
		src = *source
	}
	value, _ := cudaevidence.New(cudaevidence.Record{Unit: "percent", Source: src, Reason: noRuntimeAttributionReason})
	item.ClassifiedShare = &value
	return item
}

func clearNoRuntimeAttribution(items []Item, source *cudaevidence.Source) {
	for i := range items {
		items[i] = items[i].withoutRuntimeAttribution(source)
		clearNoRuntimeAttribution(items[i].Patterns, source)
	}
}
