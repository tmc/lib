package main

import (
	"encoding/json"
	"fmt"
	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/internal/cudaview"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func buildPipelineViewData(cap *cudacapture.Capture, name, kernel string) (*cudaview.PipelineViewData, error) {
	if len(cap.Modules) == 0 {
		return buildSinglePipelineViewData(cap, name, kernel)
	}
	var root *cudaview.PipelineViewData
	for _, mod := range cap.Modules {
		local := *cap
		local.Code = mod.Code
		local.Modules = nil
		// A module name or a matching PC is insufficient to attribute observations.
		// Keep only explicitly identified samples when multiple modules are present.
		if len(cap.Modules) > 1 {
			local.Profile = &profile.Profile{}
			if cap.Profile != nil {
				local.Profile = cap.Profile.Copy()
				samples := local.Profile.Sample
				local.Profile.Sample = nil
				for _, sample := range samples {
					if ids := sample.Label["module_id"]; len(ids) == 1 && ids[0] == moduleLabel(mod) {
						local.Profile.Sample = append(local.Profile.Sample, sample)
					}
				}
			}
		}
		data, err := buildSinglePipelineViewData(&local, name, kernel)
		if err != nil {
			return nil, fmt.Errorf("module %s: %w", mod.ID, err)
		}
		if len(cap.Modules) > 1 && cap.Profile != nil && len(local.Profile.Sample) < len(cap.Profile.Sample) {
			data.Diagnostics = append(data.Diagnostics, fmt.Sprintf("%d profile records lack an explicit identity for this module and are not overlaid", len(cap.Profile.Sample)-len(local.Profile.Sample)))
		}
		if root == nil {
			root = data
		}
		root.Modules = append(root.Modules, cudaview.PipelineModule{ID: mod.ID, Name: mod.Name, Data: data})
	}
	root.Recompile = recompileSources(cap, root.Architecture, recordedModuleOptions(cap))
	return root, nil
}

func recordedModuleOptions(cap *cudacapture.Capture) map[string][]string {
	var meta struct {
		Options map[string][]string `json:"nvrtc_options"`
	}
	_ = json.Unmarshal(cap.Metadata, &meta)
	return meta.Options
}
