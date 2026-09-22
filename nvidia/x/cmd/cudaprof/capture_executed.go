package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
)

// executedType is the dimension exact executed-instruction counts are
// counted in, next to the sampled dimension over the same instructions. A
// sample says where warps were when the hardware looked; this says what the
// hardware ran, so a line whose sample count is low because its instructions
// are cheap reads differently from one that ran rarely.
const executedType = "executed"

// projectExecuted adds the counts the target published to the capture's
// profile, with the stack shape [projectPCSamples] writes.
//
// The counts name their module by the SHA-256 of the image they were measured
// against, and the capture stores every image it saw, so the join is exact:
// two builds of one kernel are two images, and counts of one of them are not
// charged to the other. Counts whose image the capture did not keep are
// dropped rather than charged to a module that might not be theirs.
func projectExecuted(c *cudacapture.Capture, attachments map[string][]byte) {
	if c.Profile == nil {
		return
	}
	byDigest := map[string]uint32{}
	for _, m := range c.Modules {
		if m.DriverID == 0 || m.Code.CUBIN == "" {
			continue
		}
		image, ok := attachments[m.Code.CUBIN]
		if !ok {
			continue
		}
		// One image loaded twice is two modules of identical bytes, and
		// counts of those bytes belong to either. They are charged to the
		// first, which at least makes the choice a stated one: charging
		// both would count the same instructions twice.
		sum := sha256.Sum256(image)
		if digest := hex.EncodeToString(sum[:]); byDigest[digest] == 0 {
			byDigest[digest] = m.DriverID
		}
	}
	var names []string
	for name := range attachments {
		if strings.HasPrefix(name, "count-") && strings.HasSuffix(name, ".json") {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	var counts []deviceCount
	for _, name := range names {
		var published cudaprofagent.Counts
		if err := json.Unmarshal(attachments[name], &published); err != nil {
			continue
		}
		id, ok := byDigest[published.Module]
		if !ok {
			continue
		}
		for _, count := range published.Counts {
			counts = append(counts, deviceCount{
				ModuleID: id,
				Function: count.Function,
				Offset:   count.Offset,
				Value:    count.Value,
			})
		}
	}
	projectCounts(c, executedType, "instructions", counts, attachments)
}
