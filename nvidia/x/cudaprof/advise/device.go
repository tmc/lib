package advise

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// An smCount is the SM count of the device a capture ran on and where it
// came from. The source matters: an SM count the device reported is a fact
// about that run, while a datasheet figure is a fact about a product the run
// may or may not have been, and the arithmetic built on either reads the
// same.
type smCount struct {
	n int

	// source is empty when the capture recorded the count, and otherwise
	// names the profile it was read from, such as "the B200 datasheet".
	source string

	// datasheet reports that n is not from a device: an [cudadevicetable.Unverified]
	// figure, which findings built on it state and are weighed as suspected.
	datasheet bool
}

// determineSMCount returns the device SM count recorded in the capture or
// implied by the device it names. It returns ok=false when neither says.
//
// An architecture alone does not imply an SM count. sm_90 is a 132-SM H100
// SXM and a 114-SM H100 PCIe, and sm_103 is whatever GB300 or B300 enables;
// the count belongs to a product, and a capture that names only the
// instruction set has not said which product ran it. The one exception is an
// architecture with a single known product whose count was probed.
func determineSMCount(cap *cudacapture.Capture) (int, bool) {
	c, ok := captureSMCount(cap)
	return c.n, ok
}

func captureSMCount(cap *cudacapture.Capture) (smCount, bool) {
	if cap == nil {
		return smCount{}, false
	}
	if n := parseSMCountFromCapture(cap); n > 0 {
		return smCount{n: n}, true
	}
	if p, ok := productProfile(captureDeviceName(cap)); ok {
		return profileSMCount(p)
	}
	arch := parseArchFromCapture(cap)
	if p, ok := productProfile(arch); ok {
		return profileSMCount(p)
	}
	if p, ok := soleProbedProfile(arch); ok {
		return profileSMCount(p)
	}
	return smCount{}, false
}

func profileSMCount(p cudadevicetable.Profile) (smCount, bool) {
	if p.SMs.Value <= 0 {
		return smCount{}, false
	}
	if p.SMs.Source == cudadevicetable.Probed {
		return smCount{n: int(p.SMs.Value), source: "the probed " + p.Product}, true
	}
	return smCount{n: int(p.SMs.Value), source: "the " + p.Product + " datasheet", datasheet: true}, true
}

// describe returns the count with its source, for a finding's text.
func (c smCount) describe() string {
	if c.source == "" {
		return fmt.Sprintf("%d SMs", c.n)
	}
	return fmt.Sprintf("%d SMs per %s, not recorded in this capture", c.n, c.source)
}

// productProfile returns the datasheet profile of the product name names,
// such as "NVIDIA GB300" or "NVIDIA H100 80GB HBM3". The longest product
// name that appears as a word wins, so "GB200" is not read as "B200".
//
// A PCIe or NVL part is refused: the datasheet lists the SXM parts, and the
// PCIe parts of the same name enable fewer SMs.
func productProfile(name string) (cudadevicetable.Profile, bool) {
	s := strings.ToLower(name)
	if s == "" || strings.Contains(s, "pcie") || strings.Contains(s, " nvl") {
		return cudadevicetable.Profile{}, false
	}
	names := cudadevicetable.ProfileNames()
	slices.SortFunc(names, func(a, b string) int { return len(b) - len(a) })
	for _, n := range names {
		if strings.HasPrefix(n, "sm_") || !containsWord(s, strings.ToLower(n)) {
			continue
		}
		p, err := cudadevicetable.LookupProfile(n)
		if err == nil && strings.EqualFold(p.Product, n) {
			return p, true
		}
	}
	return cudadevicetable.Profile{}, false
}

func containsWord(s, word string) bool {
	for i := 0; ; {
		j := strings.Index(s[i:], word)
		if j < 0 {
			return false
		}
		start, end := i+j, i+j+len(word)
		if (start == 0 || !isAlnum(s[start-1])) && (end == len(s) || !isAlnum(s[end])) {
			return true
		}
		i = start + 1
	}
}

func isAlnum(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= '0' && b <= '9'
}

// soleProbedProfile returns the profile of arch's only listed product, when
// that product's SM count was probed.
func soleProbedProfile(arch string) (cudadevicetable.Profile, bool) {
	base := cudadevicetable.Normalize(arch)
	if base == "" {
		return cudadevicetable.Profile{}, false
	}
	var found []cudadevicetable.Profile
	for _, n := range cudadevicetable.ProfileNames() {
		p, err := cudadevicetable.LookupProfile(n)
		if err != nil || !strings.EqualFold(p.Product, n) || cudadevicetable.Normalize(p.Target) != base {
			continue
		}
		found = append(found, p)
	}
	if len(found) != 1 || found[0].SMs.Source != cudadevicetable.Probed {
		return cudadevicetable.Profile{}, false
	}
	return found[0], true
}

// captureDeviceName returns the device name the capture records, or "".
func captureDeviceName(cap *cudacapture.Capture) string {
	if cap == nil {
		return ""
	}
	sources := [][]byte{cap.Metadata}
	for _, name := range captureProvenanceFiles {
		sources = append(sources, cap.Files[name])
	}
	for _, data := range sources {
		var m map[string]any
		if len(data) == 0 || json.Unmarshal(data, &m) != nil {
			continue
		}
		if s := findStringInMap(m, "device", "device_name", "deviceName", "gpu", "product"); s != "" {
			return s
		}
	}
	if cap.Profile != nil {
		for _, c := range cap.Profile.Comments {
			if _, ok := productProfile(c); ok {
				return c
			}
		}
	}
	return ""
}

// captureProvenanceFiles are the capture files that may record device facts.
var captureProvenanceFiles = []string{"capture-provenance.json", "capture-status.json", "status.json", "manifest.json", "provenance.json"}

// findStringInMap returns the first non-empty string under one of keys in m
// or, failing that, in a map nested in m.
func findStringInMap(m map[string]any, keys ...string) string {
	for _, k := range keys {
		if s, ok := m[k].(string); ok && s != "" {
			return s
		}
	}
	for _, v := range m {
		if sub, ok := v.(map[string]any); ok {
			if s := findStringInMap(sub, keys...); s != "" {
				return s
			}
		}
	}
	return ""
}

func findSMCountInMap(m map[string]any) int {
	for _, k := range []string{"sm_count", "smCount", "multiprocessor_count", "multiprocessorCount", "sms", "sm_num"} {
		switch v := m[k].(type) {
		case float64:
			if v > 0 {
				return int(v)
			}
		case string:
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				return n
			}
		}
	}
	for _, v := range m {
		if sub, ok := v.(map[string]any); ok {
			if n := findSMCountInMap(sub); n > 0 {
				return n
			}
		}
	}
	return 0
}
