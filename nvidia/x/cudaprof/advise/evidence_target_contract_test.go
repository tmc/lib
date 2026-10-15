package advise

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func targetPTX(target string) []byte {
	return []byte(".version 7.0\n.target " + target + "\n.address_size 64\n.visible .entry kernel() { ret; }\n")
}

func TestAnalyzeEvidenceAllRetainedTargets(t *testing.T) {
	for _, change := range []struct {
		name  string
		apply func(*cudacapture.Capture)
	}{
		{"metadata", func(c *cudacapture.Capture) {
			c.Metadata = json.RawMessage(`{"compute_capability":"sm_90","sm_count":80}`)
		}},
		{"metadata-key-order-a", func(c *cudacapture.Capture) {
			c.Metadata = json.RawMessage(`{"target":"sm_70","architecture":"sm_90","sm_count":80}`)
		}},
		{"metadata-key-order-b", func(c *cudacapture.Capture) {
			c.Metadata = json.RawMessage(`{"architecture":"sm_90","target":"sm_70","sm_count":80}`)
		}},
		{"nested-array-a", func(c *cudacapture.Capture) {
			c.Metadata = json.RawMessage(`{"compute_capability":"sm_70","devices":[{"target":"sm_70"},{"target":"sm_90"}],"sm_count":80}`)
		}},
		{"nested-array-b", func(c *cudacapture.Capture) {
			c.Metadata = json.RawMessage(`{"compute_capability":"sm_70","devices":[{"target":"sm_90"},{"target":"sm_70"}],"sm_count":80}`)
		}},
		{"capture-provenance", func(c *cudacapture.Capture) { c.Files["capture-provenance.json"] = []byte(`{"target":"sm_90"}`) }},
		{"capture-status", func(c *cudacapture.Capture) {
			c.Files["capture-status.json"] = []byte(`{"status":{"capture_success":true,"run_completeness":"complete","dropped_records":0,"target":"sm_90"}}`)
		}},
		{"status", func(c *cudacapture.Capture) { c.Files["status.json"] = []byte(`{"architecture":"sm_90"}`) }},
		{"manifest", func(c *cudacapture.Capture) {
			c.Files["manifest.json"] = []byte(`{"identity":{"compute_cap":"sm_90"}}`)
		}},
		{"provenance", func(c *cudacapture.Capture) { c.Files["provenance.json"] = []byte(`{"target":"sm_90"}`) }},
		{"secondary-ptx", func(c *cudacapture.Capture) { c.Files["secondary.ptx"] = targetPTX("sm_90") }},
		{"duplicate-ptx-a", func(c *cudacapture.Capture) {
			c.Files["secondary.ptx"] = append([]byte(".target sm_90\n"), targetPTX("sm_70")...)
		}},
		{"duplicate-ptx-b", func(c *cudacapture.Capture) {
			c.Files["secondary.ptx"] = append([]byte(".target sm_70\n"), targetPTX("sm_90")...)
		}},
		{"module-role", func(c *cudacapture.Capture) {
			c.Modules = []cudacapture.Module{{ID: "m", Code: cudacapture.CodeFiles{PTX: "opaque"}}}
			c.Files["opaque"] = targetPTX("sm_90")
		}},
		{"sass", func(c *cudacapture.Capture) { c.Files["secondary.sass"] = []byte("code for sm_90\n") }},
		{"conflicting-sass", func(c *cudacapture.Capture) { c.Files["secondary.sass"] = []byte("code for sm_70\ncode for sm_90\n") }},
		{"unknown", func(c *cudacapture.Capture) {
			c.Metadata = json.RawMessage(`{"compute_capability":"unknown","sm_count":80}`)
		}},
		{"unknown-number", func(c *cudacapture.Capture) { c.Metadata = json.RawMessage(`{"compute_capability":70,"sm_count":80}`) }},
		{"duplicate-metadata", func(c *cudacapture.Capture) {
			c.Metadata = json.RawMessage(`{"compute_capability":"sm_90","compute_capability":"sm_70","sm_count":80}`)
		}},
		{"identity-suffix", func(c *cudacapture.Capture) { c.Identity.ComputeCap = "sm_70a" }},
	} {
		t.Run(change.name, func(t *testing.T) {
			for _, matchingPTX := range []bool{false, true} {
				c, s := consumerCapture(25)
				change.apply(c)
				if matchingPTX {
					c.Code.PTX = "preferred.ptx"
					c.Files[c.Code.PTX] = targetPTX(s.Identity.Target)
				}
				record := consumerFinding(t, c, &s).ClassifiedShare.Record()
				if record.Class != cudaevidence.Unread || record.Number != nil || !strings.Contains(record.Reason, "target") {
					t.Fatalf("matchingPTX=%v: contradictory/unknown target admitted: %+v", matchingPTX, record)
				}
			}
		})
	}
}

func TestAnalyzeEvidenceTargetAliasesAndUnknowns(t *testing.T) {
	for _, target := range []string{"sm_70", " SM_70 ", "sm70", "compute_70"} {
		t.Run(target, func(t *testing.T) {
			c, s := consumerCapture(25)
			c.Identity.ComputeCap = target
			c.Metadata, _ = json.Marshal(map[string]any{"architecture": target, "sm_count": 80})
			c.Code.PTX = "preferred.ptx"
			c.Files[c.Code.PTX] = targetPTX("sm_70, debug")
			record := consumerFinding(t, c, &s).ClassifiedShare.Record()
			if record.Class != cudaevidence.Derived || record.Number == nil || *record.Number != 20 {
				t.Fatalf("compatible target refused: %+v", record)
			}
		})
	}
	for _, suffix := range []string{"a", "f"} {
		c, s := consumerCapture(25)
		s.Identity.Target = "sm_70" + suffix
		c.Identity.ComputeCap = s.Identity.Target
		c.Metadata, _ = json.Marshal(map[string]any{"compute_capability": s.Identity.Target, "sm_count": 80})
		c.Code.PTX = "preferred.ptx"
		c.Files[c.Code.PTX] = targetPTX(s.Identity.Target)
		record := consumerFinding(t, c, &s).ClassifiedShare.Record()
		if record.Class != cudaevidence.Derived {
			t.Fatalf("matching declared suffix lost: %+v", record)
		}
		c.Metadata = json.RawMessage(`{"compute_capability":"sm_70","sm_count":80}`)
		record = consumerFinding(t, c, &s).ClassifiedShare.Record()
		if record.Class != cudaevidence.Unread || record.Number != nil {
			t.Fatalf("feature suffix silently erased: %+v", record)
		}
	}
	for _, target := range []string{"unknown", "sm_70?", "sm_70_a", "sm_compute_70"} {
		c, s := consumerCapture(25)
		s.Identity.Target = target
		c.Identity.ComputeCap = target
		c.Metadata, _ = json.Marshal(map[string]any{"compute_capability": target, "sm_count": 80})
		record := consumerFinding(t, c, &s).ClassifiedShare.Record()
		if record.Class != cudaevidence.Unread || record.Number != nil {
			t.Fatalf("unknown source target manufactured: %+v", record)
		}
	}

}

// Exercise ELF target interpretation directly in the host-only admission helper.
// Passing these bytes through the legacy analyzer could invoke nvdisasm; this
// control deliberately never invokes any disassembler or device operation.
func TestEvidenceTargetCUBINBaseAndUnknown(t *testing.T) {
	for _, tc := range []struct {
		name   string
		abi    byte
		sm     uint32
		source string
		want   bool
	}{
		{"matching", 8, 70, "sm_70", true},
		{"conflict", 8, 90, "sm_70", false},
		{"known-base-unknown-suffix", 8, 121, "sm_121a", true},
		{"unknown-ABI-no-target-fact", 9, 70, "sm_70", true},
		{"unknown-SM-no-target-fact", 8, 0, "sm_70", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, s := consumerCapture(25)
			s.Identity.Target = tc.source
			c.Identity.ComputeCap = tc.source
			c.Metadata, _ = json.Marshal(map[string]any{"compute_capability": tc.source, "sm_count": 80})
			var ident [16]byte
			copy(ident[:], "\x7fELF")
			ident[elf.EI_CLASS] = byte(elf.ELFCLASS64)
			ident[elf.EI_DATA] = byte(elf.ELFDATA2LSB)
			ident[elf.EI_VERSION] = byte(elf.EV_CURRENT)
			ident[elf.EI_OSABI] = 0x41
			ident[elf.EI_ABIVERSION] = tc.abi
			header := elf.Header64{Ident: ident, Type: uint16(elf.ET_EXEC), Machine: uint16(elf.EM_CUDA), Version: uint32(elf.EV_CURRENT), Flags: tc.sm << 8, Ehsize: 64}
			var b bytes.Buffer
			if err := binary.Write(&b, binary.LittleEndian, header); err != nil {
				t.Fatal(err)
			}
			c.Code.CUBIN = "host-header"
			c.Files[c.Code.CUBIN] = b.Bytes()
			reason := profileShareTargets(c, s.Identity.Target)
			if (reason == "") != tc.want {
				t.Fatalf("target admission=%q, want supported=%v", reason, tc.want)
			}
		})
	}
}
