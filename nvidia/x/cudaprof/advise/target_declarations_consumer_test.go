package advise

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func TestAnalyzeEvidenceParsedTargetDeclarations(t *testing.T) {
	for _, prefix := range []struct{ name, text string }{
		{"line-comment", "// previous .target is intentionally not a declaration\n"},
		{"block-comment", "/* .target sm_90\n.target sm_80 */\n"},
		{"file", ".file 1 \"/.target sm_90; // comment-like.cu\"\n"},
		{"file-continued", `.file 1 "prefix\
.target sm_90
suffix\" // .target sm_80"
`},
		{"pragma", ".pragma \".target sm_90; phantom: .entry fake() {}\";\n"},
		{"pragma-continued", `.pragma "prefix\
.target sm_90
suffix\"; phantom: .entry fake() {}";
`},
		{"opaque-source", ".nv_intermediate_source_section {\n.source_begin\n.target sm_90\n.source_end\n}\n"},
	} {
		for _, target := range []string{"sm_70", "sm70", "compute_70", "sm_70a", "sm_70f"} {
			t.Run(prefix.name+"/"+target, func(t *testing.T) {
				c, s := consumerCapture(25)
				suffix := ""
				if strings.HasSuffix(target, "a") || strings.HasSuffix(target, "f") {
					suffix = target[len(target)-1:]
				}
				s.Identity.Target = "sm_70" + suffix
				c.Identity.ComputeCap = s.Identity.Target
				c.Metadata, _ = json.Marshal(map[string]any{"compute_capability": s.Identity.Target, "sm_count": 80})
				c.Files["secondary.ptx"] = append([]byte(prefix.text), targetPTX(target+", debug // .target sm_90")...)
				// consumerFinding uses public AnalyzeWithEvidence and Report.JSON.
				r := consumerFinding(t, c, &s).ClassifiedShare.Record()
				if r.Class != cudaevidence.Derived || r.Number == nil || *r.Number != 20 || !strings.Contains(r.Basis, "not hardware authentication") {
					t.Fatalf("literal/comment text forced unread: %+v", r)
				}
				// A matching preferred listing cannot hide the secondary's fact.
				c.Code.PTX = "preferred.ptx"
				c.Files[c.Code.PTX] = targetPTX(s.Identity.Target)
				c.Files["secondary.ptx"] = append([]byte(prefix.text), targetPTX("sm_90")...)
				r = consumerFinding(t, c, &s).ClassifiedShare.Record()
				if r.Class != cudaevidence.Unread || r.Number != nil || !strings.Contains(r.Reason, "target") {
					t.Fatalf("real declaration contradiction lost: %+v", r)
				}
			})
		}
	}
}

func TestAnalyzeEvidenceMultipleActualTargetsRefused(t *testing.T) {
	for _, targets := range []string{".target sm_90\n.target sm_70\n", ".target sm_70\n.target sm_90\n", ".target sm_70\n.target sm_70\n"} {
		c, s := consumerCapture(25)
		c.Files["secondary.ptx"] = []byte(targets)
		r := consumerFinding(t, c, &s).ClassifiedShare.Record()
		if r.Class != cudaevidence.Unread || r.Number != nil || !strings.Contains(r.Reason, "declarations") {
			t.Fatalf("multiple actual declarations selected one target: %+v", r)
		}
	}
}

func TestAnalyzeEvidenceMissingActualTargetRefused(t *testing.T) {
	for _, text := range []string{"// .target sm_70\n.pragma \".target sm_70\";\n", ".file 1 \"/.target sm_70.cu\"\n", ".target\n"} {
		c, s := consumerCapture(25)
		c.Files["secondary.ptx"] = []byte(text)
		r := consumerFinding(t, c, &s).ClassifiedShare.Record()
		if r.Class != cudaevidence.Unread || r.Number != nil || !strings.Contains(r.Reason, "target") {
			t.Fatalf("missing target fabricated eligibility: %+v", r)
		}
	}
}
