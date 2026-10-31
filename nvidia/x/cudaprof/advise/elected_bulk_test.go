package advise

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Retained sm90a compiler-control PCs, not a fabricated native capture.
const electedBulkWitness = `
code for sm_90a
Function : k
/*9d20*/ @P0 ELECT P2, URZ, PT ;
/*9d30*/ @P2 R2UR UR20, R4 ;
/*9d40*/ @P2 R2UR UR21, R5 ;
/*9d50*/ @P2 R2UR UR16, R20 ;
/*9d60*/ @P2 PLOP3.LUT P0, PT, P2, PT, PT, 0x8, 0x0 ;
/*9d70*/ PLOP3.LUT P2, PT, PT, PT, PT, 0x8, 0x0 ;
/*9d80*/ UBLKCP.S.G [UR16], [UR20], UR12 ;
/*9d90*/ @P0 BRA.U.ANY 0x9d20 ;
`

// Retained uniform-address bulk-copy loop: no lane address gathers occur
// inside the elected loop. This must not be diagnosed as lane uniformization.
const uniformBulkWitness = `
code for sm_90a
Function : k
/*a8e0*/ @P0 ELECT P2, URZ, PT ;
/*a8f0*/ UMOV UR15, UR17 ;
/*a900*/ UBLKCP.S.G [UR14], [UR12], UR16 ;
/*a910*/ @P2 PLOP3.LUT P0, PT, P2, PT, PT, 0x8, 0x0 ;
/*a920*/ PLOP3.LUT P2, PT, PT, PT, PT, 0x8, 0x0 ;
/*a930*/ @P0 BRA.U.ANY 0xa8e0 ;
`

func TestElectedBulkPatternControls(t *testing.T) {
	for _, tt := range []struct {
		name, text string
		want       int
	}{
		{"retained", electedBulkWitness, 1},
		{"uniform bulk copy", uniformBulkWitness, 0},
		{"unrelated elect", strings.ReplaceAll(electedBulkWitness, "BRA.U.ANY 0x9d20", "BRA.U.ANY 0x9d30"), 0},
		{"no copy", strings.ReplaceAll(electedBulkWitness, "UBLKCP.S.G", "NOP"), 0},
		{"no backedge", strings.ReplaceAll(electedBulkWitness, "BRA.U.ANY 0x9d20", "BRA.U.ANY 0x9da0"), 0},
		{"irrelevant gathers", strings.ReplaceAll(electedBulkWitness, "[UR16], [UR20]", "[UR18], [UR22]"), 0},
		{"size gather is not address gather", strings.ReplaceAll(electedBulkWitness, "R2UR UR16, R20", "R2UR UR12, R20"), 0},
		{"clobbered address", strings.ReplaceAll(electedBulkWitness, "PLOP3.LUT P2, PT, PT, PT, PT, 0x8, 0x0", "UMOV UR20, UR22"), 0},
		{"complex unrelated control", strings.ReplaceAll(electedBulkWitness, "PLOP3.LUT P2, PT, PT, PT, PT, 0x8, 0x0", "BRA 0x9d90"), 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.ParseString(tt.text)
			if err != nil {
				t.Fatal(err)
			}
			if got := len(electedBulkLoops(d.Functions[0])); got != tt.want {
				t.Fatalf("got %d, want %d", got, tt.want)
			}
		})
	}
}

func electedBulkEnvelope(t *testing.T, text string) *cudacapture.Capture {
	t.Helper()
	cap := &cudacapture.Capture{
		Profile:         &profile.Profile{SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}}},
		Code:            cudacapture.CodeFiles{SASS: "kernel.sass"},
		CodeUnavailable: map[string]string{"source": "compiler witness only", "ptx": "not retained", "cubin": "not retained"},
		Files:           map[string][]byte{"kernel.sass": []byte(text)},
	}
	var b bytes.Buffer
	if err := cudacapture.Write(&b, cap); err != nil {
		t.Fatal(err)
	}
	retained, err := cudacapture.Read(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	return retained
}

func TestElectedBulkAdvicePublicEnvelope(t *testing.T) {
	for _, tt := range []struct {
		name, text string
		want       int
	}{
		{"sm90a", electedBulkWitness, 1},
		{"unknown target", strings.ReplaceAll(electedBulkWitness, "code for sm_90a", ""), 0},
		{"untested architecture", strings.ReplaceAll(electedBulkWitness, "sm_90a", "sm_100a"), 0},
		{"uniform", uniformBulkWitness, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			report, err := Analyze(electedBulkEnvelope(t, tt.text))
			if err != nil {
				t.Fatal(err)
			}
			var found []Item
			for _, item := range report.Items {
				if item.rule == "elected-bulk-copy" {
					found = append(found, item)
				}
			}
			if len(found) != tt.want {
				t.Fatalf("got %d findings, want %d", len(found), tt.want)
			}
			for _, item := range found {
				if item.EvidenceLevel != EvidenceSuspected || item.Severity != SeverityContext || item.DeviceTimeNanos != 0 || item.ShareOfProfile != 0 || item.Gap != nil {
					t.Fatalf("static pattern promoted to runtime/cost claim: %+v", item)
				}
				if item.ClassifiedShare == nil {
					t.Fatal("missing explicit unread share")
				}
				data, err := json.Marshal(item.ClassifiedShare)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Contains(data, []byte(`"class":"unread"`)) {
					t.Fatalf("share not unread: %s", data)
				}
				if !strings.Contains(item.Finding, "sm_90a") || !strings.Contains(item.Finding, "neither executed lanes nor latency cost") {
					t.Fatalf("missing architecture/evidence qualification: %s", item.Finding)
				}
			}
		})
	}
}

func TestElectedBulkKeepsFunctionsSeparate(t *testing.T) {
	d, err := cudasass.ParseString(electedBulkWitness + strings.ReplaceAll(electedBulkWitness, "Function : k", "Function : other"))
	if err != nil {
		t.Fatal(err)
	}
	r := &Report{}
	auditElectedBulkCopies(r, d, nil)
	if len(r.Items) != 2 {
		t.Fatalf("got %d findings, want 2", len(r.Items))
	}
}
