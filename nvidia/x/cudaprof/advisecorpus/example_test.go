package advisecorpus_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
	"github.com/tmc/lib/nvidia/x/cudaprof/advisecorpus"
)

func exampleRecord() (string, advisecorpus.Record, func()) {
	root, err := os.MkdirTemp("", "advisecorpus-example-")
	if err != nil {
		panic(err)
	}
	cleanup := func() { os.RemoveAll(root) }
	refs := map[string]advisecorpus.Artifact{}
	for _, arm := range []string{"baseline", "candidate"} {
		raw := []byte("// CPU example " + arm)
		name := arm + ".cu"
		if err := os.WriteFile(filepath.Join(root, name), raw, 0600); err != nil {
			cleanup()
			panic(err)
		}
		sum := sha256.Sum256(raw)
		refs[arm] = advisecorpus.Artifact{Path: name, SHA256: hex.EncodeToString(sum[:]), Bytes: int64(len(raw))}
	}
	raw, _ := json.Marshal(map[string]any{"id": "cpu-example", "source": refs, "mechanism": "reduce producer issue work", "comparison": map[string]string{"boundary": "STATIC_ONLY"}, "observation": "STATIC_STOP_NO_NATIVE", "advise": map[string]string{"status": "NOT_EVALUATED"}})
	records, err := advisecorpus.Decode(strings.NewReader(string(raw)))
	if err != nil {
		cleanup()
		panic(err)
	}
	return root, records[0], cleanup
}

func Example() {
	root, record, cleanup := exampleRecord()
	defer cleanup()
	result, err := advisecorpus.Consume(root, record, nil)
	fmt.Println(result["classification"], err)
	// Output: NOT_EVALUATED <nil>
}

func ExampleArtifact() {
	raw := []byte("source bytes")
	sum := sha256.Sum256(raw)
	ref := advisecorpus.Artifact{Path: "kernel.cu", SHA256: hex.EncodeToString(sum[:]), Bytes: int64(len(raw))}
	fmt.Println(ref.Path, ref.Bytes)
	// Output: kernel.cu 12
}

func ExampleRecord() {
	record := advisecorpus.Record{"id": json.RawMessage(`"retained-id"`)}
	fmt.Printf("%s\n", record["id"])
	// Output: "retained-id"
}

func ExampleLink() {
	link := advisecorpus.Link{Schema: "corpus-advise-link/v1", Origin: "NONBLIND_CALIBRATION", Scope: "STATIC_ONLY"}
	fmt.Println(link.Origin, link.Scope)
	// Output: NONBLIND_CALIBRATION STATIC_ONLY
}

func ExampleResult() {
	root, record, cleanup := exampleRecord()
	defer cleanup()
	result, _ := advisecorpus.Consume(root, record, nil)
	fmt.Println(result["native_revalidation"], result["mechanism_causality"])
	// Output: NOT_RUN NOT_ESTABLISHED
}

func ExampleDecode() {
	root, record, cleanup := exampleRecord()
	defer cleanup()
	_ = root
	raw, _ := json.Marshal(record)
	records, err := advisecorpus.Decode(strings.NewReader(string(raw)))
	fmt.Println(len(records), err)
	// Output: 1 <nil>
}

func ExampleConsume() {
	root, record, cleanup := exampleRecord()
	defer cleanup()
	result, err := advisecorpus.Consume(root, record, nil)
	fmt.Println(result["classification"], result["blind_success_claim"], err)
	// Output: NOT_EVALUATED false <nil>
}

func ExampleAnalyze() {
	root, record, cleanup := exampleRecord()
	defer cleanup()
	// A selected arm and exact retained capture are required; no tool runs here.
	_, err := advisecorpus.Analyze(root, record, "unknown-arm", advisecorpus.Artifact{}, cudaevidence.Source{})
	fmt.Println(err)
	// Output: unknown input arm
}

func ExampleMatch() {
	report := &advise.Report{Items: []advise.Item{{Finding: "producer issue loop", ActionableRecommendation: "A/B schedule", Location: advise.Location{Kernel: "w13"}, EvidenceLevel: advise.EvidenceSuspected}}}
	indices, err := advisecorpus.Match(report, "producer issue loop", "A/B schedule", "w13")
	fmt.Println(indices, err)
	// Output: [0] <nil>
}

func ExampleDecodeLink() {
	link, err := advisecorpus.DecodeLink(strings.NewReader(`{"schema":"corpus-advise-link/v1","record_id":"retained-id"}`))
	fmt.Println(link.RecordID, err)
	// Output: retained-id <nil>
}
