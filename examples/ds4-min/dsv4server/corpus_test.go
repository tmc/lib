package dsv4server

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadCorpusCaseAndProvenance(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"metadata.json":     `{"case_id":"G2","step":"chat","method":"POST","endpoint":"/v1/chat/completions","status":"HTTP/1.1 200 OK","source_rev":"92069b8b4010c18b72812221198c225956f39a14","binary_sha256":"65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525","model_sha256":"ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0","backend":"cuda","context":32768,"temperature":0,"think_mode":"high"}`,
		"request.json":      `{"model":"deepseek-v4-flash"}`,
		"response.json":     `{"id":"x","choices":[]}`,
		"prompt_tokens.txt": "1\n2\n",
		"tokens.txt":        "3\n4\n",
		"terminal.json":     `{"logits_f32le_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}`,
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	got, err := LoadCorpusCase(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got.CaseID != "G2" || len(got.PromptTokens) != 2 || len(got.Tokens) != 2 || len(got.Terminal) == 0 {
		t.Fatalf("loaded case = %+v", got)
	}
	if err := ValidateCorpusProvenance(got.Metadata, got.Metadata); err != nil {
		t.Fatal(err)
	}
	want := got.Metadata
	want.BinarySHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
	if err := ValidateCorpusProvenance(got.Metadata, want); err == nil {
		t.Fatal("provenance mutation unexpectedly passed")
	}
	metadataWithoutStatus := strings.Replace(files["metadata.json"], `,"status":"HTTP/1.1 200 OK"`, "", 1)
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(metadataWithoutStatus), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCorpusCase(dir); err == nil {
		t.Fatal("missing status metadata unexpectedly passed")
	}
	metadataWithoutThinking := strings.Replace(files["metadata.json"], `,"think_mode":"high"`, "", 1)
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(metadataWithoutThinking), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCorpusCase(dir); err != nil {
		t.Fatalf("error case without thinking metadata: %v", err)
	}
}

func TestCompareCorpusCaseNormalizesJSONAndSSE(t *testing.T) {
	want := CorpusCase{Status: "HTTP/1.1 200 OK", Request: []byte(`{"model":"test","n":1}`), Response: []byte(`{"id":"old","created":1,"answer":"ok"}`), Tokens: []int{1, 2}}
	got := CorpusObservation{Status: want.Status, Request: []byte(`{"n":1,"model":"test"}`), Response: []byte(`{"id":"new","created":9,"answer":"ok"}`), Tokens: []int{1, 2}}
	if err := CompareCorpusCase(want, got); err != nil {
		t.Fatal(err)
	}
	want.Response = []byte("data: {\"id\":\"old\",\"created\":1,\"delta\":{\"text\":\"x\"}}\n\ndata: [DONE]\n\n")
	got.Response = []byte("data: {\"created\":2,\"delta\":{\"text\":\"x\"},\"id\":\"new\"}\n\ndata: [DONE]\n\n")
	if err := CompareCorpusCase(want, got); err != nil {
		t.Fatal(err)
	}
}

func TestCompareCorpusHTTPDoesNotInventTokenIdentity(t *testing.T) {
	want := CorpusCase{
		Status:   "HTTP/1.1 200 OK",
		Request:  []byte(`{"model":"test"}`),
		Response: []byte(`{"answer":"ok"}`),
		Tokens:   []int{1, 2, 3},
	}
	got := CorpusObservation{
		Status:   want.Status,
		Request:  []byte(`{"model":"test"}`),
		Response: want.Response,
	}
	if err := CompareCorpusHTTP(want, got); err != nil {
		t.Fatal(err)
	}
	if err := CompareCorpusCase(want, got); err == nil {
		t.Fatal("full comparison accepted missing token identity")
	}
	got.Response = []byte(`{"answer":"changed"}`)
	if err := CompareCorpusHTTP(want, got); err == nil {
		t.Fatal("HTTP comparison accepted changed response")
	}
}

func TestCompareCorpusCaseMutationsFailClosed(t *testing.T) {
	want := CorpusCase{Status: "HTTP/1.1 200 OK", Request: []byte(`{"model":"test"}`), Response: []byte(`{"answer":"ok"}`), Tokens: []int{1}}
	cases := []CorpusObservation{
		{Status: "HTTP/1.1 400 Bad Request", Request: want.Request, Response: want.Response, Tokens: want.Tokens},
		{Status: want.Status, Request: []byte(`{"model":"changed"}`), Response: want.Response, Tokens: want.Tokens},
		{Status: want.Status, Request: want.Request, Response: []byte(`{"answer":"wrong"}`), Tokens: want.Tokens},
		{Status: want.Status, Request: want.Request, Response: want.Response, Tokens: []int{2}},
		{Status: want.Status, Request: want.Request, Response: nil, Tokens: want.Tokens},
	}
	for i, got := range cases {
		if err := CompareCorpusCase(want, got); err == nil {
			t.Fatalf("mutation %d unexpectedly matched", i)
		}
	}
}

func TestCorpusCompareRequiresExactlyOneObservationPerCase(t *testing.T) {
	want := Corpus{Cases: map[string][]CorpusCase{
		"G1": {{Step: "one", Status: "HTTP/1.1 200 OK", Request: []byte(`{"x":1}`), Response: []byte(`{"ok":true}`)}},
		"G2": {{Step: "two", Status: "HTTP/1.1 200 OK", Request: []byte(`{"x":2}`), Response: []byte(`{"ok":true}`)}},
	}}
	obs := map[string]CorpusObservation{
		"G1/one": {Status: "HTTP/1.1 200 OK", Request: []byte(`{"x":1}`), Response: []byte(`{"ok":true}`)},
		"G2/two": {Status: "HTTP/1.1 200 OK", Request: []byte(`{"x":2}`), Response: []byte(`{"ok":true}`)},
	}
	if err := want.Compare(obs); err != nil {
		t.Fatal(err)
	}
	for name, mutated := range map[string]map[string]CorpusObservation{
		"missing": {"G1/one": obs["G1/one"]},
		"extra":   {"G1/one": obs["G1/one"], "G2/two": obs["G2/two"], "G3/three": obs["G2/two"]},
	} {
		t.Run(name, func(t *testing.T) {
			if err := want.Compare(mutated); err == nil {
				t.Fatal("incomplete observation set unexpectedly passed")
			}
		})
	}
}

func TestLoadCorpusRequiresAllCasesAndBuildBinding(t *testing.T) {
	root := t.TempDir()
	corpusDir := filepath.Join(root, "corpus")
	if err := os.MkdirAll(corpusDir, 0700); err != nil {
		t.Fatal(err)
	}
	const source = "92069b8b4010c18b72812221198c225956f39a14"
	const binary = "65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525"
	const model = "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
	if err := os.WriteFile(filepath.Join(root, "provenance.json"), []byte(fmt.Sprintf(`{"source_rev":%q,"binary_sha256":%q,"model_sha256":%q,"backend":"cuda","ctx":32768,"temperature":0,"think_mode":"high"}`, source, binary, model)), 0600); err != nil {
		t.Fatal(err)
	}
	var manifest, determinism strings.Builder
	manifest.WriteString(`{"cases":{`)
	determinism.WriteString(`{"cases":{`)
	for i := 1; i <= 9; i++ {
		if i > 1 {
			manifest.WriteByte(',')
			determinism.WriteByte(',')
		}
		id := fmt.Sprintf("G%d", i)
		manifest.WriteString(fmt.Sprintf(`%q:{"dir":%q,"verdict":"PASS"}`, id, id))
		determinism.WriteString(fmt.Sprintf(`%q:{"verdict":"PASS"}`, id))
		dir := filepath.Join(corpusDir, id, "step")
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		meta := fmt.Sprintf(`{"case_id":%q,"step":"step","method":"POST","endpoint":"/x","status":"HTTP/1.1 200 OK","source_rev":%q,"binary_sha256":%q,"model_sha256":%q,"backend":"cuda","context":32768,"temperature":0,"think_mode":"high"}`, id, source, binary, model)
		for name, data := range map[string]string{"metadata.json": meta, "response.json": `{}`} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	manifest.WriteString(`}}`)
	determinism.WriteString(`}}`)
	for name, data := range map[string]string{"manifest.json": manifest.String(), "determinism.json": determinism.String()} {
		if err := os.WriteFile(filepath.Join(corpusDir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	corpus, err := LoadCorpus(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(corpus.Cases) != 9 || len(corpus.Cases["G9"]) != 1 {
		t.Fatalf("cases = %+v", corpus.Cases)
	}
	if err := os.RemoveAll(filepath.Join(corpusDir, "G9")); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCorpus(root); err == nil {
		t.Fatal("missing G9 corpus unexpectedly passed")
	}
	mutated := strings.Replace(manifest.String(), `"G9":{"dir":"G9"`, `"G9":{"dir":"../G9"`, 1)
	if err := os.WriteFile(filepath.Join(corpusDir, "manifest.json"), []byte(mutated), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCorpus(root); err == nil {
		t.Fatal("path traversal manifest unexpectedly passed")
	}
}

func TestLoadCorpusRequiresSamplingProvenance(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "provenance.json"), []byte(`{"source_rev":"r","binary_sha256":"65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525","model_sha256":"ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0","backend":"cuda","ctx":32768}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadCorpusProvenance(filepath.Join(root, "provenance.json")); err == nil {
		t.Fatal("root provenance without sampling fields unexpectedly passed")
	}
}

func TestValidateCorpusBuildUsesPerStepThinkMode(t *testing.T) {
	root := CorpusProvenance{
		SourceRevision: "source",
		BinarySHA256:   "binary",
		ModelSHA256:    "model",
		Backend:        "cuda",
		Context:        32768,
		Temperature:    0,
		ThinkMode:      "per-case",
		ThinkModeByStep: map[string]string{
			"G6/messages-thinking-disabled": "none",
			"G2/chat":                       "high",
		},
	}
	base := root
	base.ThinkMode = "none"
	if err := validateCorpusBuild(base, root, "G6/messages-thinking-disabled"); err != nil {
		t.Fatalf("per-step mode rejected: %v", err)
	}
	base.ThinkMode = "high"
	if err := validateCorpusBuild(base, root, "G2/chat"); err != nil {
		t.Fatalf("default per-step mode rejected: %v", err)
	}
	for name, step := range map[string]string{
		"missing step": "G1/unknown",
		"wrong mode":   "G6/messages-thinking-disabled",
	} {
		t.Run(name, func(t *testing.T) {
			got := base
			if name == "wrong mode" {
				got.ThinkMode = "high"
			}
			if err := validateCorpusBuild(got, root, step); err == nil {
				t.Fatal("invalid per-step mode unexpectedly passed")
			}
		})
	}
}

func TestLoadCorpusProvenancePreservesPerCaseDefault(t *testing.T) {
	dir := t.TempDir()
	data := `{"source_rev":"r","binary_sha256":"65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525","model_sha256":"ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0","backend":"cuda","ctx":32768,"temperature":0,"think_mode":"per-case","think_mode_default":"high","think_mode_by_step":{"G2/chat":"high"}}`
	if err := os.WriteFile(filepath.Join(dir, "provenance.json"), []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := loadCorpusProvenance(filepath.Join(dir, "provenance.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got.ThinkModeDefault != "high" || got.ThinkModeByStep["G2/chat"] != "high" {
		t.Fatalf("provenance = %+v", got)
	}
}

func TestLoadCorpusCaseRejectsMissingProvenanceAndMalformedTokens(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(`{"case_id":"G1"}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCorpusCase(dir); err == nil {
		t.Fatal("incomplete metadata unexpectedly passed")
	}
	for name, data := range map[string]string{
		"metadata.json": `{"case_id":"G2","step":"chat","method":"POST","endpoint":"/x","status":"HTTP/1.1 200 OK","source_rev":"r","binary_sha256":"65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525","model_sha256":"ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0","backend":"cuda","context":1,"temperature":0,"think_mode":"high"}`,
		"response.json": `[]`,
		"tokens.txt":    "not-an-int\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := LoadCorpusCase(dir); err == nil {
		t.Fatal("malformed token trace unexpectedly passed")
	}
}

func TestLoadCorpusCasePreservesMalformedErrorRequest(t *testing.T) {
	dir := t.TempDir()
	meta := `{"case_id":"G1","step":"bad","method":"POST","endpoint":"/x","status":"HTTP/1.1 400 Bad Request","source_rev":"r","binary_sha256":"65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525","model_sha256":"ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0","backend":"cuda","context":1,"temperature":0}`
	for name, data := range map[string]string{
		"metadata.json": meta, "request.json": `{"broken":`, "response.json": `{}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := LoadCorpusCase(dir); err != nil {
		t.Fatalf("malformed error request rejected: %v", err)
	}
}

func TestLoadCorpusCaseRejectsMalformedStatus(t *testing.T) {
	dir := t.TempDir()
	meta := `{"case_id":"G1","step":"bad","method":"POST","endpoint":"/x","status":"OK","source_rev":"r","binary_sha256":"65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525","model_sha256":"ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0","backend":"cuda","context":1,"temperature":0}`
	for name, data := range map[string]string{
		"metadata.json": meta, "response.json": `{}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := LoadCorpusCase(dir); err == nil {
		t.Fatal("malformed status unexpectedly passed")
	}
}

func TestLoadCorpusCaseAcceptsSSEResponse(t *testing.T) {
	dir := t.TempDir()
	meta := `{"case_id":"G3","step":"stream","method":"POST","endpoint":"/v1/chat/completions","status":"HTTP/1.1 200 OK","source_rev":"r","binary_sha256":"65110ae0a76d4bb0578a57b7d4ba5e0b4526044aaada7ec6514493fa2b2e9525","model_sha256":"ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0","backend":"cuda","context":1,"temperature":0,"think_mode":"high"}`
	if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(meta), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "response.sse"), []byte("data: {\"choices\":[]}\n\n"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := LoadCorpusCase(dir)
	if err != nil {
		t.Fatal(err)
	}
	if string(got.Response) != "data: {\"choices\":[]}\n\n" {
		t.Fatalf("response = %q", got.Response)
	}
}
