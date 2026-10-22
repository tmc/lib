package dsv4server

import (
	"strings"
	"testing"
)

const testGoldenArchive = `-- request.json --
{"model":"test","messages":[{"role":"user","content":"hello"}]}
-- response.json --
{"id":"old","created":1,"choices":[{"message":{"role":"assistant","content":"cc"}}]}
-- tokens.txt --
1
2
`

func TestGoldenTranscriptMasksVolatileResponseFields(t *testing.T) {
	golden, err := LoadGolden([]byte(testGoldenArchive))
	if err != nil {
		t.Fatal(err)
	}
	response := `{"id":"new","created":99,"choices":[{"message":{"role":"assistant","content":"cc"}}]}`
	if err := CompareGolden(golden, []byte(`{"messages":[{"content":"hello","role":"user"}],"model":"test"}`), []byte(response), []int{1, 2}); err != nil {
		t.Fatalf("comparison failed: %v", err)
	}
}

func TestGoldenTranscriptMutationsFailClosed(t *testing.T) {
	cases := []struct {
		name     string
		archive  string
		request  string
		response string
		tokens   []int
	}{
		{"missing response", strings.Replace(testGoldenArchive, "-- response.json --\n{\"id\":\"old\",\"created\":1,\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"cc\"}}]}\n", "", 1), "", "", nil},
		{"request mutation", testGoldenArchive, `{"model":"test","messages":[{"role":"user","content":"changed"}]}`, `{"choices":[{"message":{"role":"assistant","content":"cc"}}]}`, []int{1, 2}},
		{"response mutation", testGoldenArchive, `{"model":"test","messages":[{"role":"user","content":"hello"}]}`, `{"choices":[{"message":{"role":"assistant","content":"wrong"}}]}`, []int{1, 2}},
		{"token mutation", testGoldenArchive, `{"model":"test","messages":[{"role":"user","content":"hello"}]}`, `{"choices":[{"message":{"role":"assistant","content":"cc"}}]}`, []int{1, 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			golden, err := LoadGolden([]byte(tc.archive))
			if err == nil && tc.name == "missing response" {
				t.Fatal("missing response archive loaded")
			}
			if tc.name == "missing response" {
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := CompareGolden(golden, []byte(tc.request), []byte(tc.response), tc.tokens); err == nil {
				t.Fatal("mutation unexpectedly matched golden")
			}
		})
	}
}

func TestGoldenTranscriptRejectsDuplicateAndUnknownFiles(t *testing.T) {
	for _, archive := range []string{
		testGoldenArchive + "-- tokens.txt --\n3\n",
		strings.Replace(testGoldenArchive, "-- tokens.txt --", "-- extra.txt --", 1),
	} {
		if _, err := LoadGolden([]byte(archive)); err == nil {
			t.Fatalf("archive unexpectedly loaded: %q", archive)
		}
	}
	if _, err := LoadGolden(append([]byte(testGoldenArchive), make([]byte, maxGoldenBytes)...)); err == nil {
		t.Fatal("oversized archive unexpectedly loaded")
	}
}

func TestGoldenTranscriptRejectsDuplicateJSONKeys(t *testing.T) {
	archive := "-- request.json --\n{\"model\":\"test\",\"model\":\"other\"}\n-- response.json --\n{}\n-- tokens.txt --\n1\n"
	if _, err := LoadGolden([]byte(archive)); err == nil || !strings.Contains(err.Error(), "duplicate object key") {
		t.Fatalf("LoadGolden duplicate key error = %v", err)
	}

	golden, err := LoadGolden([]byte(testGoldenArchive))
	if err != nil {
		t.Fatal(err)
	}
	request := `{"model":"test","messages":[{"role":"user","content":"hello"}],"model":"other"}`
	if err := CompareGolden(golden, []byte(request), []byte(`{"id":"new","choices":[]}`), []int{1, 2}); err == nil || !strings.Contains(err.Error(), "duplicate object key") {
		t.Fatalf("CompareGolden duplicate key error = %v", err)
	}
}

func TestGoldenTranscriptMasksNestedGeneratedIdentity(t *testing.T) {
	archive := "-- request.json --\n{}\n-- response.json --\n" +
		`{"id":"old","created":1,"signature":"sig-old","output_item_id":"item-old","choices":[{"message":{"content":"same","tool_call_id":"call-old"}}]}` +
		"\n-- tokens.txt --\n1\n"
	golden, err := LoadGolden([]byte(archive))
	if err != nil {
		t.Fatal(err)
	}
	response := `{"id":"new","created":2,"signature":"sig-new","output_item_id":"item-new","choices":[{"message":{"content":"same","tool_call_id":"call-new"}}]}`
	if err := CompareGolden(golden, []byte(`{}`), []byte(response), []int{1}); err != nil {
		t.Fatalf("generated response identity changed comparison: %v", err)
	}
	response = `{"id":"new","created":2,"signature":"sig-new","output_item_id":"item-new","choices":[{"message":{"content":"changed","tool_call_id":"call-new"}}]}`
	if err := CompareGolden(golden, []byte(`{}`), []byte(response), []int{1}); err == nil {
		t.Fatal("content mutation unexpectedly matched golden")
	}
}
