package dsv4tools

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestValidateSchemas(t *testing.T) {
	got, err := Validate([]json.RawMessage{json.RawMessage(`{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}`)})
	if err != nil || len(got) != 1 || got[0].Name != "weather" || !strings.Contains(string(got[0].Raw), `"name":"weather"`) {
		t.Fatalf("Validate = %#v, %v", got, err)
	}
	if _, err := Validate([]json.RawMessage{json.RawMessage(`{"name":"bad","parameters":[]}`)}); err == nil {
		t.Fatal("accepted non-object parameters")
	}
}

func TestParseToolCalls(t *testing.T) {
	content, calls, err := Parse(`answer <tool_calls><invoke name="weather"><parameter name="city" string="true">New York</parameter><parameter name="days" string="false">3</parameter></invoke></tool_calls>`)
	if err != nil || content != "answer" || len(calls) != 1 {
		t.Fatalf("Parse = %q %#v, %v", content, calls, err)
	}
	if string(calls[0].Arguments) != `{"city":"New York","days":3}` {
		t.Fatalf("arguments = %s", calls[0].Arguments)
	}
}

func TestParseDSMLForms(t *testing.T) {
	const full = `<｜DSML｜tool_calls><｜DSML｜invoke name="weather"><｜DSML｜parameter name="city" string="true">New York</｜DSML｜parameter><｜DSML｜parameter name="options"><｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter></｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`
	const short = `<DSML｜tool_calls><DSML｜invoke name="weather"><DSML｜parameter name="days" string="false">3</DSML｜parameter></DSML｜invoke></DSML｜tool_calls>`
	for _, test := range []struct {
		name, input, want string
	}{
		{"full", full, `{"city":"New York","options":{"days":3}}`},
		{"short", short, `{"days":3}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			content, calls, err := Parse("prefix " + test.input + " suffix")
			if err != nil || content != "prefix  suffix" || len(calls) != 1 {
				t.Fatalf("Parse = %q %#v, %v", content, calls, err)
			}
			if string(calls[0].Arguments) != test.want {
				t.Fatalf("arguments = %s, want %s", calls[0].Arguments, test.want)
			}
		})
	}
}

func TestParseDSMLRejectsMutations(t *testing.T) {
	const valid = `<｜DSML｜tool_calls><｜DSML｜invoke name="x"><｜DSML｜parameter name="n" string="false">1</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`
	for _, test := range []struct {
		name, input string
	}{
		{"missing wrapper", `<｜DSML｜invoke name="x">`},
		{"missing close", strings.TrimSuffix(valid, `</｜DSML｜tool_calls>`)},
		{"missing invoke close", strings.Replace(valid, `</｜DSML｜invoke>`, "", 1)},
		{"missing parameter close", strings.Replace(valid, `</｜DSML｜parameter>`, "", 1)},
		{"non-json", strings.Replace(valid, ">1</", `>not-json</`, 1)},
		{"unexpected body", strings.Replace(valid, `><｜DSML｜parameter`, `>junk<｜DSML｜parameter`, 1)},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := Parse(test.input); err == nil {
				t.Fatalf("accepted mutation %q", test.input)
			}
		})
	}
}

func TestParseRejectsPartialMarkup(t *testing.T) {
	if content, calls, err := Parse("please invoke the weather tool"); err != nil || content != "please invoke the weather tool" || calls != nil {
		t.Fatalf("plain text = %q %#v, %v", content, calls, err)
	}
	for _, text := range []string{"<invoke name=\"x\">", "<tool_calls><invoke name=\"x\"></tool_calls>"} {
		if _, _, err := Parse(text); err == nil {
			t.Fatalf("accepted %q", text)
		}
	}
}
