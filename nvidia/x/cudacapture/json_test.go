package cudacapture

import (
	"strings"
	"testing"
)

func TestRejectDuplicateJSONMembers(t *testing.T) {
	for _, tc := range []struct {
		name string
		data string
		want string
	}{
		{name: "top-level", data: `{"version":3,"version":3}`, want: "duplicate JSON member"},
		{name: "nested", data: `{"files":{"a":"x","a":"y"}}`, want: "duplicate JSON member"},
		{name: "trailing", data: `{"version":3}{}`, want: "trailing JSON value"},
		{name: "valid", data: `{"files":{"a":"x"},"items":[{"id":1}]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := rejectDuplicateJSON([]byte(tc.data))
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
}
