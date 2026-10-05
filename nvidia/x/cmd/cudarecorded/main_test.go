package main

import "testing"

func TestMalformedArguments(t *testing.T) {
	if err := run(nil); err == nil || err.Error() != "recorded helper: expected keeper or supervisor, config path and digest" {
		t.Fatalf("error=%v", err)
	}
}
