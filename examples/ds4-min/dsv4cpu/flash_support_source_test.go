package dsv4cpu

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// This is a source placement control, not a complete target model run.
func TestFlashSupportActualCaptureSite(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("source location unavailable")
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(file), "flash_state.go"))
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	start := strings.Index(source, "func (m *flashModel) evalLayer(")
	if start < 0 {
		t.Fatal("actual layer body unavailable")
	}
	end := strings.Index(source[start:], "// setSteering")
	if end < 0 {
		t.Fatal("actual layer body unavailable")
	}
	body := source[start : start+end]
	if !strings.Contains(body, "return s.combineLayer(dst, block, after, post, comb, il)") {
		t.Fatal("actual layer does not use observed final Combine body")
	}
	helper, err := os.ReadFile(filepath.Join(filepath.Dir(file), "flash_support.go"))
	if err != nil {
		t.Fatal(err)
	}
	combine := strings.Index(string(helper), "if err := Combine(dst, block, after, post, comb); err != nil {")
	capture := strings.Index(string(helper), "return s.support.capture(il, dst)")
	if combine < 0 || capture <= combine {
		t.Fatal("support capture is not after successful final Combine")
	}
	finite := strings.Index(source, "return errors.New(\"nonfinite Flash logits\")")
	publish := strings.Index(source, "s.support.publish(s.position, token)")
	commit := strings.Index(source, "s.position++")
	if finite < 0 || publish <= finite || commit <= publish {
		t.Fatal("support publication does not precede actual finite-logit position commit")
	}
}
