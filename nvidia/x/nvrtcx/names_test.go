package nvrtcx

import (
	"crypto/sha256"
	"fmt"
	"testing"
)

func TestNameExpressionOptions(t *testing.T) {
	names := []string{"kernel<int>"}
	option := WithNameExpressions(names...)
	names[0] = "changed"
	var cfg options
	option(&cfg)
	if cfg.err != nil || len(cfg.expressions) != 1 || cfg.expressions[0] != "kernel<int>" {
		t.Fatalf("options: %+v", cfg)
	}
	for _, name := range []string{"", "a\x00b"} {
		var cfg options
		WithNameExpressions(name)(&cfg)
		if cfg.err == nil {
			t.Fatalf("accepted %q", name)
		}
	}
}

func TestLoweredNameCache(t *testing.T) {
	cm := &CompiledModule{output: OutputPTX, ptx: "ptx", lowered: map[string]string{"kernel<int>": "_Z6kernelIiEvv"}}
	encoded, err := encodeCacheEntry(cm)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeCacheEntry("key", encoded)
	if err != nil {
		t.Fatal(err)
	}
	cm.lowered["kernel<int>"] = "changed"
	if name, ok := got.LoweredName("kernel<int>"); !ok || name != "_Z6kernelIiEvv" {
		t.Fatalf("name=%q ok=%v", name, ok)
	}
	if _, ok := got.LoweredName("missing"); ok {
		t.Fatal("missing name found")
	}
	var absent *CompiledModule
	if _, ok := absent.LoweredName("missing"); ok {
		t.Fatal("nil module found name")
	}
}

func ExampleWithNameExpressions() {
	option := WithNameExpressions("kernel<int>")
	fmt.Println(option != nil)
	// Output: true
}

func ExampleCompiledModule_LoweredName() {
	var module *CompiledModule
	_, ok := module.LoweredName("kernel<int>")
	fmt.Println(ok)
	// Output: false
}

func TestRejectOldNameCache(t *testing.T) {
	cm := &CompiledModule{output: OutputPTX, ptx: "ptx"}
	b, err := encodeCacheEntry(cm)
	if err != nil {
		t.Fatal(err)
	}
	b[len(cacheMagic)-1] = 4
	sum := sha256.Sum256(b[:len(b)-sha256.Size])
	copy(b[len(b)-sha256.Size:], sum[:])
	if _, err := decodeCacheEntry("old", b); err == nil {
		t.Fatal("accepted version 4 cache")
	}
}

func TestRejectInvalidCachedNames(t *testing.T) {
	for _, tc := range []struct{ name, expression, lowered string }{
		{"empty expression", "", "kernel"},
		{"empty lowered name", "kernel", ""},
		{"NUL expression", "kernel\x00other", "kernel"},
		{"NUL lowered name", "kernel", "kernel\x00other"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cm := &CompiledModule{output: OutputPTX, ptx: "ptx", lowered: map[string]string{tc.expression: tc.lowered}}
			data, err := encodeCacheEntry(cm)
			if err != nil {
				t.Fatal(err)
			}
			// The checksum is valid: rejection must come from the name contract,
			// rather than treating this as a generic damaged-file case.
			if _, err := decodeCacheEntry("key", data); err == nil {
				t.Fatal("accepted invalid cached name")
			}
		})
	}
}
