package main

import (
	"strings"
	"testing"
)

func TestSidecarEmitterBuiltinMath(t *testing.T) {
	source, _, err := sidecarExpressionFixture(false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(source.CUDA, "#include") {
		t.Fatal("actual emitter requires a host header")
	}
	ref := &sidecarReference{source: source}
	if err := sidecarABI(ref, "chain"); err != nil {
		t.Fatal(err)
	}
	for _, header := range []string{
		"#include <math.h>",
		"#include <other.h>",
		"#include \"math.h\"",
		"#include <math.h>\n#include <math.h>",
		"#include <math.h>\n#define float int",
		"#include <math.h>\n#pragma pack(1)",
		"#include <math.h> \\\n#define float int",
		"#include <math.h> int disguised;",
	} {
		t.Run(header, func(t *testing.T) {
			copy := *ref
			copy.source = source
			copy.source.CUDA = header + "\n" + source.CUDA
			if sidecarABI(&copy, "chain") == nil {
				t.Fatal("unsupported directive accepted")
			}
		})
	}
}

func TestSidecarCommentContinuation(t *testing.T) {
	source, _, err := sidecarExpressionFixture(false)
	if err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{"\\\n", "\\\r\n", "\\ \n", "\\\t\n", "\\\v\n", "\\\f\n"} {
		t.Run(suffix, func(t *testing.T) {
			copy := source
			copy.CUDA = "// hidden declaration " + suffix + source.CUDA
			if sidecarABI(&sidecarReference{source: copy}, "chain") == nil {
				t.Fatal("raw comment continuation accepted")
			}
		})
	}
}
