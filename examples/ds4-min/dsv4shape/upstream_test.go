package dsv4shape

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

const (
	oracleCommit = "11e8cad4473716a41baf4d62e92ed9764a00fce6"
	oracleBlob   = "20b523b581ff6f67d80327e7680c71d8494c6408"
	oracleSHA256 = "62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7"
)

func checkOraclePin(b []byte) error {
	if got := fmt.Sprintf("%x", sha256.Sum256(b)); got != oracleSHA256 {
		return fmt.Errorf("oracle source sha256 %s, want %s", got, oracleSHA256)
	}
	return nil
}

func frozenOracle(t *testing.T) []byte {
	t.Helper()
	p := os.Getenv("DS4_SHAPE_ORACLE")
	if p == "" {
		t.Skip("DS4_SHAPE_ORACLE not set")
	}
	// The path locates the repository, not the file to trust. Read the exact
	// frozen Git object even when the checkout has moved or is modified.
	repo := filepath.Dir(p)
	object := oracleCommit + ":ds4.c"
	blob, err := exec.Command("git", "-C", repo, "rev-parse", object).Output()
	if err != nil || strings.TrimSpace(string(blob)) != oracleBlob {
		t.Fatalf("oracle Git object: %q %v", blob, err)
	}
	b, err := exec.Command("git", "-C", repo, "show", object).Output()
	if err != nil {
		t.Fatal(err)
	}
	if err := checkOraclePin(b); err != nil {
		t.Fatal(err)
	}
	return b
}

func TestOraclePin(t *testing.T) {
	if checkOraclePin([]byte("unqualified source")) == nil {
		t.Fatal("bad source admitted")
	}
	b := frozenOracle(t)
	b[0] ^= 1
	if checkOraclePin(b) == nil {
		t.Fatal("one-bit source mutation admitted")
	}
}

func TestUpstreamShapeTables(t *testing.T) {
	b := frozenOracle(t)
	s := string(b)
	for _, v := range []Variant{Flash, Pro} {
		name := "FLASH"
		if v == Pro {
			name = "PRO"
		}
		start := strings.Index(s, "static const ds4_shape DS4_SHAPE_"+name+" = {")
		if start < 0 {
			t.Fatal("shape table missing")
		}
		end := strings.Index(s[start:], "\n};")
		if end < 0 {
			t.Fatal("shape table unterminated")
		}
		c, _ := For(v)
		fields := map[string]int{"n_layer": c.Layers, "n_embd": c.Embed, "n_vocab": c.Vocab, "n_head": c.Heads, "n_head_kv": c.HeadsKV, "n_head_dim": c.HeadDim, "n_value_dim": c.ValueDim, "n_rot": c.RopeDim, "n_lora_q": c.QLoRA, "n_lora_o": c.OutLoRA, "n_out_group": c.OutGroups, "n_expert": c.Experts, "n_expert_used": c.ExpertsUsed, "n_expert_shared": c.ExpertsShared, "n_ff_exp": c.ExpertFF, "n_hash_layer": c.HashLayers, "n_swa": c.SlidingWindow, "n_indexer_head": c.IndexerHeads, "n_indexer_head_dim": c.IndexerHeadDim, "n_indexer_top_k": c.IndexerTopK, "n_hc": c.HC, "n_hc_sinkhorn_iter": c.SinkhornIters}
		for field, want := range fields {
			m := regexp.MustCompile(`\.` + field + `\s*=\s*([0-9]+)\s*,`).FindStringSubmatch(s[start : start+end])
			if len(m) != 2 {
				t.Fatalf("missing literal %s %s", name, field)
			}
			got, err := strconv.Atoi(m[1])
			if err != nil || got != want {
				t.Fatalf("%s %s C=%s Go=%d", name, field, m[1], want)
			}
		}
		if len(fields) != 22 {
			t.Fatalf("integer fields %d, want 22", len(fields))
		}
		m := regexp.MustCompile(`\.expert_weight_scale\s*=\s*([0-9.]+)f\s*,`).FindStringSubmatch(s[start : start+end])
		if len(m) != 2 {
			t.Fatalf("missing scale literal %s", name)
		}
		scale, err := strconv.ParseFloat(m[1], 32)
		if err != nil || float32(scale) != c.ExpertWeightScale {
			t.Fatalf("%s expert scale C=%s Go=%g", name, m[1], c.ExpertWeightScale)
		}
	}
}

// This optional source oracle executes the literal upstream schedule body.
// It does not compile the model, allocate a device, or prove kernel support.
func TestUpstreamSchedule(t *testing.T) {
	b := frozenOracle(t)
	s := string(b)
	start := strings.Index(s, "static uint32_t ds4_expected_layer_compress_ratio(uint32_t il) {")
	if start < 0 {
		t.Fatal("missing upstream schedule")
	}
	end := strings.Index(s[start:], "\n}")
	if end < 0 {
		t.Fatal("unterminated upstream schedule")
	}
	body := s[start : start+end+2]
	pre := "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#define DS4_VARIANT_FLASH 0\n#define DS4_VARIANT_PRO 1\n#define DS4_VARIANT_FLASH41 2\nstatic int DS4_MODEL_VARIANT, DS4_N_LAYER;\nstatic void ds4_die(const char *s) { (void)s; exit(2); }\n"
	main := "\nint main(void) { for(int v=0;v<2;v++) { DS4_MODEL_VARIANT=v; DS4_N_LAYER=v?61:43; for(int i=0;i<DS4_N_LAYER;i++) printf(\"%d %d %u\\n\",v,i,ds4_expected_layer_compress_ratio(i)); } }\n"
	d := t.TempDir()
	source := filepath.Join(d, "schedule.c")
	binary := filepath.Join(d, "schedule")
	if err := os.WriteFile(source, []byte(pre+body+main), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-std=c11", "-Wall", "-Werror", source, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v: %s", err, out)
	}
	out, err := exec.Command(binary).Output()
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) != 104 {
		t.Fatalf("rows %d want 104", len(lines))
	}
	for _, line := range lines {
		var v, i, ratio int
		if _, err := fmt.Sscanf(line, "%d %d %d", &v, &i, &ratio); err != nil {
			t.Fatal(err)
		}
		c, _ := For(Variant(v))
		l, err := c.Layer(i)
		if err != nil || l.CompressRatio != ratio {
			t.Fatalf("C %s Go %+v %v", line, l, err)
		}
	}
}
