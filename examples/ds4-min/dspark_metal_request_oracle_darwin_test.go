//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// The oracle compiles only pinned CPU helpers. It neither includes the MSL
// body nor calls the native encoder, so a shared shader defect cannot supply
// its own expected answer. Its sparse weights are identical to the recipe.
const requestHeadOracleCPP = `
int main(){
 if(std::fesetround(FE_TONEAREST)!=0)return 20;
 float latent[32]={1.0f,.001f};int8_t qlatent[32]={};float slatent[1]={};
 uint8_t selected[34]={};selected[0]=0;selected[1]=0x3c;selected[3]=1;
 quantize_q8_0_activation(latent,qlatent,slatent,32);
 float bias=dot_q8_0_row(selected,qlatent,slatent,32,1);
 float input[4128]={};input[0]=1;memcpy(input+4096,latent,sizeof(latent));
 int8_t qinput[4128]={};float scales[129]={};uint8_t confidence[129*34]={};
 memcpy(confidence+128*34,selected,34);
 quantize_q8_0_activation(input,qinput,scales,4128);
 float z=dot_q8_0_row(confidence,qinput,scales,4128,129);
 if(bias!=0||z!=0)return 21;
 const char *names[]={"zero-accept","confidence-decline","decline-before-nan","zero-nan-reject","unquantized-token","unquantized-decline","premature-logit-scan","sigmoid-as-raw"};
 const float thresholds[]={0,.5001f,.5001f,0,0,.5001f,.5001f,0};
 const bool poison[]={false,false,true,true,false,false,true,false};
 printf("[");
 for(int arm=0;arm<8;arm++){
  double values[8]={0,0,-99,-99,-99,-99,-99,-99};
  for(int row=0;row<2;row++){
   float probability=1.0f/(1.0f+expf(-z));values[6+row]=z;values[4+row]=probability;
   if(probability<thresholds[arm])break;
   // All finite target output weights are zero. The poisoned first Q8
   // scale produces NaN before any bias addition, at vocabulary index 0.
   if(poison[arm]){values[1]=1;break;}
   float best=-INFINITY;int token=0;
   for(int id=0;id<129280;id++){float value=id==1?bias:0;if(value>best){best=value;token=id;}}
   values[2+row]=token;values[0]++;
  }
  printf("%s{\"Name\":\"%s\",\"Values\":[",arm?",":"",names[arm]);
  for(int i=0;i<8;i++)printf("%s%.17g",i?",":"",values[i]);
  printf("]}");
 }
 printf("]\n");return 0;
}
`

func TestMetalDSparkRequestHeadFreeze(t *testing.T) {
	requestHeadFreeze(t, requestHeadFixtureDir(t))
}

func requestHeadFixtureDir(t *testing.T) string {
	t.Helper()
	if dir := os.Getenv("DS4_METAL_DSPARK_FIXTURE_DIR"); dir != "" {
		return dir
	}
	return t.TempDir()
}

func TestMetalDSparkRequestHeadFreezeOrdinaryEnvironment(t *testing.T) {
	t.Setenv("DS4_METAL_DSPARK_FIXTURE_DIR", "")
	dir := requestHeadFixtureDir(t)
	requestHeadFreeze(t, dir)
	data, err := os.ReadFile(filepath.Join(dir, "request-head.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest requestHeadManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if len(manifest.Oracle) != 8 || len(manifest.Arms) != 8 {
		t.Fatal("ordinary environment did not freeze all oracle records")
	}
}

func requestHeadFreeze(t *testing.T, dir string) {
	t.Helper()
	if Flash.Vocab != 129280 {
		t.Fatal("pinned oracle vocabulary differs from fixture")
	}
	pinned, err := os.ReadFile("testdata/dspark_q8_head_11e8cad.inc")
	if err != nil {
		t.Fatal(err)
	}
	source := []byte(metalDSparkHeadCPPPrelude + string(pinned) + requestHeadOracleCPP)
	probe := t.TempDir()
	path, binary := filepath.Join(probe, "oracle.cpp"), filepath.Join(probe, "oracle")
	if err := os.WriteFile(path, source, 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("clang++", "-std=c++17", "-U__ARM_NEON", "-U__ARM_FEATURE_DOTPROD", path, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("independent oracle compile: %v\n%s", err, out)
	}
	out, err := exec.Command(binary).CombinedOutput()
	if err != nil {
		t.Fatalf("independent oracle: %v\n%s", err, out)
	}
	var oracle []requestHeadOracle
	if err := json.Unmarshal(out, &oracle); err != nil {
		t.Fatal(err)
	}
	arms := requestHeadArms()
	if len(oracle) != len(arms) {
		t.Fatal("independent oracle record count")
	}
	for i, arm := range arms {
		if oracle[i].Name != arm.Name {
			t.Fatal("independent oracle record identity")
		}
	}
	finite, err := requestRecipeSeals(false)
	if err != nil {
		t.Fatal(err)
	}
	poison, err := requestRecipeSeals(true)
	if err != nil {
		t.Fatal(err)
	}
	pinnedHash, sourceHash := sha256.Sum256(pinned), sha256.Sum256(source)
	manifest := requestHeadManifest{PinnedC: fmt.Sprintf("11e8cad4473716a41baf4d62e92ed9764a00fce6:%x", pinnedHash), OracleSourceSHA256: fmt.Sprintf("%x", sourceHash), Arms: arms, Oracle: oracle, Finite: finite, Poison: poison}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "request-head.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "oracle.cpp"), source, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "oracle-output.json"), out, 0600); err != nil {
		t.Fatal(err)
	}
	libraries, err := requestHeadLibrarySources()
	if err != nil {
		t.Fatal(err)
	}
	for name, source := range libraries {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("ORACLE records=8 denominator=8 pinned=%s source=%s", manifest.PinnedC, manifest.OracleSourceSHA256)
}

// Separate device admission: compile all production owners and same-site
// shader mutants without allocating tensors or submitting command buffers.
func TestMetalDSparkRequestHeadLibraryGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_REQUEST_LIBRARY_GATE") != "1" {
		t.Skip("separate library compile admission required")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("canonical gpu-lock required")
	}
	dir := os.Getenv("DS4_METAL_DSPARK_FIXTURE_DIR")
	if dir == "" {
		t.Fatal("frozen library directory required")
	}
	if err := requestHeadCompileLibraries(dir, t.Logf); err != nil {
		t.Fatal(err)
	}
	t.Log("REQUEST_LIBRARY_TOTAL libraries=5 pipelines=11 terminal=no-submission cleanup=released")
}
