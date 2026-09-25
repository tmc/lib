//go:build arm64 && (darwin || linux)

package nccl

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// This fixture checks the generated ABI, not NCCL's GPU behavior. Run it in a
// child so the fake library never becomes another test's process-wide NCCL.
func TestTeamFFI(t *testing.T) {
	if lib := os.Getenv("NCCL_FFI_TEST_LIBRARY"); lib != "" {
		testTeamFFI(t, lib)
		return
	}
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("C compiler unavailable")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "fixture.c")
	lib := filepath.Join(dir, "fixture.so")
	if err := os.WriteFile(src, []byte(teamFFIFixture), 0600); err != nil {
		t.Fatal(err)
	}
	shared := "-shared"
	if runtime.GOOS == "darwin" {
		shared = "-dynamiclib"
	}
	if out, err := exec.Command(cc, shared, "-fPIC", src, "-o", lib).CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}
	child := exec.Command(os.Args[0], "-test.run=^TestTeamFFI$", "-test.v")
	child.Env = append(os.Environ(), "NCCL_FFI_TEST_LIBRARY="+lib)
	if out, err := child.CombinedOutput(); err != nil {
		t.Fatalf("native fixture: %v\n%s", err, out)
	}
}

func testTeamFFI(t *testing.T, lib string) {
	if err := Load(lib); err != nil {
		t.Fatal(err)
	}
	defer Close()
	if failures := BindFailures(); len(failures) != 0 {
		t.Fatalf("bind failures: %+v", failures)
	}
	team := Team{NRanks: 0x11111111, Rank: 0x22222222, Stride: 0x33333333}
	comm := Comm(7)
	for _, call := range []func(Team, int32) (int32, error){comm.TeamRankToWorld, comm.TeamRankToLsa} {
		if got, err := call(team, 42); err != nil || got != 42 {
			t.Fatalf("rank = %d, %v", got, err)
		}
		bad := team
		bad.Stride = 0
		if got, err := call(bad, 42); err != nil || got != -1 {
			t.Fatalf("negative rank = %d, %v", got, err)
		}
	}
	var req DevResourceRequirements
	var gin GinBarrierHandle
	if err := comm.GinBarrierCreateRequirement(team, 42, &gin, &req); err != nil || gin.Signal0 != 42 {
		t.Fatalf("gin = %+v, %v", gin, err)
	}
	var lsa LsaBarrierHandle
	if err := LsaBarrierCreateRequirement(team, 42, &lsa, &req); err != nil || lsa.NBarriers != 42 {
		t.Fatalf("lsa = %+v, %v", lsa, err)
	}
	var cft CftBarrierHandle
	if err := CftBarrierCreateRequirement(team, 42, &cft, &req); err != nil || cft.NBarriers != 42 {
		t.Fatalf("cft = %+v, %v", cft, err)
	}
	if id, offset, err := Window(7).CftDeviceLeInfo(19, 42, team); err != nil || id != 42 || offset != 19 {
		t.Fatalf("device = %d/%d, %v", id, offset, err)
	}
	// The 128-byte UniqueId stays on stock purego, independently of Team.
	if !HasCommInitRank() {
		t.Fatal("large struct registration lost")
	}
	var id UniqueId
	id.Internal[0], id.Internal[127] = 1, 2
	if got, err := CommInitRank(1, id, 0); err != nil || got != Comm(9) {
		t.Fatalf("large struct call = %v, %v", got, err)
	}
}

const teamFFIFixture = `
#include <stdint.h>
#include <stddef.h>
typedef struct { int32_t n, r, s; } team;
static int valid(team t) { return t.n==0x11111111 && t.r==0x22222222 && t.s==0x33333333; }
int ncclTeamRankToWorld(void *comm, team t, int rank) { return comm==(void*)7 && valid(t) ? rank : -1; }
int ncclTeamRankToLsa(void *comm, team t, int rank) { return ncclTeamRankToWorld(comm,t,rank); }
int ncclGinBarrierCreateRequirement(void *comm, team t, int n, uint32_t *out, void *req) {
 if(comm!=(void*)7 || !valid(t) || !out || !req) return 4;
 out[0]=n; return 0;
}
int ncclLsaBarrierCreateRequirement(team t, int n, uint32_t *out, void *req) {
 if(!valid(t) || !out || !req) return 4;
 out[1]=n; return 0;
}
int ncclCftBarrierCreateRequirement(team t, int n, uint32_t *out, void *req) { return ncclLsaBarrierCreateRequirement(t,n,out,req); }
int ncclGetCftDeviceLeInfo(void *win, size_t off, int peer, team t, uint32_t *id, size_t *offset) {
 if(win!=(void*)7 || !valid(t) || !id || !offset) return 4;
 *id=peer; *offset=off; return 0;
}
typedef struct { char data[128]; } unique;
int ncclCommInitRank(void **out, int n, unique id, int rank) {
 if(n!=1 || rank!=0 || id.data[0]!=1 || id.data[127]!=2) return 4;
 *out=(void*)9; return 0;
}
`
