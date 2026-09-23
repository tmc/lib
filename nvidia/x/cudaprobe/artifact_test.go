package cudaprobe

import (
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestArtifactIdentityRequiresRegistration(t *testing.T) {
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry k() {ret;}"), "k", cudakernel.Schema{}, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	e, err := Decode(capture("cuLaunchKernel", 5, 99, 1, 1, 1, 1, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	e.Identity.AddressSpace = 1
	b := Binding{Process: e.Identity.Process, AddressSpace: 1, Function: 99, ModuleGeneration: 7, First: 2, Last: 8, Evidence: "synthetic registration", Artifact: a}
	k, err := Identify(e, []Binding{b})
	if err != nil || k.Digest != a.Digest || k.EntryPoint != "k" || k.ModuleGeneration != 7 {
		t.Fatalf("kernel=%+v error=%v", k, err)
	}
	r, err := e.RecordWithBindings(time.Now(), []Binding{b})
	if err != nil || r.KernelName != "k" || r.Labels["kernel.digest"] != a.Digest || r.Labels["duration_status"] != "unavailable" {
		t.Fatalf("record=%+v error=%v", r, err)
	}
	r, err = e.RecordWithBindings(time.Now(), nil)
	if err == nil || r.KernelName != "" || r.Labels["kernel_identity"] != "unavailable" || r.Labels["probe.symbol"] != "cuLaunchKernel" {
		t.Fatalf("missing join discarded or misattributed observation: %+v %v", r, err)
	}
	if _, err := Identify(e, nil); err == nil {
		t.Fatal("guessed missing registration")
	}
	if _, err := Identify(e, []Binding{b, b}); err == nil {
		t.Fatal("accepted ambiguous registrations")
	}
	for _, change := range []func(*Binding){
		func(b *Binding) { b.Last = 4 }, func(b *Binding) { b.First = 6 },
		func(b *Binding) { b.Process.Generation++ }, func(b *Binding) { b.AddressSpace++ },
		func(b *Binding) { b.Function++ }, func(b *Binding) { b.ModuleGeneration = 0 },
		func(b *Binding) { b.Evidence = "" }, func(b *Binding) { b.Artifact = nil },
	} {
		x := b
		change(&x)
		if _, err := Identify(e, []Binding{x}); err == nil {
			t.Fatalf("accepted stale/incomplete binding: %+v", x)
		}
	}
	a.Raw[0] = 'x'
	if _, err := Identify(e, []Binding{b}); err == nil {
		t.Fatal("accepted corrupted artifact")
	}
}
