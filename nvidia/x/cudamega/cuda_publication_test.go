package cudamega

import (
	"strings"
	"testing"
)

// Publication must gather every lane after its release fence and before
// thread zero advances completion. A thread-zero-only release is insufficient.
func TestCUDAPublicationGathersReleasedStores(t *testing.T) {
	c := &cudaSource{}
	if err := c.Publish(7); err != nil {
		t.Fatal(err)
	}
	want := "      __syncthreads();\n        if (threadIdx.x == 0) atomicAdd(completed + 7, 1u); // publish task 7\n"
	if got := c.body.String(); got != want {
		t.Fatalf("publication sequence=%q, want %q", got, want)
	}
	p := mixtureProgram(t, 4)
	source, err := p.Source()
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(source, "\n") {
		if strings.Contains(line, "atomicAdd(completed +") && !strings.Contains(line, "publish task") {
			t.Fatal("unrecognized completion publication")
		}
	}
	if strings.Count(source, "__syncthreads();\n        if (threadIdx.x == 0) atomicAdd(completed +") != 3 {
		t.Fatal("typed task completion lacks gather barrier")
	}
}
