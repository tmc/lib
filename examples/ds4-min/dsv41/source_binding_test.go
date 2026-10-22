package dsv41

import "testing"

// This exercises the real Bind path with the existing header-only model fixture.
// It does not fabricate a full-file hash or claim BindVerified asset admission.
func TestModelRetainsBoundSource(t *testing.T) {
	g := modelFixture(t)
	source := new(countReader)
	m, err := Bind(source, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	if m.boundSource != source {
		t.Fatal("Bind lost actual source reader")
	}
	if m.boundSize != int64(g.Size) {
		t.Fatal("Bind lost source size")
	}
	if m.fileSize != g.Size {
		t.Fatal("Bind lost routed-triplet file size")
	}
	if source.reads != 0 {
		t.Fatal("Bind source retention read payload")
	}
	if got, size, err := m.VerifiedSource(); err == nil || got != nil || size != 0 {
		t.Fatal("plain Bind source exposed as verified")
	}
}
