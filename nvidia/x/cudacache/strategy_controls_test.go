package cudacache

import (
	"bytes"
	"fmt"
	"io"
	"sync"
	"testing"
)

type strategyWriterFunc func([]byte) (int, error)

func (f strategyWriterFunc) Write(b []byte) (int, error) { return f(b) }

func TestKnowledgeBaseExternalWriterAndConcurrentSnapshots(t *testing.T) {
	k := new(StrategyKnowledgeBase)
	if err := k.Update(strategyFixture()); err != nil {
		t.Fatal(err)
	}
	w := strategyWriterFunc(func(b []byte) (int, error) { s := strategyFixture(); s.Confidence = 0.25; return len(b), k.Update(s) })
	if err := k.Write(w); err != nil {
		t.Fatal(err)
	}
	if err := k.Write(strategyWriterFunc(func(b []byte) (int, error) { return len(b) - 1, nil })); err != io.ErrShortWrite {
		t.Fatalf("short write: %v", err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				if err := k.Update(strategyFixture()); err != nil {
					t.Error(err)
				}
				var b bytes.Buffer
				if err := k.Write(&b); err != nil {
					t.Error(err)
				}
				if err := new(StrategyKnowledgeBase).Load(&b); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
}

func FuzzKnowledgeBaseLoad(f *testing.F) {
	var k StrategyKnowledgeBase
	k.Update(strategyFixture())
	var initial bytes.Buffer
	k.Write(&initial)
	f.Add(initial.Bytes())
	f.Add([]byte(`{"version":1,"version":1}`))
	f.Add([]byte(`{"version":1,"rows":[]}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		var before, after bytes.Buffer
		k.Write(&before)
		err := k.Load(bytes.NewReader(data))
		k.Write(&after)
		if err != nil && !bytes.Equal(before.Bytes(), after.Bytes()) {
			t.Fatal("failed load changed catalog")
		}
		if err == nil {
			var next StrategyKnowledgeBase
			if err := next.Load(bytes.NewReader(after.Bytes())); err != nil {
				t.Fatalf("successful load wrote invalid snapshot: %v", err)
			}
		}
	})
}

func TestKernelBlasterGroupMappingDetached(t *testing.T) {
	one := KernelBlasterGroups()
	one["api_overhead_dominant"][0] = "changed"
	if fmt.Sprint(KernelBlasterGroups()["api_overhead_dominant"][0]) != "api_overhead_dominant" {
		t.Fatal("mapping alias")
	}
}
