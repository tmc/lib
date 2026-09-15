package cuptiowner

import "testing"

func TestLease(t *testing.T) {
	a, err := AcquireDevice(1)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Release()
	if _, err := AcquireDevice(1); err == nil {
		t.Fatal("conflict admitted")
	}
	b, err := AcquireDevice(2)
	if err != nil {
		t.Fatal(err)
	}
	b.Release()
	a.Release()
	c, err := AcquireDevice(1)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Release()
	a.Release() // An old lease cannot release a newer owner.
	if _, err := AcquireDevice(1); err == nil {
		t.Fatal("stale release relinquished newer owner")
	}
}
