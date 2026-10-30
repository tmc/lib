package main

import "testing"

func TestAfmoeRuntimeClosedWithoutPreparation(t *testing.T) {
	if decoder, err := newAfmoeDecoder(nil, nil, nil, 0, false); decoder != nil || err == nil {
		t.Fatalf("unprepared runtime accepted: decoder=%v error=%v", decoder, err)
	}
	var decoder *afmoeDecoder
	if err := decoder.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := decoder.forward(0, 0, true); err == nil {
		t.Fatal("unprepared forward entered execution")
	}
	if err := (&afmoeDecoder{}).Close(); err != nil {
		t.Fatal(err)
	}
}
