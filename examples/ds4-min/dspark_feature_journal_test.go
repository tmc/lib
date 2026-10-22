package main

import "testing"

func TestDSparkFeatureJournal(t *testing.T) {
	j, err := newDSparkFeatureJournal(3, 9)
	if err != nil {
		t.Fatal(err)
	}
	if err := j.append(1, 3, 9); err == nil {
		t.Fatal("accepted hole")
	}
	if err := j.append(0, 3, 8); err == nil {
		t.Fatal("accepted wrong epoch")
	}
	for i := 0; i < 3; i++ {
		if err := j.append(i, i+10, 9); err != nil {
			t.Fatal(err)
		}
	}
	backup := j.clone()
	for row := 0; row < 3; row++ {
		position, slot, err := j.groupedRow(3, row, 3, 9)
		if err != nil || position != 3+row || slot != row {
			t.Fatalf("row %d: %d/%d %v", row, position, slot, err)
		}
	}
	if _, _, err := j.groupedRow(2, 0, 3, 9); err == nil {
		t.Fatal("accepted wrong grouped frontier")
	}
	if err := j.append(3, 20, 9); err != nil {
		t.Fatal(err)
	}
	if _, err := j.row(0, 9); err == nil {
		t.Fatal("accepted overwritten row")
	}
	if got, err := backup.row(0, 9); err != nil || got.token != 10 {
		t.Fatalf("snapshot aliased: %+v %v", got, err)
	}
	if backup.frontier != 3 || backup.prefix == j.prefix {
		t.Fatal("snapshot frontier or prefix changed")
	}
	if _, err := j.row(3, 8); err == nil {
		t.Fatal("accepted foreign row epoch")
	}
	if err := j.validate(4); err != nil {
		t.Fatal(err)
	}
	changed := j.clone()
	changed.rows[0].token++
	if err := changed.validate(4); err == nil {
		t.Fatal("accepted changed trusted history")
	}
}

func TestDSparkFeatureJournalShape(t *testing.T) {
	rows, values, err := dsparkFeatureJournalShape(4096, 5, 3, 4096)
	if err != nil || rows != Flash.SlidingWindow+6 || values != rows*3*4096 {
		t.Fatalf("shape %d %d %v", rows, values, err)
	}
	for _, shape := range [][4]int{{0, 5, 3, 4096}, {4, 5, 3, 4096}, {4096, 5, int(^uint(0) >> 1), 4096}} {
		if _, _, err := dsparkFeatureJournalShape(shape[0], shape[1], shape[2], shape[3]); err == nil {
			t.Fatalf("accepted shape %v", shape)
		}
	}
}
