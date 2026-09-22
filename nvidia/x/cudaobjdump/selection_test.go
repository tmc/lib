package cudaobjdump

import (
	"context"
	"fmt"
	"testing"
)

func TestExtractExactlyOneImage(t *testing.T) {
	tool := fake(t, `case "$1" in
--list-elf) printf 'ELF file 1: first.cubin\nELF file 2: second.cubin\n';;
--extract-elf) test "$2" = second.cubin || exit 9; printf selected > second.cubin;;
*) exit 8;;
esac`)
	image, receipts, err := tool.ExtractImage(context.Background(), input(t), ELF, "second.cubin")
	if err != nil || image.Name != "second.cubin" || string(image.Data) != "selected" || len(receipts) != 2 || receipts[1].Args[1] != "second.cubin" {
		t.Fatalf("image=%v receipts=%v err=%v", image, receipts, err)
	}
}

func TestExtractImageRefusesAmbiguousSelection(t *testing.T) {
	tool := fake(t, `if test "$1" = --list-elf; then
printf 'ELF file 1: a.cubin\nELF file 2: extra.a.cubin\n'
else exit 99; fi`)
	for _, name := range []string{"a.cubin", "missing"} {
		_, receipts, err := tool.ExtractImage(context.Background(), input(t), ELF, name)
		if err == nil || len(receipts) != 1 {
			t.Fatalf("%q reached extraction: %v %v", name, receipts, err)
		}
	}
	for _, name := range []string{"", "all", "a,b"} {
		_, receipts, err := tool.ExtractImage(context.Background(), input(t), ELF, name)
		if err == nil || len(receipts) != 0 {
			t.Fatalf("special selector %q accepted", name)
		}
	}
}

func TestExtractImageRejectsUnexpectedOutput(t *testing.T) {
	tool := fake(t, `if test "$1" = --list-elf; then
printf 'ELF file 1: a.cubin\nELF file 2: b.cubin\n'
else printf a > a.cubin; printf b > b.cubin; fi`)
	if _, _, err := tool.ExtractImage(context.Background(), input(t), ELF, "a.cubin"); err == nil {
		t.Fatal("accepted unselected output")
	}
}

func ExampleTool_ExtractImage() {
	_, commands, err := (Tool{}).ExtractImage(context.Background(), "input.fatbin", ELF, "all")
	fmt.Println(err != nil, len(commands))
	// Output: true 0
}
