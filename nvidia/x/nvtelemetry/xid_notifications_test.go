package nvtelemetry_test

import (
	"fmt"
	"io"
	"strings"
	"testing"

	telemetry "github.com/tmc/lib/nvidia/x/nvtelemetry"
	"github.com/tmc/lib/nvidia/x/xidcatalog"
	"github.com/tmc/lib/nvidia/x/xidlog"
)

func xidOptions() xidlog.Options {
	return xidlog.Options{Source: xidlog.Source{BootID: "boot", InputID: "supplied"}, Product: xidcatalog.H100, Driver: "580"}
}
func ExampleReadXIDNotifications() {
	result, err := telemetry.ReadXIDNotifications(strings.NewReader("NVRM: Xid (PCI:0000:65:00): 79, constructed\n"), xidlog.Text, xidOptions(), nil)
	fmt.Println(result.Records[0].Code, result.Records[0].DeviceIdentityKnown, result.LossUnknown, err)
	// Output: 79 false true <nil>
}
func TestXIDNotificationConsumer(t *testing.T) {
	raw := "6,10,123,-;NVRM: Xid (PCI:0000:65:00): 79, private-process-secret\n"
	record, err := xidlog.ParseKmsg(raw, xidOptions())
	if err != nil {
		t.Fatal(err)
	}
	// A record-framed reader returns one complete record per Read.
	reader := &oneXIDRecord{raw: raw}
	result, err := telemetry.ReadXIDNotifications(reader, xidlog.KernelRecords, xidOptions(), map[string]string{"0000:65:00": "GPU-a"})
	if err != nil {
		t.Fatal(err)
	}
	n := result.Records[0]
	if n.Code != 79 || !n.DeviceIdentityKnown || n.DeviceUUID != "GPU-a" || !n.HasKernelTimestamp || n.KernelMonotonicMicroseconds != 123 || n.Catalog.Code != record.Catalog.Code || !result.LossUnknown || !result.InputEnded {
		t.Fatal(result)
	}
	if strings.Contains(fmt.Sprint(result), "private-process-secret") {
		t.Fatal("raw payload retained")
	}
	for _, tc := range []struct {
		raw     string
		options xidlog.Options
	}{
		{"6,10,123,-;unrelated\n6,10,124,-;unrelated\n", xidOptions()},
		{"6,10,123,-;unrelated\n6,11,122,-;unrelated\n", xidOptions()},
		{strings.Repeat("x", xidlog.MaxRecordBytes+1), xidOptions()},
	} {
		format := xidlog.Text
		if strings.HasPrefix(tc.raw, "6,") {
			format = xidlog.KernelRecords
		}
		var input interface{ Read([]byte) (int, error) } = strings.NewReader(tc.raw)
		if format == xidlog.KernelRecords {
			input = &xidRecords{records: strings.SplitAfter(tc.raw, "\n")}
		}
		if _, err := telemetry.ReadXIDNotifications(input, format, tc.options, nil); err == nil {
			t.Fatal("invalid input admitted")
		}
	}
	options := xidOptions()
	options.Source.BootID = ""
	if _, err := telemetry.ReadXIDNotifications(strings.NewReader(raw), xidlog.Text, options, nil); err == nil {
		t.Fatal("unknown source admitted")
	}
}

type oneXIDRecord struct{ raw string }

func (r *oneXIDRecord) Read(p []byte) (int, error) {
	if r.raw == "" {
		return 0, io.EOF
	}
	n := copy(p, r.raw)
	r.raw = ""
	return n, nil
}

type xidRecords struct{ records []string }

func (r *xidRecords) Read(p []byte) (int, error) {
	if len(r.records) == 0 || r.records[0] == "" {
		return 0, io.EOF
	}
	n := copy(p, r.records[0])
	r.records = r.records[1:]
	return n, nil
}

func TestXIDNotificationUnknowns(t *testing.T) {
	options := xidOptions()
	options.Product = "L4"
	result, err := telemetry.ReadXIDNotifications(strings.NewReader("NVRM: Xid (PCI:0000:65:00): 79, constructed\nNVRM: Xid corrupt\n"), xidlog.Text, options, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Records) != 2 || result.Records[0].Applicability != xidcatalog.UncoveredProduct || result.Records[0].DeviceIdentityKnown || result.Records[0].HasKernelTimestamp || result.Records[1].Kind != xidlog.Malformed || !result.LossUnknown {
		t.Fatal(result)
	}
}
