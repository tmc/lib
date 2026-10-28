package ncclinspector_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// These traces are constructed host controls that follow the printf formats
// of plugins/profiler/example/print_event.cc at the pinned commits. They are
// not captures of a running plugin.

type tphase struct {
	id         int
	start, end uint64
	ts         string // host callback time, deliberately unrelated to the GPU timer
}

type tchan struct {
	id          int
	start, stop uint64
	phases      []tphase
	proxy       bool
}

type tcoll struct {
	name     string
	seq      uint64
	hash     uint64
	rank     int
	count    uint64
	datatype string
	sym      int
	channels []tchan
}

const traceHash = 1311768467294899695 // 0x1234567890abcdef, above 2^53

var phaseName = []string{"initial_sync", "compute", "final_sync"}

func dur(start, stop uint64) string {
	if stop <= start {
		return "0.000000"
	}
	return fmt.Sprintf("%f", float64(stop-start)/1000.0)
}

func (c tcoll) lines(pid int) []string {
	var l []string
	add := func(format string, a ...any) { l = append(l, fmt.Sprintf(format, a...)) }
	add(`{"name": "%s", "cat": "COLL", "ph": "b", "id": 0, "pid": %d, "tid": 1, "ts": 10.500000, "args": {"SeqNum": %d, "CommHash": %d, "Rank": %d, "Count": %d, "Datatype": "%s", "Algorithm": "Ring", "Protocol": "Simple", "nChannels": %d, "KernelVariant": "", "IsSymColl": %d}},`,
		c.name, pid, c.seq, c.hash, c.rank, c.count, c.datatype, len(c.channels), c.sym)
	for _, ch := range c.channels {
		add(`{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": %d, "tid": 1, "ts": 11.250000, "args": {"Channel": %d, "StartGpuClk": %d, "StopGpuClk": %d}},`, pid, ch.id, ch.start, ch.stop)
		for _, p := range ch.phases {
			add(`{"name": "%s", "cat": "GPU", "ph": "X", "id": 0, "pid": %d, "tid": 1, "ts": %s, "dur": %s, "args": {"Channel": %d, "PhaseId": %d, "StartGpuClk": %d, "StopGpuClk": %d}},`,
				phaseName[p.id], pid, p.ts, dur(p.start, p.end), ch.id, p.id, p.start, p.end)
		}
		if ch.proxy {
			add(`{"name": "ScheduleSend", "cat": "PROXY", "ph": "b", "id": 0, "pid": %d, "tid": 1, "ts": 12.000000, "args": {"Channel": %d, "Peer": 1, "Steps": 4, "ChunkSize": 524288, "transSize": 4096}},`, pid, ch.id)
			add(`{"name": "ScheduleSend", "cat": "PROXY", "ph": "e", "id": 0, "pid": %d, "tid": 1, "ts": 12.500000},`, pid)
		}
		add(`{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": %d, "tid": 1, "ts": 13.000000},`, pid)
	}
	add(`{"name": "%s", "cat": "COLL", "ph": "e", "id": 0, "pid": %d, "tid": 1, "ts": 14.750000},`, c.name, pid)
	return l
}

func traceOf(pid int, colls ...tcoll) []byte {
	var b strings.Builder
	b.WriteString("[\n")
	for _, c := range colls {
		for _, l := range c.lines(pid) {
			b.WriteString(l + "\n")
		}
	}
	b.WriteString("{}]\n")
	return []byte(b.String())
}

// phases returns the three phases of a symmetric kernel with host callback
// times that are later than, and unrelated to, the GPU timer values.
func phases(base uint64) []tphase {
	return []tphase{
		{0, base, base + 200, "11.300000"},
		{1, base + 200, base + 900, "11.400000"},
		{2, base + 900, base + 1000, "11.500000"},
	}
}

func goodColl() tcoll {
	return tcoll{name: "AllReduce", seq: 7, hash: traceHash, rank: 1, count: 1024, datatype: "ncclFloat32", sym: 1,
		channels: []tchan{
			{id: 0, start: 5000, stop: 6000, phases: phases(5000), proxy: true},
			{id: 3, start: 5100, stop: 6100, phases: phases(5100)},
		}}
}

func capture(b []byte, p ncclinspector.ProfilerProfile) ncclinspector.ProfilerCapture {
	sum := sha256.Sum256(b)
	return ncclinspector.ProfilerCapture{Profile: p, Build: "constructed-build", SHA256: hex.EncodeToString(sum[:])}
}

func readTrace(b []byte, p ncclinspector.ProfilerProfile, l ncclinspector.Limits) (*ncclinspector.ProfilerTrace, error) {
	return ncclinspector.ReadProfilerTrace(bytes.NewReader(b), capture(b, p), l)
}

func TestReadProfilerTrace(t *testing.T) {
	b := traceOf(2101, goodColl())
	tr, err := readTrace(b, ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if tr.PID != 2101 || len(tr.Collectives) != 1 || len(tr.PointToPoints) != 0 || tr.Events != 14 || len(tr.Other) != 2 {
		t.Fatalf("trace = pid %d colls %d p2p %d events %d other %d", tr.PID, len(tr.Collectives), len(tr.PointToPoints), tr.Events, len(tr.Other))
	}
	c := tr.Collectives[0]
	if c.CommHash != traceHash || c.Communicator() != "0x1234567890abcdef" || c.Rank != 1 || c.Sequence != 7 || !c.Symmetric {
		t.Fatalf("collective = %+v", c)
	}
	if n, ok := c.Bytes(); !ok || n != 4096 {
		t.Fatalf("bytes = %d, %v", n, ok)
	}
	if c.BeginMicros != "10.500000" || c.EndMicros != "14.750000" || !bytes.Contains(c.Begin, []byte(`"CommHash": 1311768467294899695`)) {
		t.Fatalf("raw coll facts = %q %q %q", c.BeginMicros, c.EndMicros, c.Begin)
	}
	if len(c.KernelChannels) != 2 || c.KernelChannels[1].Channel != 3 || len(c.KernelChannels[1].Phases) != 3 {
		t.Fatalf("channels = %+v", c.KernelChannels)
	}
	p := c.KernelChannels[0].Phases[1]
	ns, ok := p.ReportedNS()
	if p.Name != "compute" || !ok || ns != 700 || p.DurationMicros != "0.700000" || p.Micros != "11.400000" || p.Withheld != "" {
		t.Fatalf("phase = %+v ns %d", p, ns)
	}
	if !bytes.Contains(p.Raw, []byte(`"StartGpuClk": 5200`)) || !bytes.HasSuffix(p.Raw, []byte("},\n")) {
		t.Fatalf("phase raw = %q", p.Raw)
	}
	if len(c.KernelChannels[0].Begin) == 0 || len(c.KernelChannels[0].End) == 0 {
		t.Fatal("channel framing lines not retained")
	}
}

func TestReadProfilerTraceUnorderableTimestampIsWithheld(t *testing.T) {
	// A timestamp that overflows int64 without its decimal point cannot be
	// ordered against the COLL begin: the channel is withheld, not admitted.
	s := strings.Replace(string(traceOf(2101, goodColl())), `"ts": 11.250000, "args": {"Channel": 0`, `"ts": 99999999999999.000000, "args": {"Channel": 0`, 1)
	tr, err := readTrace([]byte(s), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	ch := tr.Collectives[0].KernelChannels
	if ch[0].Withheld == "" || ch[1].Withheld != "" {
		t.Fatalf("withheld = %q, %q", ch[0].Withheld, ch[1].Withheld)
	}
}

func TestReadProfilerTraceAdjacentEndpointConflict(t *testing.T) {
	// The producer prints phase i as ts[i] -> ts[i+1] of one array, with
	// ts[BEGIN] and ts[END] equal to the KernelCh start and stop. Boundaries
	// that disagree withhold that channel only; channel 3 stays admitted.
	tests := []struct {
		name string
		mut  func(*tchan)
		want string // withheld reason; "" means admitted
	}{
		{"exact shared boundaries", func(*tchan) {}, ""},
		{"stop before next start", func(c *tchan) { c.phases[1].start++ }, "initial_sync and compute disagree"},
		{"stop after next start", func(c *tchan) { c.phases[1].start-- }, "initial_sync and compute disagree"},
		{"final boundary mismatch", func(c *tchan) { c.phases[2].start++ }, "compute and final_sync disagree"},
		{"skipped compute exact", func(c *tchan) { c.phases = []tphase{{0, 5000, 5200, "11.300000"}, {2, 5200, 6000, "11.500000"}} }, ""},
		{"skipped compute mismatch", func(c *tchan) { c.phases = []tphase{c.phases[0], c.phases[2]} }, "initial_sync and final_sync disagree"},
		{"folded after_open exact", func(c *tchan) { c.phases = []tphase{{1, 5000, 5900, "11.400000"}, c.phases[2]} }, ""},
		{"reversed shared boundaries", func(c *tchan) { c.phases[0].end, c.phases[1].start = 4900, 4900 }, "initial_sync GPU boundaries are not in increasing order"},
		{"unprinted zero interval", func(c *tchan) { c.phases[2].end = c.phases[2].start }, "final_sync GPU boundaries"},
		{"initial_sync start differs", func(c *tchan) { c.start++ }, "initial_sync start differs"},
		{"final_sync stop differs", func(c *tchan) { c.stop-- }, "final_sync stop differs"},
		{"unrecorded KernelCh stop", func(c *tchan) { c.stop = 0 }, ""},
		{"skipped initial_sync exact", func(c *tchan) { c.phases = []tphase{{1, 5000, 5900, "11.400000"}, c.phases[2]} }, ""},
		{"skipped final_sync exact", func(c *tchan) { c.phases = []tphase{c.phases[0], {1, 5200, 6000, "11.400000"}} }, ""},
		{"compute only exact", func(c *tchan) { c.phases = []tphase{{1, 5000, 6000, "11.400000"}} }, ""},
		{"compute only, start mismatch", func(c *tchan) { c.phases = []tphase{{1, 5100, 6000, "11.400000"}} }, "compute start differs"},
		{"compute and final_sync, start mismatch", func(c *tchan) { c.phases = c.phases[1:] }, "compute start differs"},
		{"initial_sync only, stop mismatch", func(c *tchan) { c.phases = []tphase{{0, 5000, 5900, "11.300000"}} }, "initial_sync stop differs"},
		{"initial_sync and compute, stop mismatch", func(c *tchan) { c.phases = c.phases[:2] }, "compute stop differs"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := goodColl()
			tt.mut(&c.channels[0])
			tr, err := readTrace(traceOf(2101, c), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
			if err != nil {
				t.Fatalf("conflict refused the whole trace: %v", err)
			}
			ch := tr.Collectives[0].KernelChannels
			if !strings.Contains(ch[0].Withheld, tt.want) || (tt.want == "") != (ch[0].Withheld == "") {
				t.Fatalf("withheld = %q, want %q", ch[0].Withheld, tt.want)
			}
			for _, p := range ch[0].Phases {
				if p.Withheld != ch[0].Withheld {
					t.Errorf("%s withheld = %q, channel %q", p.Name, p.Withheld, ch[0].Withheld)
				}
			}
			if ch[1].Withheld != "" || ch[1].Phases[1].Withheld != "" {
				t.Fatalf("independent channel withheld: %+v", ch[1])
			}
		})
	}
}

// writerChannel prints channel id as profilerFireKernelPhaseEvents does for
// one device array ts, whose ts[0] and ts[3] are the KernelCh start and stop.
// A stop of 0 is a KernelCh stop not yet recorded when printed.
func writerChannel(id int, ts [4]uint64, stop uint64) tchan {
	ch := tchan{id: id, start: ts[0], stop: stop}
	if ts[1] == 0 && ts[2] == 0 {
		return ch
	}
	if ts[1] == 0 && ts[0] != 0 {
		ts[1] = ts[0]
	}
	if ts[2] == 0 && ts[3] != 0 {
		ts[2] = ts[3]
	}
	for i := range 3 {
		if ts[i] == 0 || ts[i+1] == 0 || ts[i] == ts[i+1] {
			continue
		}
		ch.phases = append(ch.phases, tphase{i, ts[i], ts[i+1], "11.300000"})
	}
	return ch
}

func TestReadProfilerTraceWriterBoundariesAdmitted(t *testing.T) {
	// Every channel the pinned writer can print from one monotone device
	// array, with folded, skipped or absent boundaries, is admitted.
	for _, base := range []uint64{5000, 1_700_000_000_000_000_000, math.MaxUint64 - 3000} {
		for _, o1 := range []int64{-1, 0, 200, 1000} {
			for _, o2 := range []int64{-1, 0, 200, 500, 1000} {
				if o1 > o2 && o2 >= 0 {
					continue // device stamps are taken in program order
				}
				ts := [4]uint64{base, 0, 0, base + 1000}
				if o1 >= 0 {
					ts[1] = base + uint64(o1)
				}
				if o2 >= 0 {
					ts[2] = base + uint64(o2)
				}
				for _, stop := range []uint64{ts[3], 0} {
					for _, p := range []ncclinspector.ProfilerProfile{ncclinspector.ProfilerNCCL2323, ncclinspector.ProfilerNCCL2312} {
						c := goodColl()
						c.channels[0] = writerChannel(0, ts, stop)
						tr, err := readTrace(traceOf(2101, c), p, ncclinspector.Limits{})
						if err != nil {
							t.Fatalf("ts %v stop %d %s: %v", ts, stop, p, err)
						}
						ch := tr.Collectives[0].KernelChannels[0]
						if ch.Withheld != "" {
							t.Errorf("ts %v stop %d %s withheld: %q", ts, stop, p, ch.Withheld)
						}
						for _, ph := range ch.Phases {
							if ns, ok := ph.ReportedNS(); ph.Withheld != "" || !ok || uint64(ns) != ph.StopGpuClk-ph.StartGpuClk {
								t.Errorf("ts %v %s: phase %+v", ts, p, ph)
							}
						}
					}
				}
			}
		}
	}
}

func TestReadProfilerTraceEmptyAndSeveral(t *testing.T) {
	tr, err := readTrace([]byte("[\n{}]\n"), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
	if err != nil || tr.Events != 0 || tr.PID != 0 {
		t.Fatalf("empty trace = %+v, %v", tr, err)
	}
	// Every kernelId and collId in a trace may be 0: association is lexical.
	a, b := goodColl(), goodColl()
	b.seq, b.channels = 8, []tchan{{id: 1, start: 7000, stop: 7100, phases: phases(7000)}}
	tr, err = readTrace(traceOf(9, a, b), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
	if err != nil || len(tr.Collectives) != 2 || len(tr.Collectives[1].KernelChannels) != 1 || tr.Collectives[1].KernelChannels[0].Channel != 1 {
		t.Fatalf("two collectives = %+v, %v", tr, err)
	}
}

func TestReadProfilerTraceRefusals(t *testing.T) {
	good := string(traceOf(2101, goodColl()))
	replace := func(old, new string) string {
		if !strings.Contains(good, old) {
			t.Fatalf("fixture lacks %q", old)
		}
		return strings.Replace(good, old, new, 1)
	}
	p2p := `{"name": "Send", "cat": "P2P", "ph": "b", "id": 0, "pid": 2101, "tid": 1, "ts": 1.000000, "args": {"CommHash": 1, "Rank": 0, "Peer": 1, "Count": 8, "Datatype": "ncclInt8", "nChannels": 1}},
{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 2101, "tid": 1, "ts": 1.000000, "args": {"Channel": 0, "StartGpuClk": 10, "StopGpuClk": 20}},
{"name": "initial_sync", "cat": "GPU", "ph": "X", "id": 0, "pid": 2101, "tid": 1, "ts": 1.000000, "dur": 0.010000, "args": {"Channel": 0, "PhaseId": 0, "StartGpuClk": 10, "StopGpuClk": 20}},
{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 2101, "tid": 1, "ts": 1.000000},
{"name": "Send", "cat": "P2P", "ph": "e", "id": 0, "pid": 2101, "tid": 1, "ts": 1.000000},
`
	phase0 := `{"name": "initial_sync", "cat": "GPU", "ph": "X", "id": 0, "pid": 2101, "tid": 1, "ts": 11.300000, "dur": 0.200000, "args": {"Channel": 0, "PhaseId": 0, "StartGpuClk": 5000, "StopGpuClk": 5200}},`
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"truncated", good[:len(good)-4], "terminal"},
		{"no terminal", strings.TrimSuffix(good, "{}]\n"), "terminal"},
		{"trailing data", good + "x", "after terminal"},
		{"no opening", good[2:], "opening"},
		{"missing comma", replace(`"IsSymColl": 1}},`, `"IsSymColl": 1}}`), "separator"},
		{"duplicate event key", replace(`"pid": 2101, "tid": 1, "ts": 10.500000`, `"pid": 2101, "pid": 2101, "tid": 1, "ts": 10.500000`), "duplicate"},
		{"duplicate args key", replace(`"Rank": 1,`, `"Rank": 1, "Rank": 2,`), "duplicate"},
		{"case aliased key", replace(`"Rank": 1,`, `"Rank": 1, "rank": 2,`), "duplicate"},
		{"null field", replace(`"Rank": 1,`, `"Rank": null,`), "Rank"},
		{"null args", "[\n" + `{"name": "AllReduce", "cat": "COLL", "ph": "b", "id": 0, "pid": 2101, "tid": 1, "ts": 10.500000, "args": null},` + "\n{}]\n", "args"},
		{"float hash", replace(`"CommHash": 1311768467294899695`, `"CommHash": 1311768467294899695.0`), "CommHash"},
		{"exponent hash", replace(`"CommHash": 1311768467294899695`, `"CommHash": 1.3e18`), "CommHash"},
		{"hash overflow", replace(`"CommHash": 1311768467294899695`, `"CommHash": 18446744073709551616`), "CommHash"},
		{"string hash", replace(`"CommHash": 1311768467294899695`, `"CommHash": "1311768467294899695"`), "CommHash"},
		{"leading zero count", replace(`"Count": 1024`, `"Count": 01024`), "invalid"},
		{"short timestamp", replace(`"ts": 10.500000`, `"ts": 10.5`), "ts"},
		{"unexpected field", replace(`"IsSymColl": 1}}`, `"IsSymColl": 1, "Extra": 1}}`), "unexpected"},
		{"missing field", replace(`"Protocol": "Simple", `, ``), "Protocol"},
		{"bad sym flag", replace(`"IsSymColl": 1`, `"IsSymColl": 2`), "identity"},
		{"negative rank", replace(`"Rank": 1,`, `"Rank": -1,`), "identity"},
		{"unknown category", replace(`"cat": "COLL", "ph": "b"`, `"cat": "SURPRISE", "ph": "b"`), "category"},
		{"tid", replace(`"pid": 2101, "tid": 1, "ts": 10.500000`, `"pid": 2101, "tid": 2, "ts": 10.500000`), "thread"},
		{"pid change", replace(`"pid": 2101, "tid": 1, "ts": 14.750000`, `"pid": 2102, "tid": 1, "ts": 14.750000`), "process"},
		{"end name", replace(`{"name": "AllReduce", "cat": "COLL", "ph": "e"`, `{"name": "AllGather", "cat": "COLL", "ph": "e"`), "name"},
		{"end id", replace(`"id": 0, "pid": 2101, "tid": 1, "ts": 14.750000`, `"id": 1, "pid": 2101, "tid": 1, "ts": 14.750000`), "ID"},
		{"channel end id", replace(`"id": 0, "pid": 2101, "tid": 1, "ts": 13.000000`, `"id": 1, "pid": 2101, "tid": 1, "ts": 13.000000`), "ID"},
		{"phase for another channel", replace(`"args": {"Channel": 0, "PhaseId": 0,`, `"args": {"Channel": 3, "PhaseId": 0,`), "channel differs"},
		{"phase name and id", replace(`{"name": "compute"`, `{"name": "final_sync"`), "disagree"},
		{"phase id beyond kinds", replace(`"PhaseId": 2,`, `"PhaseId": 3,`), "disagree"},
		{"phase duration", replace(`"dur": 0.700000`, `"dur": 0.800000`), "duration"},
		{"repeated phase", strings.Replace(good, phase0+"\n", phase0+"\n"+phase0+"\n", 1), "repeated"},
		{"phase outside kernel", "[\n" + phase0 + "\n" + strings.TrimPrefix(good, "[\n"), "outside"},
		{"phase after proxy", replace(`{"name": "ScheduleSend", "cat": "PROXY", "ph": "e"`, strings.TrimSuffix(phase0, ",")+",\n"+`{"name": "ScheduleSend", "cat": "PROXY", "ph": "e"`), "after channel"},
		{"channel beyond bound", replace(`"Channel": 3, "StartGpuClk"`, `"Channel": 64, "StartGpuClk"`), "bound"},
		{"channel order", replace(`"Channel": 3, "StartGpuClk"`, `"Channel": 0, "StartGpuClk"`), "out-of-order"},
		{"duplicate collective", string(traceOf(2101, goodColl(), goodColl())), "duplicate"},
		{"nested collective", replace(`{"name": "KernelCh", "cat": "GPU", "ph": "b"`, strings.Split(good, "\n")[1]+"\n"+`{"name": "KernelCh", "cat": "GPU", "ph": "b"`), "nested"},
		{"unterminated scope", strings.Replace(good, strings.Split(good, "\n")[len(strings.Split(good, "\n"))-3]+"\n", "", 1), "unterminated"},
		{"collective event in channel", replace(`{"name": "KernelCh", "cat": "GPU", "ph": "e"`, `{"name": "g", "cat": "GROUP_API", "ph": "b", "id": 0, "pid": 2101, "tid": 1, "ts": 1.000000},`+"\n"+`{"name": "KernelCh", "cat": "GPU", "ph": "e"`), "inside"},
		{"end without begin", "[\n" + `{"name": "AllReduce", "cat": "COLL", "ph": "e", "id": 0, "pid": 2101, "tid": 1, "ts": 1.000000},` + "\n{}]\n", "without"},
		{"2.32 point-to-point phase", "[\n" + p2p + "{}]\n", "never instruments"},
		{"2.32 regular phase", string(traceOf(2101, func() tcoll { c := goodColl(); c.sym = 0; return c }())), "never instruments"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := readTrace([]byte(tt.in), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestReadProfilerTraceContext(t *testing.T) {
	b := traceOf(2101, goodColl())
	c := capture(b, ncclinspector.ProfilerNCCL2323)
	tests := []struct {
		name string
		mod  func(*ncclinspector.ProfilerCapture)
		l    ncclinspector.Limits
		want string
	}{
		{"future profile", func(c *ncclinspector.ProfilerCapture) { c.Profile = "nccl-v2.33.0-1/example-profiler-v8" }, ncclinspector.Limits{}, "unsupported profile"},
		{"no profile", func(c *ncclinspector.ProfilerCapture) { c.Profile = "" }, ncclinspector.Limits{}, "unsupported profile"},
		{"no build", func(c *ncclinspector.ProfilerCapture) { c.Build = "" }, ncclinspector.Limits{}, "build"},
		{"no digest", func(c *ncclinspector.ProfilerCapture) { c.SHA256 = "" }, ncclinspector.Limits{}, "SHA-256"},
		{"upper digest", func(c *ncclinspector.ProfilerCapture) { c.SHA256 = strings.ToUpper(c.SHA256) }, ncclinspector.Limits{}, "SHA-256"},
		{"wrong digest", func(c *ncclinspector.ProfilerCapture) { c.SHA256 = strings.Repeat("0", 64) }, ncclinspector.Limits{}, "digest mismatch"},
		{"byte limit", nil, ncclinspector.Limits{Bytes: 100}, "byte limit"},
		{"line limit", nil, ncclinspector.Limits{Line: 50}, "line limit"},
		{"event limit", nil, ncclinspector.Limits{Records: 3}, "event limit"},
		{"invalid limits", nil, ncclinspector.Limits{Bytes: 1 << 30}, "limits"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := c
			if tt.mod != nil {
				tt.mod(&c)
			}
			_, err := ncclinspector.ReadProfilerTrace(bytes.NewReader(b), c, tt.l)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
	if _, err := ncclinspector.ReadProfilerTrace(nil, c, ncclinspector.Limits{}); err == nil {
		t.Fatal("nil reader accepted")
	}
	if _, err := ncclinspector.ReadProfilerTrace(errReader{}, c, ncclinspector.Limits{}); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("reader error = %v", err)
	}
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, fmt.Errorf("boom") }

func TestProfilerSources(t *testing.T) {
	a, err := ncclinspector.ProfilerNCCL2323.Source()
	if err != nil {
		t.Fatal(err)
	}
	b, err := ncclinspector.ProfilerNCCL2312.Source()
	if err != nil {
		t.Fatal(err)
	}
	if a.Commit != "12df1a11afad322be5a204a2db890161cbf8131d" || b.Commit != "7b83616df3ae082a1f32bb74c27458bfe8153a13" {
		t.Fatalf("commits = %s %s", a.Commit, b.Commit)
	}
	if a.PrintEventSHA256 != b.PrintEventSHA256 || a.PluginSHA256 == b.PluginSHA256 {
		t.Fatal("print_event.cc must be identical and plugin.cc distinct across the profiles")
	}
	if !a.RegularPhasesGated || b.RegularPhasesGated || !a.APIRanks || b.APIRanks || a.MaxChannels != 64 || b.MaxChannels != 32 {
		t.Fatalf("sources = %+v %+v", a, b)
	}
}

func TestProfiler2312Distinctions(t *testing.T) {
	// 2.31.2 has the v7 phase format and is not refused wholesale.
	reg := goodColl()
	reg.sym = 0
	b := traceOf(2101, reg)
	tr, err := readTrace(b, ncclinspector.ProfilerNCCL2312, ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if got := tr.Collectives[0].KernelChannels[0].Phases[0].Withheld; got == "" {
		t.Fatal("regular phase in a profile without gating must be withheld")
	}
	if tr.Collectives[0].Rank != 1 {
		t.Fatal("rank of COLL discarded for 2.31")
	}
	// The 2.31 example has 32 channels; 2.32 has 64.
	wide := goodColl()
	wide.channels = []tchan{{id: 40, start: 1, stop: 2}}
	b = traceOf(2101, wide)
	if _, err := readTrace(b, ncclinspector.ProfilerNCCL2312, ncclinspector.Limits{}); err == nil {
		t.Fatal("channel 40 accepted for 2.31")
	}
	if _, err := readTrace(b, ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{}); err != nil {
		t.Fatal(err)
	}
}

func TestProfilerP2PAndOther(t *testing.T) {
	// P2P is retained apart from collectives and keeps no sequence.
	p2p := `[
{"name": "Send", "cat": "P2P", "ph": "b", "id": 0, "pid": 5, "tid": 1, "ts": 1.000000, "args": {"CommHash": 18446744073709551615, "Rank": 0, "Peer": 1, "Count": 8, "Datatype": "ncclInt8", "nChannels": 1}},
{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 5, "tid": 1, "ts": 1.000000, "args": {"Channel": 0, "StartGpuClk": 10, "StopGpuClk": 20}},
{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 5, "tid": 1, "ts": 1.000000},
{"name": "Send", "cat": "P2P", "ph": "e", "id": 0, "pid": 5, "tid": 1, "ts": 1.000000},
{"name": "Group", "cat": "GROUP_API", "ph": "b", "id": 0, "pid": 5, "tid": 1, "ts": 1.000000, "args": {"graphCaptured": false}},
{"name": "Group", "cat": "GROUP_API", "ph": "e", "id": 0, "pid": 5, "tid": 1, "ts": 2.000000},
{}]
`
	tr, err := readTrace([]byte(p2p), ncclinspector.ProfilerNCCL2323, ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.PointToPoints) != 1 || len(tr.Collectives) != 0 || tr.PointToPoints[0].CommHash != 18446744073709551615 || len(tr.Other) != 2 {
		t.Fatalf("trace = %+v", tr)
	}
	if tr.Other[0].Category != "GROUP_API" || !bytes.Contains(tr.Other[0].Raw, []byte("graphCaptured")) {
		t.Fatalf("other = %+v", tr.Other)
	}
}

func BenchmarkReadProfilerTrace(b *testing.B) {
	var cs []tcoll
	for i := range 64 {
		c := goodColl()
		c.seq = uint64(i)
		cs = append(cs, c)
	}
	data := traceOf(2101, cs...)
	c := capture(data, ncclinspector.ProfilerNCCL2323)
	b.SetBytes(int64(len(data)))
	for b.Loop() {
		if _, err := ncclinspector.ReadProfilerTrace(bytes.NewReader(data), c, ncclinspector.Limits{}); err != nil {
			b.Fatal(err)
		}
	}
}
