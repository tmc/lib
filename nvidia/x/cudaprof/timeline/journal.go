package timeline

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// MaxRecords is the number of journal records a reader decodes from one
// journal before refusing it.
const MaxRecords = 1 << 20

// Errors wrapped by the refusals of [Journals]. A caller can use them to
// choose a remedy; the refusal itself already says what is missing.
var (
	// ErrJournalOmitted reports a journal the artifact budget left out of
	// the capture.
	ErrJournalOmitted = errors.New("journal omitted")
	// ErrRecordsDropped reports a capture that dropped records before they
	// reached a journal.
	ErrRecordsDropped = errors.New("records dropped")
	// ErrJournalBytes reports a journal whose decompressed size exceeds the
	// bound it was read under.
	ErrJournalBytes = errors.New("journal exceeds byte limit")
)

// refusal is an error whose text is msg alone and which wraps a sentinel.
type refusal struct {
	msg  string
	kind error
}

func (e *refusal) Error() string { return e.msg }
func (e *refusal) Unwrap() error { return e.kind }

// status is the part of capture-status.json that says whether the
// per-record detail of a capture is present at all.
type status struct {
	OmittedFiles []struct {
		Name   string `json:"name"`
		Bytes  int64  `json:"bytes"`
		SHA256 string `json:"sha256"`
		Reason string `json:"reason"`
	} `json:"omitted_files"`
	DetailedRecords string `json:"detailed_records"`
	RunCompleteness string `json:"run_completeness"`
	DroppedRecords  uint64 `json:"dropped_records"`
	ActivityRecords uint64 `json:"activity_records"`
	// UnqualifiedGenerations counts generations that never finalized. Their
	// activities are absent from the journal entirely, so a timeline built
	// from what remains has holes no drop count reports.
	UnqualifiedGenerations uint64 `json:"unqualified_generations"`
}

func readStatus(cap *cudacapture.Capture) (status, bool) {
	data, ok := cap.Files["capture-status.json"]
	if !ok {
		return status{}, false
	}
	var envelope struct {
		Status status `json:"status"`
	}
	if json.Unmarshal(data, &envelope) != nil {
		return status{}, false
	}
	return envelope.Status, true
}

// Journals returns the journal members of a capture, refusing every capture
// whose per-record detail is not wholly present. The refusal names the
// missing artifact and why the timeline cannot be reconstructed; it never
// falls back to an estimate, and never returns the part that survived.
//
// A refusal over an omitted journal wraps [ErrJournalOmitted], and one over
// dropped records wraps [ErrRecordsDropped].
func Journals(cap *cudacapture.Capture) (map[string][]byte, error) {
	st, haveStatus := readStatus(cap)
	var omitted []string
	for _, f := range st.OmittedFiles {
		if cudacapture.IsJournalName(f.Name) {
			omitted = append(omitted, fmt.Sprintf("%s (%d bytes, sha256 %s: %s)", f.Name, f.Bytes, f.SHA256, f.Reason))
		}
	}
	if len(omitted) != 0 {
		sort.Strings(omitted)
		return nil, &refusal{fmt.Sprintf("capture omits the per-record journal, so the device timeline cannot be reconstructed: %s; capture-status.json reports run_completeness=%q and detailed_records=%q, and gap analysis needs the omitted journal itself, not the pprof projection",
			strings.Join(omitted, ", "), st.RunCompleteness, st.DetailedRecords), ErrJournalOmitted}
	}
	journals := make(map[string][]byte)
	for name, data := range cap.Files {
		if cudacapture.IsJournalName(name) {
			journals[name] = data
		}
	}
	if len(journals) == 0 {
		detail := "capture-status.json is absent"
		if haveStatus {
			detail = fmt.Sprintf("capture-status.json reports run_completeness=%q and detailed_records=%q", st.RunCompleteness, st.DetailedRecords)
		}
		return nil, fmt.Errorf("capture contains no capture.journal member, so the device timeline cannot be reconstructed: %s; pprof samples are aggregates and carry no timestamps", detail)
	}
	if st.DroppedRecords > 0 {
		return nil, &refusal{fmt.Sprintf("capture dropped %d record(s) and retained %d, so the journal is missing activities the gaps would span: a duty cycle computed from the remainder would be wrong, not approximate", st.DroppedRecords, st.ActivityRecords), ErrRecordsDropped}
	}
	// A run can lose whole generations without dropping a record: a generation
	// that never finalized contributes nothing, and its absence is invisible in
	// a drop count. capture_success qualifies the target generation alone and
	// stays true through both, which is why this reads run_completeness.
	if st.RunCompleteness == "partial" {
		return nil, fmt.Errorf("capture reports run_completeness=%q with %d unqualified generation(s), so activities the idle gaps would span were never recorded: capture_success qualifies the target generation alone and does not report this", st.RunCompleteness, st.UnqualifiedGenerations)
	}
	return journals, nil
}

// Journal is one decoded capture journal member of the capture.
type Journal struct {
	Name       string `json:"name"`
	Compressed bool   `json:"compressed"`
	Bytes      int    `json:"bytes"`
	Records    int    `json:"records"`
	// NoCUDA reports a journal that never finalized because its process ran
	// no CUDA: it contributes nothing to the timeline, and it is listed so
	// that nothing was set aside silently.
	NoCUDA bool `json:"no_cuda,omitempty"`
}

// Records is everything [Decode] reads out of the journals.
type Records struct {
	UnifiedMemory  []UnifiedMemory
	MemoryPolicies []MemoryPolicy
	Activities     []Activity
	// APIs holds the host API calls by correlation id. Their Device is zero
	// and their Thread is the host thread that made the call.
	APIs     map[uint32]Span
	Journals []Journal

	// Calls holds the same host API calls with what they called, in
	// journal order, and Markers the NVTX markers.
	Calls   []Call
	Markers []Marker

	// Ranges holds the NVTX ranges, each a START marker paired with the
	// END that closed it. UnpairedMarkers counts the halves that found no
	// partner, such as a range still open when the process ended.
	Ranges          []Range
	UnpairedMarkers int

	// GEMMs holds the cuBLASLt calls, in journal order, when the capture
	// recorded them (cudaprof capture --gemms). It is empty otherwise.
	GEMMs []GEMM

	// Spans that explain idle rather than occupy the device. None of the
	// three is in Activities: see the exclusions in decodeJournal.
	Overheads []Span
	Graphs    []Activity
	// GraphLifecycle contains explicit native lifecycle receipts, when retained.
	GraphLifecycle []GraphEvent
	// Dependencies contains explicit stream, event and synchronization receipts.
	Dependencies []DependencyEvent
	Syncs        []Span
	rawSyncs     []rawSync

	// GraphNodes counts device records that name the graph they ran as a
	// node of. It is what decides whether excluding the replay envelope is
	// sound: the exclusion exists because the envelope spans those records,
	// and with none of them present it spans nothing.
	GraphNodes int

	// contextDevice maps a CUPTI context to the device the records that carry
	// both reported it on. A synchronization record carries a context and no
	// device, so this is the only thing that can place one on a device
	// timeline. Context 0 is never entered: it is the value a record that
	// named no context leaves behind, and treating it as a context would
	// resolve every unattributable sync onto whichever device happened to
	// leave a zero there.
	contextDevice map[uint32]uint32

	// Syncs whose context no device-carrying record ever named. They are
	// reported, not placed: assigning them to device 0 would put the
	// explanation for one device's idle on another device's timeline.
	UnresolvedSyncs int
	UnresolvedNanos int64
}

// An Activity is one device record: a kernel, memcpy, memset, or graph
// replay envelope.
type Activity struct {
	Sequence uint64 // source journal record sequence
	// Journal identifies the process generation that retained this record.
	Journal string

	Device      uint32
	Start, End  int64
	Correlation uint32

	// Carried for consumers that need the launch itself rather than only its
	// place on the timeline. They are unset for the spans that are not kernel
	// launches, and nothing in the gap analysis reads them.
	Subtype string
	Name    string
	Stream  uint32
	GraphID uint32
	// GraphNodeID and Context retain the native graph identity.
	GraphNodeID uint64
	Context     uint32
	// RuntimeCorrelation is distinct from the driver correlation.
	RuntimeCorrelation uint32
	Grid, Block        [3]int

	// SharedMemory is a kernel's dynamic shared memory in bytes, or nil when
	// the record does not carry it. Read only by compare.
	SharedMemory *uint32

	// Set only for a memcpy, and read only by compare.
	Bytes    uint64
	CopyKind uint8

	// Module is the driver's id of the module a kernel ran from, or 0 when
	// the record does not say. Read only by advise.
	Module uint32
}

// A GEMM is one cuBLASLt matmul call as the library logged it, joined to
// the kernels it launched.
type GEMM struct {
	Journal string
	ID      uint64
	Thread  int64
	// Time is when the library logged the call, which is before its
	// launches, on the host clock the activity records use.
	Time int64
	// API is the entry the call arrived through: cublasLtMatmul for a
	// direct call, an internal name such as cublasLtTSTMatmul for one made
	// by cuBLAS.
	API string
	// Log is the library's description of the call, unparsed.
	Log string
	// Launches counts the kernel launches the agent linked to the call.
	// Kernels holds those whose device record is in the capture; a launch
	// with none (one recorded into a CUDA graph, say) is counted only.
	Launches int
	Kernels  []Activity
}

// A Span is one span on one device's timeline. Overhead spans carry no
// device -- CUPTI's cost is charged to a host thread -- and are held with
// Device left at zero and applied to every device, which gap analysis states
// rather than hides.
type Span struct {
	Device     uint32
	Start, End int64
	Thread     uint32 // host thread, for an API call
}

// A Range is one NVTX range on one host thread.
type Range struct {
	Journal string // process generation journal

	Name            string // "domain/name" when the range has a domain
	Process, Thread uint32
	Start, End      int64
}

type markerKey struct {
	process, thread, id uint32
	domain              string
}

// pairMarkers pairs one journal's START and END markers into ranges, LIFO
// within a thread, id and domain because NVTX ranges nest. It returns the
// ranges and how many halves found no partner.
func pairMarkers(markers []recordPayload) ([]Range, int) {
	// Journal order is drain order, which producer threads interleave.
	sort.SliceStable(markers, func(i, j int) bool { return markers[i].Start < markers[j].Start })
	var ranges []Range
	unpaired := 0
	open := make(map[markerKey][]recordPayload)
	for _, m := range markers {
		k := markerKey{m.ProcessID, m.ThreadID, m.MarkerID, m.MarkerDomain}
		switch {
		case m.MarkerFlags&MarkerStart != 0:
			open[k] = append(open[k], m)
		case m.MarkerFlags&MarkerEnd != 0:
			stack := open[k]
			if len(stack) == 0 {
				unpaired++
				continue
			}
			start := stack[len(stack)-1]
			open[k] = stack[:len(stack)-1]
			name := start.Name
			if start.MarkerDomain != "" {
				name = start.MarkerDomain + "/" + name
			}
			ranges = append(ranges, Range{Name: name, Process: start.ProcessID, Thread: start.ThreadID, Start: int64(start.Start), End: int64(m.End)})
		}
	}
	for _, stack := range open {
		unpaired += len(stack)
	}
	return ranges, unpaired
}

// A Call is one host driver or runtime API call.
type Call struct {
	Sequence uint64 // source journal record sequence
	Journal  string // process generation journal
	Process  uint32

	Domain      string // "driver" or "runtime"
	CallbackID  uint32
	Thread      uint32
	Correlation uint32
	Start, End  int64
}

// Name returns the API the call made, as [CallName] spells it.
func (c Call) Name() string { return CallName(c.Domain, c.CallbackID) }

// Marker flags, as CUPTI_ACTIVITY_FLAG_MARKER_* define them.
const (
	MarkerInstantaneous = 1 << 0
	MarkerStart         = 1 << 1
	MarkerEnd           = 1 << 2
)

// A Marker is one NVTX marker record. CUPTI reports a range as a start
// record and an end record sharing an ID; only the start carries the name.
type Marker struct {
	Sequence        uint64 // source journal record sequence
	Journal         string
	Process, Thread uint32
	Name            string
	Domain          string
	ID              uint32
	Flags           uint32
	Time            int64
}

var (
	runtimeVersionRE = regexp.MustCompile(`V[0-9]+$`)
	driverVersionRE  = regexp.MustCompile(`V([0-9]+)(Ptds|Ptsz)?$`)
)

// CallName returns the name a program calls a driver or runtime API by,
// such as cuMemAlloc_v2 or cudaMalloc, given the callback ID CUPTI
// reported it under. An unknown ID is named by its number.
func CallName(domain string, cbid uint32) string {
	var name string
	switch domain {
	case "driver":
		// The stringer spells cuMemcpyHtoD_v2 as CuMemcpyHtoDV2.
		name = strings.TrimPrefix(cupti.DriverTraceCBID(cbid).String(), "DriverCBID")
		name = driverVersionRE.ReplaceAllStringFunc(name, func(m string) string {
			sub := driverVersionRE.FindStringSubmatch(m)
			v := "_v" + sub[1]
			if sub[2] != "" {
				v += "_" + strings.ToLower(sub[2])
			}
			return v
		})
	case "runtime":
		name = strings.TrimPrefix(cupti.RuntimeTraceCBID(cbid).String(), "RuntimeCBID")
		// Runtime callbacks carry the release that introduced them
		// (cudaMemcpy_v3020); the call a reader wrote has no suffix.
		name = runtimeVersionRE.ReplaceAllString(name, "")
	}
	if name == "" || strings.Contains(name, "(") {
		return fmt.Sprintf("%s:%d", domain, cbid)
	}
	return strings.ToLower(name[:1]) + name[1:]
}

// rawSync is a synchronization span before its device is known.
type rawSync struct {
	context    uint32
	start, end int64
}

// recordPayload is the subset of a typed journal payload the decoder reads.
type recordPayload struct {
	memoryPayload
	Kind               string  `json:"kind"`
	Subtype            string  `json:"subtype"`
	Name               string  `json:"name"`
	Stream             uint32  `json:"stream"`
	Bytes              uint64  `json:"bytes"`
	CopyKind           uint8   `json:"copy_kind"`
	GridX              uint32  `json:"grid_x"`
	GridY              uint32  `json:"grid_y"`
	GridZ              uint32  `json:"grid_z"`
	BlockX             uint32  `json:"block_x"`
	BlockY             uint32  `json:"block_y"`
	BlockZ             uint32  `json:"block_z"`
	SharedMemory       *uint32 `json:"dynamic_shared_memory"`
	State              string  `json:"state"`
	Domain             string  `json:"domain"`
	Valid              bool    `json:"valid"`
	Start              uint64  `json:"start"`
	End                uint64  `json:"end"`
	Device             uint32  `json:"device"`
	Context            uint32  `json:"context"`
	Correlation        uint32  `json:"correlation"`
	CallbackID         uint32  `json:"callback_id"`
	ThreadID           uint32  `json:"thread_id"`
	MarkerID           uint32  `json:"marker_id"`
	MarkerFlags        uint32  `json:"marker_flags"`
	MarkerDomain       string  `json:"marker_domain"`
	GraphID            uint32  `json:"graph_id"`
	LifecycleEvent     string  `json:"lifecycle_event"`
	TimestampDomain    string  `json:"timestamp_domain"`
	NodeHandle         uint64  `json:"node_handle"`
	SourceGraphHandle  uint64  `json:"source_graph_handle"`
	SourceNodeHandle   uint64  `json:"source_node_handle"`
	NodeID             uint64  `json:"node_id"`
	SourceNodeID       uint64  `json:"source_node_id"`
	GraphHandle        uint64  `json:"graph_handle"`
	ExecutableHandle   uint64  `json:"executable_handle"`
	Success            *bool   `json:"success"`
	Child              bool    `json:"child"`
	GraphNodeID        uint64  `json:"graph_node_id"`
	RuntimeCorrelation uint32  `json:"runtime_correlation"`
	ModuleID           uint32  `json:"module_id"`
	OverheadKind       uint32  `json:"overhead_kind"`
	SyncType           uint32  `json:"sync_type"`
	Dropped            uint64  `json:"dropped"`
	ProcessID          uint32  `json:"process_id"`
	ID                 uint64  `json:"id"`
	Thread             int64   `json:"thread"`
	API                string  `json:"api"`
	Log                string  `json:"log"`
}

// Decode decodes every journal member, reading at most maxRecords records
// and maxBytes decompressed bytes from each. A journal whose frames do not
// decode to completion, or that never finalized, is refused: both leave a
// timeline whose end is unknown, and a truncated prefix is indistinguishable
// from a short run once the records are counted. The one unfinalized journal
// admitted is a process that ran no CUDA (see [NoCUDA]), which has no
// timeline to end.
func Decode(journals map[string][]byte, maxRecords, maxBytes int64) (Records, error) {
	out := Records{APIs: make(map[uint32]Span), contextDevice: make(map[uint32]uint32)}
	names := make([]string, 0, len(journals))
	for name := range journals {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		reader, compressed, err := openJournal(journals[name], maxBytes)
		if err != nil {
			return Records{}, fmt.Errorf("read journal %s: %w", name, err)
		}
		summary := Journal{Name: name, Compressed: compressed}
		if err := decodeJournal(name, reader, maxRecords, &out, &summary); err != nil {
			_ = reader.Close()
			return Records{}, err
		}
		if err := reader.Close(); err != nil {
			return Records{}, fmt.Errorf("close journal %s: %w", name, err)
		}
		summary.Bytes = int(reader.n)
		if summary.Bytes == 0 {
			return Records{}, fmt.Errorf("journal %s is empty, so no device timeline was recorded", name)
		}
		out.Journals = append(out.Journals, summary)
	}
	if len(out.Activities) == 0 {
		return Records{}, errors.New("journals hold no valid activity records, so there is no device timeline to analyze")
	}
	// Resolution runs once every journal has been read, not per journal: the
	// kernel that names a context and the sync that shares it need not have
	// been drained into the same journal member.
	out.resolveSyncDevices()
	return out, nil
}

// NoCUDA decides whether a journal that never finalized came from a process
// that ran no CUDA at all, rather than one that lost its records. The capture
// controller admits such a generation as no-cuda and publishes its journal
// beside the others, and a reader that refuses it refuses a capture the
// controller called complete. A shell or launcher in front of the CUDA
// program is the ordinary case.
//
// A reader fills it with [NoCUDA.Observe] as it decodes; the controller sets
// the fields from its own journal state.
type NoCUDA struct {
	// DeviceActive reports the producer's device_active note: CUPTI asked
	// for an activity buffer.
	DeviceActive bool
	// Failed reports a producer failure record.
	Failed bool
	// Work counts activity records other than the producer's bookkeeping.
	// Any one of them means CUPTI delivered a buffer, which a process that
	// ran no CUDA never has.
	Work uint64
	// Capability is the payload of the producer's capability record.
	Capability json.RawMessage
}

// Observe records one journal record of the given kind.
func (n *NoCUDA) Observe(kind uint16, subtype string, payload []byte) {
	switch kind {
	case wire.KindCapability:
		n.Capability = append(json.RawMessage(nil), payload...)
	case wire.KindFailure:
		n.Failed = true
	case wire.KindActivity:
		switch subtype {
		case "device_active":
			n.DeviceActive = true
		case "capture_range":
		default:
			n.Work++
		}
	}
}

// Proven reports whether the journal ran no CUDA at all.
//
// The evidence is the absence of the producer's device_active note, and that
// absence means nothing unless the producer writes one -- an agent built
// before the note leaves the same journal for a process that lost every
// record as for one that had none, and reading the first as the second turns
// a failed capture into a silently empty success. So the capability record is
// asked whether this producer reports the note at all, and a producer that
// does not gets the refusal it got before.
func (n NoCUDA) Proven() bool {
	if n.DeviceActive || n.Failed || n.Work != 0 {
		return false
	}
	var capability struct {
		DeviceActiveNote bool `json:"device_active_note"`
	}
	return json.Unmarshal(n.Capability, &capability) == nil && capability.DeviceActiveNote
}

// resolveSyncDevices places each synchronization span on the device its
// context belongs to.
//
// A sync that cannot be resolved is counted and dropped rather than defaulted.
// Device 0 is the wrong answer twice over: on a single-device run it is
// indistinguishable from a correct one, so the defect never shows; on a
// multi-device run it explains one device's idle with another device's wait.
func (r *Records) resolveSyncDevices() {
	for _, y := range r.rawSyncs {
		device, ok := r.contextDevice[y.context]
		if !ok {
			r.UnresolvedSyncs++
			r.UnresolvedNanos += y.end - y.start
			continue
		}
		r.Syncs = append(r.Syncs, Span{Device: device, Start: y.start, End: y.end})
	}
}

// OpenJournal returns a reader of the journal frames in data, transparently
// decompressing a gzip-stored member, that fails once it has read more than
// maxBytes. Detection is by content, so a journal is read the same way
// whether or not its name carries the .gz suffix.
func OpenJournal(data []byte, maxBytes int64) (io.ReadCloser, error) {
	r, _, err := openJournal(data, maxBytes)
	if err != nil {
		return nil, err
	}
	return r, nil
}

// openJournal is OpenJournal, also reporting whether data was compressed and
// returning the reader whose count of bytes read Decode reports.
func openJournal(data []byte, maxBytes int64) (*boundedReader, bool, error) {
	if maxBytes <= 0 {
		return nil, false, fmt.Errorf("invalid journal byte limit %d", maxBytes)
	}
	if len(data) < 2 || data[0] != 0x1f || data[1] != 0x8b {
		return newBoundedReader(io.NopCloser(bytes.NewReader(data)), maxBytes), false, nil
	}
	zr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, true, fmt.Errorf("open gzip journal: %w", err)
	}
	return newBoundedReader(zr, maxBytes), true, nil
}

// boundedReader counts the bytes read through it and fails a read that
// would take it past limit.
type boundedReader struct {
	io.ReadCloser
	limit int64
	n     int64
}

func newBoundedReader(r io.ReadCloser, limit int64) *boundedReader {
	return &boundedReader{ReadCloser: r, limit: limit}
}

func (r *boundedReader) Read(p []byte) (int, error) {
	remaining := r.limit - r.n
	if remaining < 0 {
		return 0, r.overLimit()
	}
	if remaining == 0 {
		var one [1]byte
		n, err := r.ReadCloser.Read(one[:])
		if n > 0 {
			return 0, r.overLimit()
		}
		return 0, err
	}
	if int64(len(p)) > remaining {
		p = p[:remaining]
	}
	n, err := r.ReadCloser.Read(p)
	r.n += int64(n)
	return n, err
}

func (r *boundedReader) overLimit() error {
	return &refusal{fmt.Sprintf("journal exceeds %d bytes decompressed", r.limit), ErrJournalBytes}
}

// Measure returns the most records and the most decompressed bytes held by
// any one of journals. It reads every frame but keeps none, so it costs the
// decompression and none of the memory Decode needs; a reader uses it to say
// what bound would admit a journal that Decode refused.
func Measure(journals map[string][]byte) (records, size int64, err error) {
	for name, data := range journals {
		r, _, err := openJournal(data, math.MaxInt64)
		if err != nil {
			return 0, 0, fmt.Errorf("read journal %s: %w", name, err)
		}
		d := wire.NewDecoder(r, wire.Limits{MaxRecords: math.MaxUint32})
		for {
			_, err = d.Next()
			if err != nil {
				break
			}
		}
		r.Close()
		if !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
			return 0, 0, fmt.Errorf("measure journal %s: %w", name, err)
		}
		records = max(records, int64(d.Stats().Records))
		size = max(size, r.n)
	}
	return records, size, nil
}

func decodeJournal(name string, reader io.Reader, maxRecords int64, out *Records, summary *Journal) error {
	limitsMaxRecords := uint32(math.MaxUint32)
	if maxRecords > 0 && maxRecords < math.MaxUint32 {
		limitsMaxRecords = uint32(maxRecords)
	}
	d := wire.NewDecoder(reader, wire.Limits{MaxRecords: limitsMaxRecords})
	first, finalized := true, false
	var noCUDA NoCUDA
	// API records are held until the journal is admitted: a journal set
	// aside as having run no CUDA must leave nothing behind.
	apis := make(map[uint32]Span)
	var calls []Call
	var markers []Marker
	var nvtx []recordPayload
	// So are GEMM calls, which are joined to this journal's kernels by
	// correlation id once it has been read: the launch that links a call to
	// a kernel is made before the kernel's record is drained, and ids are
	// unique only within a process.
	var gemms []GEMM
	gemmIndex := make(map[uint64]int)
	gemmLaunch := make(map[uint32]int)
	firstActivity := len(out.Activities)
	for {
		r, err := d.Next()
		if err != nil {
			switch {
			case errors.Is(err, io.EOF):
				if summary.Records == 0 {
					return fmt.Errorf("journal %s is empty, so no device timeline was recorded", name)
				}
				if !finalized {
					if noCUDA.Proven() {
						summary.NoCUDA = true
						return nil
					}
					return fmt.Errorf("journal %s decoded %d record(s) but never finalized, so the recorded timeline has no known end", name, summary.Records)
				}
				if stats := d.Stats(); stats.Gaps > 0 || stats.Dropped > 0 {
					return fmt.Errorf("journal %s lost records (%d sequence gap(s), %d dropped), so activities the idle gaps would span are absent", name, stats.Gaps, stats.Dropped)
				}
				for c, a := range apis {
					out.APIs[c] = a
				}
				out.Calls = append(out.Calls, calls...)
				out.Markers = append(out.Markers, markers...)
				ranges, unpaired := pairMarkers(nvtx)
				for i := range ranges {
					ranges[i].Journal = name
				}
				out.Ranges = append(out.Ranges, ranges...)
				out.UnpairedMarkers += unpaired
				for _, a := range out.Activities[firstActivity:] {
					if i, ok := gemmLaunch[a.Correlation]; ok && a.Subtype == "kernel" {
						gemms[i].Kernels = append(gemms[i].Kernels, a)
					}
				}
				out.GEMMs = append(out.GEMMs, gemms...)
				return nil
			case errors.Is(err, io.ErrUnexpectedEOF):
				return fmt.Errorf("journal %s is truncated after %d record(s): its final frame is incomplete, so the timeline ends at an unknown point", name, summary.Records)
			default:
				return fmt.Errorf("decode journal %s after %d record(s): %w", name, summary.Records, err)
			}
		}
		if first {
			d.ExpectSession(r.Session)
			first = false
		}
		summary.Records++
		var payload recordPayload
		if len(r.Payload) > 0 && json.Unmarshal(r.Payload, &payload) != nil {
			return fmt.Errorf("journal %s record %d is not a typed capture record", name, r.Sequence)
		}
		noCUDA.Observe(r.Kind, payload.Subtype, r.Payload)
		switch r.Kind {
		case wire.KindActivity:
			if payload.Dropped > 0 {
				return fmt.Errorf("journal %s reports %d dropped record(s), so the timeline is missing activities", name, payload.Dropped)
			}
			// CUPTI_TIMESTAMP_UNKNOWN is zero, including for instantaneous NVTX
			// markers. A contradictory valid=true does not supply a timestamp.
			if !payload.Valid || payload.Start == 0 || payload.End < payload.Start || payload.End > math.MaxInt64 {
				continue
			}
			// Neither an NVTX marker nor a capture-range edge is device
			// work: the first is a host annotation, the second is the
			// producer stating when it started and stopped collecting.
			// Counting either would inflate Activities, stretch SpanNanos to
			// whichever one sits furthest from the device timeline, and split
			// a real device gap in two at its timestamp.
			if payload.Subtype == "marker" {
				markers = append(markers, Marker{
					Sequence: r.Sequence,
					Journal:  name,
					Process:  payload.ProcessID,
					Thread:   payload.ThreadID,
					Name:     payload.Name,
					Domain:   payload.MarkerDomain,
					ID:       payload.MarkerID,
					Flags:    payload.MarkerFlags,
					Time:     int64(payload.Start),
				})
				nvtx = append(nvtx, payload)
				continue
			}
			if payload.Subtype == "capture_range" {
				continue
			}
			// The same exclusion, for three more kinds and a sharper reason.
			// This filter is a denylist -- anything it does not name counts
			// as device busy -- so a span that merely explains idle, added
			// upstream and not named here, would be subtracted from the idle
			// it was added to explain, and nothing would fail. That is the
			// whole trap:
			//
			//   overhead    CUPTI's own cost, paid on a host thread. Counting
			//               it as busy makes the tool's headline idle number
			//               shrink in proportion to how much the tool cost.
			//   sync        the host waiting. This is the reason the device
			//               was idle; charging it to busy erases the idle it
			//               explains, exactly.
			//   graph_trace the replay envelope, which spans the kernel
			//               records of the nodes replayed inside it. It would
			//               both double-count their device time and paper
			//               over the scheduling gaps between them.
			//
			// Each is kept as a span instead and intersected with the idle
			// gaps below, which is attribution rather than occupancy.
			switch payload.Subtype {
			case "unified_memory":
				out.UnifiedMemory = append(out.UnifiedMemory, UnifiedMemory{Sequence: r.Sequence, Journal: name, Process: payload.ProcessID, Start: int64(payload.Start), End: int64(payload.End), Valid: payload.Valid, CounterKind: payload.CounterKind, Value: payload.Value, Address: payload.Address, RawEnd: payload.RawEnd, EndValid: payload.EndValid, Source: payload.Source, Destination: payload.Destination, SourceValid: payload.SourceValid, DestinationValid: payload.DestinationValid, RawStream: payload.RawStream, Flags: payload.Flags})
				continue
			case "overhead":
				out.Overheads = append(out.Overheads, Span{Start: int64(payload.Start), End: int64(payload.End)})
				continue
			case "graph_trace":
				out.Graphs = append(out.Graphs, Activity{
					Sequence:    r.Sequence,
					Journal:     name,
					Device:      payload.Device,
					Start:       int64(payload.Start),
					End:         int64(payload.End),
					Correlation: payload.Correlation,
					Subtype:     payload.Subtype,
					Name:        payload.Name,
					Stream:      payload.Stream,
					GraphID:     payload.GraphID,
					Context:     payload.Context,
				})
				continue
			case "sync":
				out.rawSyncs = append(out.rawSyncs, rawSync{context: payload.Context, start: int64(payload.Start), end: int64(payload.End)})
				continue
			}
			// A record that carries both a context and the device it ran on
			// is the only source for the map a synchronization record has to
			// be resolved through. First writer wins: a context belongs to
			// one device for its lifetime, and a later disagreement is a
			// reused id, not a migration.
			if payload.GraphID != 0 {
				out.GraphNodes++
			}
			if payload.Context != 0 {
				if _, ok := out.contextDevice[payload.Context]; !ok {
					out.contextDevice[payload.Context] = payload.Device
				}
			}
			out.Activities = append(out.Activities, Activity{
				Sequence:           r.Sequence,
				Journal:            name,
				Device:             payload.Device,
				Start:              int64(payload.Start),
				End:                int64(payload.End),
				Correlation:        payload.Correlation,
				Subtype:            payload.Subtype,
				Name:               payload.Name,
				Stream:             payload.Stream,
				GraphID:            payload.GraphID,
				GraphNodeID:        payload.GraphNodeID,
				Context:            payload.Context,
				RuntimeCorrelation: payload.RuntimeCorrelation,
				Grid:               [3]int{int(payload.GridX), int(payload.GridY), int(payload.GridZ)},
				Block:              [3]int{int(payload.BlockX), int(payload.BlockY), int(payload.BlockZ)},
				SharedMemory:       payload.SharedMemory,
				Bytes:              payload.Bytes,
				CopyKind:           payload.CopyKind,
				Module:             payload.ModuleID,
			})
		case wire.KindAPI:
			if payload.Kind == "memory_policy" {
				var event MemoryPolicy
				if err := json.Unmarshal(r.Payload, &event); err != nil {
					return fmt.Errorf("decode memory policy record: %w", err)
				}
				if event.Time < 0 {
					return fmt.Errorf("invalid memory policy timestamp")
				}
				event.Journal, event.Sequence = name, r.Sequence
				out.MemoryPolicies = append(out.MemoryPolicies, event)
				continue
			}
			if payload.Kind == "dependency" {
				var event DependencyEvent
				if err := json.Unmarshal(r.Payload, &event); err != nil {
					return fmt.Errorf("decode dependency record: %w", err)
				}
				event.Journal, event.Sequence = name, r.Sequence
				out.Dependencies = append(out.Dependencies, event)
				continue
			}
			if payload.End < payload.Start || payload.End > math.MaxInt64 {
				continue
			}
			if payload.Kind == "graph_lifecycle" {
				out.GraphLifecycle = append(out.GraphLifecycle, GraphEvent{Sequence: r.Sequence, Journal: name, Domain: payload.Domain, TimestampDomain: payload.TimestampDomain, CallbackID: payload.CallbackID, Context: payload.Context, Process: payload.ProcessID, Thread: payload.ThreadID, NodeHandle: payload.NodeHandle, NodeID: payload.NodeID, SourceNodeID: payload.SourceNodeID, SourceGraphHandle: payload.SourceGraphHandle, SourceNodeHandle: payload.SourceNodeHandle, Kind: payload.LifecycleEvent, Graph: payload.GraphHandle, Executable: payload.ExecutableHandle, Correlation: payload.Correlation, Start: int64(payload.Start), End: int64(payload.End), Success: payload.Success, Child: payload.Child})
				continue
			}
			if payload.End < payload.Start || payload.End > math.MaxInt64 {
				continue
			}
			apis[payload.Correlation] = Span{Start: int64(payload.Start), End: int64(payload.End), Thread: payload.ThreadID}
			calls = append(calls, Call{
				Sequence:    r.Sequence,
				Journal:     name,
				Process:     payload.ProcessID,
				Domain:      payload.Domain,
				CallbackID:  payload.CallbackID,
				Thread:      payload.ThreadID,
				Correlation: payload.Correlation,
				Start:       int64(payload.Start),
				End:         int64(payload.End),
			})
		case wire.KindGEMM:
			switch payload.Kind {
			case "gemm":
				gemmIndex[payload.ID] = len(gemms)
				gemms = append(gemms, GEMM{Journal: name, ID: payload.ID, Thread: payload.Thread, Time: int64(r.Timestamp), API: payload.API, Log: payload.Log})
			case "gemm_launch":
				if i, ok := gemmIndex[payload.ID]; ok && payload.Correlation != 0 {
					gemms[i].Launches++
					gemmLaunch[payload.Correlation] = i
				}
			}
		case wire.KindFinalized:
			finalized = true
			if payload.Dropped > 0 {
				return fmt.Errorf("journal %s finalized with %d dropped record(s), so the timeline is missing activities", name, payload.Dropped)
			}
		}
	}
}
