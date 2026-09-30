package timeline

// DependencyEvent is a native callback receipt. Ordinal orders callback
// observations within Journal, independently of queue drain order. Enter and
// exit receipts share Domain and Correlation; only successful paired receipts
// establish a submission. Handles are process-local and may be reused after
// destruction. TimestampDomain is monotonic only with a unified CUPTI clock.
type DependencyEvent struct {
	Journal         string   `json:"-"`
	Sequence        uint64   `json:"-"`
	Domain          string   `json:"domain"`
	Kind            string   `json:"dependency_event"`
	Site            string   `json:"site"`
	Ordinal         uint64   `json:"ordinal"`
	CallbackID      uint32   `json:"callback_id"`
	Correlation     uint32   `json:"correlation"`
	Context         uint32   `json:"context"`
	Stream          uint64   `json:"stream_handle"`
	Event           uint64   `json:"event_handle"`
	Flags           uint32   `json:"flags"`
	Graph           uint64   `json:"graph_handle"`
	Executable      uint64   `json:"executable_handle"`
	Node            uint64   `json:"graph_node_id"`
	NodeHandle      uint64   `json:"node_handle"`
	ChildGraph      uint64   `json:"child_graph_handle"`
	Dependencies    []uint64 `json:"dependencies"`
	Start           int64    `json:"start"`
	End             int64    `json:"end"`
	TimestampDomain string   `json:"timestamp_domain"`
	Success         *bool    `json:"success"`
}
