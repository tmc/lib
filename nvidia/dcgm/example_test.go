package dcgm_test

import (
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

// Example watches two fields on every supported GPU and prints the most recent
// sample of each. It returns early when DCGM is not available, which is what
// running it on a machine without the library or without a host engine does.
//
// The shape is the one every DCGM read has: watch first, let the host engine
// sample, then retrieve. A read before the first sample succeeds and returns
// blank values rather than failing, so Blank is checked on every reading.
func Example() {
	if err := dcgm.Init(); err != nil {
		fmt.Println("DCGM unavailable:", err)
		return
	}
	defer dcgm.Shutdown()

	// StartEmbedded runs the host engine in this process, so this Session
	// owns it and Close stops it. dcgm.Connect would attach to an engine
	// something else started, and Close would only disconnect.
	s, err := dcgm.StartEmbedded(dcgm.OperationModeManual)
	if err != nil {
		fmt.Println("start embedded host engine:", err)
		return
	}
	defer s.Close()

	gpus, err := s.Handle.SupportedDevices()
	if err != nil {
		fmt.Println("supported devices:", err)
		return
	}
	if len(gpus) == 0 {
		fmt.Println("no DCGM-supported GPUs on this host")
		return
	}

	// The group and the field group are created through the Session, so
	// Close destroys them. Created through the raw entry points they would
	// outlive the process and stay on the host engine.
	group, err := s.NewGroup("example-group", dcgm.GroupDefault)
	if err != nil {
		fmt.Println("create group:", err)
		return
	}
	fields := []dcgm.FieldID{dcgm.FieldDevGpuTempCelsius, dcgm.FieldDevSmClock}
	fieldGroup, err := s.NewFieldGroup("example-fields", fields)
	if err != nil {
		fmt.Println("create field group:", err)
		return
	}
	if err := s.Watch(group, fieldGroup, time.Second, time.Minute, 100); err != nil {
		fmt.Println("watch:", err)
		return
	}

	// In OperationModeManual nothing is sampled until this call. In
	// OperationModeAuto a background thread samples, and a read this soon
	// after Watch would come back blank.
	if err := dcgm.UpdateAllFields(s.Handle, 1); err != nil {
		fmt.Println("update fields:", err)
		return
	}

	for _, gpu := range gpus {
		values, err := s.Handle.LatestValues(gpu, fields)
		if err != nil {
			fmt.Println("latest values:", err)
			return
		}
		for i := range values {
			v := &values[i]
			switch {
			case v.Err() != nil:
				fmt.Printf("gpu %d %v: %v\n", gpu, v.ID(), v.Err())
			case v.Blank():
				// Not an error. The field is unsupported on this
				// entity, or no sample has arrived yet.
				fmt.Printf("gpu %d %v: no data\n", gpu, v.ID())
			default:
				n, _ := v.Int64()
				fmt.Printf("gpu %d %v = %d at %s\n", gpu, v.ID(), n, v.Time().Format(time.RFC3339))
			}
		}
	}
}

// ExampleConnect attaches to a host engine that is already running, which is
// the usual arrangement: nv-hostengine runs as a service and several clients
// read from it at once.
//
// The Session returned does not own that host engine. Close disconnects and
// leaves it running, because stopping it would stop everyone else's
// monitoring.
func ExampleConnect() {
	if err := dcgm.Init(); err != nil {
		fmt.Println("DCGM unavailable:", err)
		return
	}
	defer dcgm.Shutdown()

	s, err := dcgm.Connect("127.0.0.1:5555", dcgm.ConnectOptions{Timeout: 5 * time.Second})
	if err != nil {
		fmt.Println("connect:", err)
		return
	}
	defer s.Close()

	fmt.Println("owns the host engine:", s.OwnsHostEngine())

	// Groups found this way belong to whoever created them. Destroying one
	// would take it from its owner, so this only reads.
	groups, err := s.Handle.Groups()
	if err != nil {
		fmt.Println("groups:", err)
		return
	}
	for _, g := range groups {
		entities, err := s.Handle.GroupEntities(g)
		if err != nil {
			fmt.Printf("group %d: %v\n", g, err)
			continue
		}
		fmt.Printf("group %d has %d entities, owned by this session: %v\n", g, len(entities), s.Owns(g))
	}
}
