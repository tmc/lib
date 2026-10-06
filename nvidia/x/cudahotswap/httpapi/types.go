package httpapi

import (
	"context"
	"encoding/json"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Service is the transport's complete service surface. Implementations must
// preserve the ownership and completion contracts of cudahotswap.Service.
type Service interface {
	Err() error
	Register(context.Context, string, cudahotswap.Source, string) (cudahotswap.State, error)
	Push(context.Context, string, cudahotswap.Source, string) (*cudahotswap.PushResult, error)
	List() []cudahotswap.State
	Status(string) (cudahotswap.State, error)
	Versions(string) ([]cudahotswap.Version, error)
	Select(context.Context, string, uint64, string) (cudahotswap.State, error)
	Revert(context.Context, string, string) (cudahotswap.State, error)
	SetCompileFlags(string, []string, string) (cudahotswap.State, error)
	SetGeometry(string, *cudahotswap.LaunchGeometry, string) (cudahotswap.State, error)
	SetCheck(string, cudahotswap.CheckConfig, string) (cudahotswap.State, error)
	SetWatchPaths(string, *cudahotswap.WatchPaths, string) (cudahotswap.State, error)
	WatchPaths(string) (*cudahotswap.WatchPaths, error)
	Watch(context.Context, cudahotswap.WatchConfig, func(string, *cudahotswap.PushResult, error) error) error
	Subscribe(context.Context, uint64) (*cudahotswap.Subscription, error)
	Pin(string) (*cudahotswap.Pin, error)
	Launch(context.Context, string, cudalaunch.Config, ...any) (*cudahotswap.PendingLaunch, error)
	RecordFixture(context.Context, string, cudahotswap.FixtureConfig) (*cudahotswap.Fixture, error)
	RegisterGraph(context.Context, string, *cudagraph.Exec, *cudagraph.Node, *cuda.Stream, driver.Dim3, ...any) (cudahotswap.State, error)
	LaunchGraph(context.Context, *cudagraph.Exec) (*cudahotswap.GraphLaunch, error)
	Close(context.Context) error
}

// Graph names an application-owned graph binding. Args and all referenced
// native allocations must remain live until Handler.Close succeeds.
type Graph struct {
	Exec   *cudagraph.Exec
	Node   *cudagraph.Node
	Stream *cuda.Stream
	Grid   driver.Dim3
	Args   []any
}

// Resources names objects supplied by the embedding application. Clients cannot
// supply device addresses, Go functions, streams or graph handles. The handler
// copies the maps; the application keeps the native owners alive and excludes
// concurrent buffer writes until Handler.Close succeeds. Checkers implement the
// application's reference policy; a transport never invents a correctness hook.
type Resources struct {
	// Arguments fixes each kernel's application-declared argument kinds.
	// PTX integer storage alone does not prove scalar versus pointer intent.
	// Launch and RecordFixture fail closed without this binding.
	Arguments map[string][]string
	Checkers  map[string]cudahotswap.Checker
	Buffers   map[string]cudamem.Slice[byte]
	Streams   map[string]*cuda.Stream
	Graphs    map[string]Graph
	// Stages fixes each stage's builder, schema and named launch resources.
	// All owners remain live until Handler.Close succeeds.
	Stages map[string]StageBinding
}

// Argument is a typed scalar or a named application buffer. Supported scalar
// types are u32, i32, u64, i64, f32 and f64. Buffer arguments have Type "buffer"
// and Name, with no Value. Pointer integers are never accepted.
type Argument struct {
	Type  string          `json:"type"`
	Name  string          `json:"name,omitempty"`
	Value json.RawMessage `json:"value,omitempty"`
}

// Buffer describes a fixture buffer by its server registry name and byte extent.
type Buffer struct {
	Name   string `json:"name"`
	Arg    int    `json:"arg"`
	Bytes  int    `json:"bytes"`
	Stable bool   `json:"stable"`
}

// Request is the bounded JSON request shared by endpoints. An endpoint checks
// required fields before calling its corresponding service method.
type Request struct {
	Name          string                      `json:"name,omitempty"`
	Label         string                      `json:"label,omitempty"`
	Source        cudahotswap.Source          `json:"source,omitempty"`
	StageCode     string                      `json:"stage_code,omitempty"`
	Version       uint64                      `json:"version,omitempty"`
	Flags         []string                    `json:"flags,omitempty"`
	Geometry      *cudahotswap.LaunchGeometry `json:"geometry,omitempty"`
	Paths         *cudahotswap.WatchPaths     `json:"paths,omitempty"`
	Checker       string                      `json:"checker,omitempty"`
	Fixture       string                      `json:"fixture,omitempty"`
	Policy        *cudahotswap.ReplayPolicy   `json:"policy,omitempty"`
	KeepOnFailure bool                        `json:"keep_on_failure,omitempty"`
	Stream        string                      `json:"stream,omitempty"`
	Graph         string                      `json:"graph,omitempty"`
	Args          []Argument                  `json:"args,omitempty"`
	Buffers       []Buffer                    `json:"buffers,omitempty"`
	Outputs       []cudahotswap.FixtureOutput `json:"outputs,omitempty"`
	Watch         cudahotswap.WatchConfig     `json:"watch,omitempty"`
	Handle        string                      `json:"handle,omitempty"`
}

// Failure preserves a service failure without turning a failed check into a
// transport success. Non-poisoning service failures have HTTP status 200.
type Failure struct {
	Message  string              `json:"message"`
	Outcome  cudahotswap.Outcome `json:"outcome"`
	Poisoned bool                `json:"poisoned,omitempty"`
}

// Response returns service values and opaque lifetime handles. A nonempty
// Handle with Failure still owns resources and must be waited or released.
type Response struct {
	// RequestLabel identifies the transport request. It never replaces the
	// service's State.Label or invents a generation or configuration revision.
	RequestLabel string                  `json:"request_label,omitempty"`
	State        *cudahotswap.State      `json:"state,omitempty"`
	States       []cudahotswap.State     `json:"states,omitempty"`
	Versions     []cudahotswap.Version   `json:"versions,omitempty"`
	Push         *cudahotswap.PushResult `json:"push,omitempty"`
	Paths        *cudahotswap.WatchPaths `json:"paths,omitempty"`
	Handle       string                  `json:"handle,omitempty"`
	Failure      *Failure                `json:"failure,omitempty"`
}
