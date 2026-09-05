package dcgm

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// This file is hand-written. It exists for one reason: DCGM's objects outlive
// the process that created them, and the raw API gives no way to tell an
// object this process created from one it merely found.
//
// A host engine keeps its groups, field groups and watches until something
// removes them. A monitoring agent, a Kubernetes device plugin and this
// program can all be attached to the same host engine at once. Calling
// dcgmGroupDestroy on a group id another process created, or dcgmStopEmbedded
// on a host engine this process did not start, is silently destructive: it
// succeeds, and the other program's monitoring stops.
//
// A Session therefore records what it created and removes exactly that.

// A Session is a host-engine connection together with the objects this process
// created through it.
//
// Ownership is the whole point of the type:
//
//   - Objects a Session created -- groups from [Session.NewGroup], field
//     groups from [Session.NewFieldGroup], watches from [Session.Watch] --
//     are owned by it, and [Session.Close] removes them.
//   - Objects it merely found, through [Handle.Groups] or the pseudo-groups
//     such as [GroupAllGPUs], are not owned, and Close does not touch them.
//   - A host engine reached with [Connect] is not owned. Close disconnects
//     from it and never stops it. Only a Session from [StartEmbedded] owns a
//     host engine, and only that Session stops one.
//
// [Session.Handle] is exported so the whole generated API remains reachable.
// Anything created through that handle directly is the caller's to remove;
// the Session does not know about it.
//
// A Session is safe for concurrent use. The library calls it makes are the
// library's own to serialize.
type Session struct {
	// Handle is the host-engine handle. It stays valid until Close.
	Handle Handle

	mu          sync.Mutex
	closed      bool
	embedded    bool
	groups      []GpuGrp
	fieldGroups []FieldGrp
	watches     []watchRecord
}

// A watchRecord is one watch this Session established, kept so Close can
// remove that watch and no other.
type watchRecord struct {
	group      GpuGrp
	fieldGroup FieldGrp
}

// ErrClosed reports use of a Session after Close.
var ErrClosed = errors.New("dcgm: session is closed")

// StartEmbedded starts a host engine inside this process and returns a Session
// that owns it.
//
// [Init] must have been called first; DCGM requires it before any other entry
// point and this package does not call it implicitly, because [Shutdown] is
// process-wide and a package that started the library would have to guess when
// to stop it.
//
// The embedded engine runs in this process, so it disappears when the process
// does. In [OperationModeAuto] a background thread samples watched fields; in
// [OperationModeManual] nothing is sampled until [Handle.UpdateAllFields] is
// called, which is the mode to use when samples must line up with something
// the program is doing.
//
// [Session.Close] stops the engine. Nothing else should: an embedded engine is
// this process's, and stopping it from elsewhere would strand this Session's
// handle.
func StartEmbedded(mode OperationMode) (*Session, error) {
	params := StartEmbeddedV2ParamsV1{
		Version:  StartEmbeddedV2ParamsVersion1,
		OpMode:   mode,
		Severity: DcgmLoggingSeverityUnspecified,
	}
	if err := StartEmbeddedV2(&params); err != nil {
		return nil, err
	}
	if params.DcgmHandle == 0 {
		return nil, errors.New("dcgm: StartEmbedded returned success with a zero handle")
	}
	return &Session{Handle: params.DcgmHandle, embedded: true}, nil
}

// ConnectOptions are the options for [Connect]. The zero value connects to a
// TCP/IP address with DCGM's own timeout and does not persist state.
type ConnectOptions struct {
	// Timeout bounds the connection attempt. Zero leaves DCGM's default,
	// which the header does not state; pass an explicit value when the
	// caller has a deadline of its own.
	Timeout time.Duration

	// UnixSocket says the address is a filesystem socket path rather than a
	// TCP/IP address.
	UnixSocket bool

	// PersistAfterDisconnect leaves the watches this connection created in
	// place after it goes away.
	//
	// It is false by default, and that default is the safe one. DCGM
	// normally removes a connection's watches when the connection ends, so
	// a crash or a missed Close does not leave this program's watches
	// running on a host engine it no longer talks to. Setting it true makes
	// those watches outlive the process, and nothing in this package can
	// then remove them.
	PersistAfterDisconnect bool
}

// Connect attaches to a host engine this process did not start, at address --
// a TCP/IP address, or a Unix socket path when opts.UnixSocket is set.
//
// [Init] must have been called first.
//
// The returned Session does not own the host engine. [Session.Close]
// disconnects from it and leaves it running, because something else started it
// and something else is monitoring through it.
func Connect(address string, opts ConnectOptions) (*Session, error) {
	if address == "" {
		return nil, errors.New("dcgm: Connect: empty address")
	}
	params := ConnectV2Params{
		Version:   ConnectV2ParamsVersion2,
		TimeoutMs: uint32(opts.Timeout.Milliseconds()),
	}
	if opts.PersistAfterDisconnect {
		params.PersistAfterDisconnect = 1
	}
	if opts.UnixSocket {
		params.AddressIsUnixSocket = 1
	}
	cstr := append([]byte(address), 0)
	h, err := ConnectV2(&cstr[0], &params)
	runtime.KeepAlive(cstr)
	if err != nil {
		return nil, err
	}
	if h == 0 {
		return nil, errors.New("dcgm: Connect returned success with a zero handle")
	}
	return &Session{Handle: h}, nil
}

// NewGroup creates an entity group and records it as this Session's, so
// [Session.Close] destroys it.
//
// The name must be unique on the host engine; DCGM rejects a duplicate. Use
// [GroupEmpty] and [Session.AddEntity] to choose the members, or one of the
// GroupDefault types to have DCGM populate the group.
func (s *Session) NewGroup(name string, groupType GroupType) (GpuGrp, error) {
	if name == "" {
		return 0, errors.New("dcgm: NewGroup: empty group name")
	}
	if len(name) >= MaxStrLength {
		return 0, fmt.Errorf("dcgm: NewGroup: name is %d bytes, DCGM allows %d including the terminator", len(name), MaxStrLength)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrClosed
	}
	cstr := append([]byte(name), 0)
	id, err := GroupCreate(s.Handle, groupType, &cstr[0])
	runtime.KeepAlive(cstr)
	if err != nil {
		return 0, err
	}
	s.groups = append(s.groups, id)
	return id, nil
}

// AddEntity adds one entity to a group. It is here rather than left to the
// generated method only so a caller building an owned group does not have to
// mix the two APIs; it does not change ownership.
func (s *Session) AddEntity(groupID GpuGrp, entityGroup FieldEntityGroup, entityID FieldEid) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	return GroupAddEntity(s.Handle, groupID, entityGroup, entityID)
}

// NewFieldGroup creates a field group holding the given field ids and records
// it as this Session's, so [Session.Close] destroys it.
//
// The field ids are the [FieldID] constants generated from dcgm_fields.h.
// fields is read during the call and not retained.
func (s *Session) NewFieldGroup(name string, fields []FieldID) (FieldGrp, error) {
	if name == "" {
		return 0, errors.New("dcgm: NewFieldGroup: empty field group name")
	}
	if len(name) >= MaxStrLength {
		return 0, fmt.Errorf("dcgm: NewFieldGroup: name is %d bytes, DCGM allows %d including the terminator", len(name), MaxStrLength)
	}
	if len(fields) == 0 {
		return 0, errors.New("dcgm: NewFieldGroup: no fields")
	}
	if len(fields) > MaxFieldIDsPerFieldGroup {
		return 0, ErrTooManyFields
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, ErrClosed
	}
	cstr := append([]byte(name), 0)
	ids := fieldIDsToUint16(fields)
	id, err := FieldGroupCreate(s.Handle, int32(len(ids)), &ids[0], &cstr[0])
	runtime.KeepAlive(cstr)
	runtime.KeepAlive(ids)
	if err != nil {
		return 0, err
	}
	s.fieldGroups = append(s.fieldGroups, id)
	return id, nil
}

// Watch tells the host engine to sample fieldGroup's fields for groupID's
// entities, and records the watch as this Session's so [Session.Close] removes
// it.
//
// updateFreq is how often the engine samples. maxKeepAge is how long samples
// are retained; maxKeepSamples caps how many are, and zero means unlimited. A
// watch is host-engine state that outlives this process unless it is removed,
// which is why it is recorded rather than left to the caller to remember.
//
// Watching does not produce a sample. In [OperationModeManual] nothing is
// sampled until [Handle.UpdateAllFields]; in [OperationModeAuto] the first
// sample arrives no sooner than updateFreq, so a read immediately after Watch
// returns blank values rather than an error.
func (s *Session) Watch(groupID GpuGrp, fieldGroup FieldGrp, updateFreq, maxKeepAge time.Duration, maxKeepSamples int32) error {
	if updateFreq <= 0 {
		return errors.New("dcgm: Watch: update frequency must be positive")
	}
	if maxKeepAge < 0 {
		return errors.New("dcgm: Watch: maximum sample age must not be negative")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	err := WatchFields(s.Handle, groupID, fieldGroup, updateFreq.Microseconds(), maxKeepAge.Seconds(), maxKeepSamples)
	if err != nil {
		return err
	}
	s.watches = append(s.watches, watchRecord{group: groupID, fieldGroup: fieldGroup})
	return nil
}

// Close removes what this Session created and releases the host engine.
//
// The order is the reverse of creation: watches, then field groups, then
// groups, then the connection. Nothing else is touched. In particular:
//
//   - A group or field group obtained from anywhere but [Session.NewGroup] or
//     [Session.NewFieldGroup] is left alone, because another process may be
//     using it.
//   - A host engine reached with [Connect] is disconnected from, never
//     stopped.
//   - [Shutdown] is not called. It is process-wide, and another Session may
//     still be open.
//
// Close reports the first error and continues, so one failing teardown step
// does not strand the rest. It is idempotent: a second call returns nil.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true

	var firstErr error
	fail := func(err error) {
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	for i := len(s.watches) - 1; i >= 0; i-- {
		w := s.watches[i]
		fail(UnwatchFields(s.Handle, w.group, w.fieldGroup))
	}
	s.watches = nil
	for i := len(s.fieldGroups) - 1; i >= 0; i-- {
		fail(FieldGroupDestroy(s.Handle, s.fieldGroups[i]))
	}
	s.fieldGroups = nil
	for i := len(s.groups) - 1; i >= 0; i-- {
		fail(GroupDestroy(s.Handle, s.groups[i]))
	}
	s.groups = nil

	if s.embedded {
		fail(StopEmbedded(s.Handle))
	} else {
		fail(Disconnect(s.Handle))
	}
	s.Handle = 0
	return firstErr
}

// Owns reports whether this Session created the given group and would destroy
// it on Close. It is here so a caller holding a group id from elsewhere can
// tell whether removing it is its own responsibility.
func (s *Session) Owns(groupID GpuGrp) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range s.groups {
		if id == groupID {
			return true
		}
	}
	return false
}

// OwnsFieldGroup reports whether this Session created the given field group.
func (s *Session) OwnsFieldGroup(fieldGroup FieldGrp) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range s.fieldGroups {
		if id == fieldGroup {
			return true
		}
	}
	return false
}

// OwnsHostEngine reports whether this Session started the host engine and will
// stop it on Close. It is false for every Session from [Connect].
func (s *Session) OwnsHostEngine() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.embedded
}
