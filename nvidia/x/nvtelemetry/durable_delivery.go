package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// DurableDeliveryConfig sets explicit bounds for disk-spooled OTLP HTTP delivery.
// SpoolDir must be an exclusively task-owned directory private to the calling
// user (permissions 0700, no group/other access). Owner is a non-empty,
// non-secret task identifier (printable ASCII, max 128 bytes).
// MaxSpoolBatches bounds the maximum number of spooled batches stored on disk.
// MaxSpoolBytes bounds the cumulative disk bytes of spooled batch payloads.
// Delivery sets HTTP transport bounds and timeouts.
type DurableDeliveryConfig struct {
	Delivery        DeliveryConfig
	SpoolDir        string
	Owner           string
	MaxSpoolBatches int
	MaxSpoolBytes   int64
}

// DurableDeliveryStats records cumulative durable delivery outcomes.
// It separates producer admission drops (budget exhaustion) from downstream
// delivery failures.
type DurableDeliveryStats struct {
	DeliveryStats

	SpooledBatches    uint64 // Batches successfully committed to disk
	SpooledBytes      uint64 // Payload bytes committed to disk
	RecoveredBatches  uint64 // Batches recovered from disk on startup
	PendingSpool      int    // Current uncompleted batches on disk
	PendingSpoolBytes int64  // Current uncompleted payload bytes on disk
}

// DurableDelivery sends bounded OTLP HTTP JSON batches on one background worker
// backed by a persistent disk spool queue. The zero value is not usable.
//
// Delivery semantics:
//
//   - Durable: Once Enqueue returns nil, the batch has been validated and
//     committed to the private, task-owned spool directory on disk via file
//     and directory fsync. Spooled batches survive graceful and ungraceful
//     process termination. Durability against host power loss depends on the
//     underlying filesystem mount and disk write-barrier semantics; callers
//     must not assume guarantees stronger than POSIX fsync.
//
//   - Duplicated: Delivery guarantees at-least-once transport within configured
//     retry bounds. If a network interruption, client timeout, or process
//     restart occurs before an HTTP 200 response acknowledgment is processed,
//     the spooled batch will be retried or recovered on restart. Exactly-once
//     delivery is impossible across distributed systems and is never promised.
//
//   - Discarded: When the spool queue or byte budget is exhausted, Enqueue
//     immediately rejects new batches with ErrDeliveryFull and increments
//     DroppedBatches (producer loss). If an admitted batch encounters an
//     unretryable HTTP status (e.g. 4xx client errors) or exhausts MaxAttempts,
//     it is removed from the spool to preserve queue progress, incrementing
//     FailedBatches (delivery loss).
//
//   - Unknown: When Close times out or is cancelled, pending spooled batches
//     remain on disk; their eventual delivery status is unknown until a
//     subsequent process instance recovers the spool directory.
//
//   - Privacy & Security: Authorization credentials, endpoint query strings,
//     payload contents, and raw response bodies are never exposed in error
//     messages, statistics, or logs.
type DurableDelivery struct {
	mu               sync.Mutex
	config           DurableDeliveryConfig
	isLogs           bool
	root             *os.Root
	lockFile         *os.File
	spoolDir         string
	queue            []spooledBatch
	stats            DurableDeliveryStats
	nextID           uint64
	closing          bool
	workerDone       bool
	leaseReleased    bool
	terminal         error
	activeAdmissions int
	wake             chan struct{}
	done             chan struct{}
	flushWake        chan struct{}
	shutdownExpired  chan struct{}
	cancel           context.CancelFunc
	closeOnce        sync.Once
	leaseOnce        sync.Once
}

type spooledBatch struct {
	id      uint64
	name    string
	size    int64
	payload []byte
}

// DurableDeliveryRecoverySchemaVersion identifies the recovery manifest schema for DurableDelivery.
const DurableDeliveryRecoverySchemaVersion = 1

// DurableDeliveryRecoveryManifest records the versioned recovery state of an owned durable delivery spool.
type DurableDeliveryRecoveryManifest struct {
	SchemaVersion    int    `json:"schema_version"`
	MarkerVersion    int    `json:"marker_version"`
	Owner            string `json:"owner"`
	RecoveredLease   bool   `json:"recovered_lease"`
	RemovedTempFiles int    `json:"removed_temp_files"`
	PendingBatches   int    `json:"pending_batches"`
	PendingBytes     int64  `json:"pending_bytes"`
}

// ValidateDurableDeliveryRecoveryManifest checks schema version, marker version, owner identity, and bounds.
func ValidateDurableDeliveryRecoveryManifest(m DurableDeliveryRecoveryManifest) error {
	if m.SchemaVersion != DurableDeliveryRecoverySchemaVersion {
		return fmt.Errorf("telemetry delivery recovery manifest: unsupported schema version %d", m.SchemaVersion)
	}
	if m.MarkerVersion != 1 {
		return fmt.Errorf("telemetry delivery recovery manifest: unsupported marker version %d", m.MarkerVersion)
	}
	if len(m.Owner) == 0 || len(m.Owner) > 128 {
		return fmt.Errorf("telemetry delivery recovery manifest: invalid owner")
	}
	for _, b := range m.Owner {
		if b < '!' || b > '~' {
			return fmt.Errorf("telemetry delivery recovery manifest: invalid owner character")
		}
	}
	if m.RemovedTempFiles < 0 || m.RemovedTempFiles > 65536 || m.PendingBatches < 0 || m.PendingBatches > 65536 || m.PendingBytes < 0 || m.PendingBytes > 1<<30 {
		return fmt.Errorf("telemetry delivery recovery manifest: invalid bounds")
	}
	return nil
}

// WriteDurableDeliveryRecoveryManifest validates and writes one JSON recovery manifest line.
func WriteDurableDeliveryRecoveryManifest(w io.Writer, m DurableDeliveryRecoveryManifest) error {
	if w == nil {
		return fmt.Errorf("telemetry delivery recovery manifest: nil writer")
	}
	if err := ValidateDurableDeliveryRecoveryManifest(m); err != nil {
		return err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

// ReadDurableDeliveryRecoveryManifest strictly decodes and validates a JSON recovery manifest.
func ReadDurableDeliveryRecoveryManifest(r io.Reader) (DurableDeliveryRecoveryManifest, error) {
	if r == nil {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery recovery manifest: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, 4097))
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, err
	}
	if len(b) == 0 || len(b) > 4096 {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery recovery manifest: invalid byte length")
	}
	var m DurableDeliveryRecoveryManifest
	if err := decodeOperationsJSON(b, &m); err != nil {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery recovery manifest: decode: %w", err)
	}
	if err := ValidateDurableDeliveryRecoveryManifest(m); err != nil {
		return DurableDeliveryRecoveryManifest{}, err
	}
	return m, nil
}

type spoolOwnerMarker struct {
	Version       int    `json:"version"`
	SchemaVersion int    `json:"schema_version,omitempty"`
	Owner         string `json:"owner"`
}

func validateDurableDeliveryConfig(c *DurableDeliveryConfig) error {
	if c == nil {
		return fmt.Errorf("telemetry delivery: nil configuration")
	}
	if len(c.Owner) == 0 || len(c.Owner) > 128 {
		return fmt.Errorf("telemetry delivery: invalid owner")
	}
	for _, b := range c.Owner {
		if b < '!' || b > '~' {
			return fmt.Errorf("telemetry delivery: invalid owner character")
		}
	}
	if c.SpoolDir == "" {
		return fmt.Errorf("telemetry delivery: missing spool directory")
	}
	if c.MaxSpoolBatches <= 0 || c.MaxSpoolBatches > 65536 {
		return fmt.Errorf("telemetry delivery: invalid spool batch limit")
	}
	if c.MaxSpoolBytes <= 0 || c.MaxSpoolBytes > 1<<30 {
		return fmt.Errorf("telemetry delivery: invalid spool byte limit")
	}
	if c.Delivery.MaxQueuedBatches == 0 {
		c.Delivery.MaxQueuedBatches = min(c.MaxSpoolBatches, 4096)
	}
	if c.Delivery.MaxQueuedBytes == 0 {
		c.Delivery.MaxQueuedBytes = int(min(c.MaxSpoolBytes, 64<<20))
	}
	if int64(c.Delivery.MaxBatchBytes) > c.MaxSpoolBytes {
		return fmt.Errorf("telemetry delivery: batch byte limit exceeds spool byte limit")
	}
	if err := validateDeliveryConfig(c.Delivery); err != nil {
		return err
	}
	return nil
}

func openSpoolRoot(path string) (*os.Root, error) {
	if err := os.Mkdir(path, 0700); err != nil && !os.IsExist(err) {
		return nil, fmt.Errorf("telemetry delivery: create spool directory: %w", err)
	}
	info, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("telemetry delivery: stat spool directory: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("telemetry delivery: spool directory cannot be a symbolic link")
	}
	if !info.IsDir() || info.Mode()&0077 != 0 {
		return nil, fmt.Errorf("telemetry delivery: spool directory must be private")
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, fmt.Errorf("telemetry delivery: open spool root: %w", err)
	}
	return root, nil
}

func readSpoolDirEntries(root *os.Root, maxEntries int) ([]os.DirEntry, error) {
	f, err := root.Open(".")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var entries []os.DirEntry
	for {
		batch, err := f.ReadDir(1024)
		if len(batch) > 0 {
			entries = append(entries, batch...)
			if maxEntries > 0 && len(entries) > maxEntries {
				return nil, fmt.Errorf("telemetry delivery: spool directory exceeds entry budget")
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	for _, e := range entries {
		if e.Type()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("telemetry delivery: spool directory contains symbolic link: %q", e.Name())
		}
		if e.Name() == "owner.json" {
			continue
		}
		if e.Name() == "lease" && e.IsDir() {
			continue
		}
		if isPendingTmpName(e.Name()) {
			continue
		}
		if !e.Type().IsRegular() || !isSpoolFileName(e.Name()) {
			return nil, fmt.Errorf("telemetry delivery: spool directory contains foreign file: %q", e.Name())
		}
	}
	return entries, nil
}

func readSpoolMarker(root *os.Root) ([]byte, error) {
	f, err := root.Open("owner.json")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, 1025))
	if len(b) > 1024 {
		return nil, fmt.Errorf("telemetry delivery: owner marker exceeds byte limit")
	}
	return b, err
}

func checkOrWriteOwnerMarker(root *os.Root, owner string) error {
	entries, err := readSpoolDirEntries(root, 65536)
	if err != nil {
		return err
	}
	markerBytes, err := readSpoolMarker(root)
	if os.IsNotExist(err) {
		if len(entries) != 0 {
			return fmt.Errorf("telemetry delivery: spool directory contains foreign files")
		}
		data, _ := json.Marshal(spoolOwnerMarker{Version: 1, SchemaVersion: DurableDeliveryRecoverySchemaVersion, Owner: owner})
		if err := writeSpoolFile(root, "owner.json", data); err != nil {
			return err
		}
		return syncSpoolDir(root)
	}
	if err != nil {
		return err
	}
	var m spoolOwnerMarker
	if err := decodeOperationsJSON(markerBytes, &m); err != nil || m.Version != 1 || (m.SchemaVersion != 0 && m.SchemaVersion != DurableDeliveryRecoverySchemaVersion) || m.Owner != owner {
		return fmt.Errorf("telemetry delivery: spool owner mismatch or corrupt marker")
	}
	return nil
}

func isSpoolFileName(s string) bool {
	if len(s) != 24 || !strings.HasPrefix(s, "spool-") || !strings.HasSuffix(s, ".json") {
		return false
	}
	for _, c := range s[6:19] {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func isPendingTmpName(s string) bool {
	if len(s) != 25 || !strings.HasPrefix(s, "pending-") || !strings.HasSuffix(s, ".tmp") {
		return false
	}
	for _, c := range s[8:21] {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func parseSpoolID(s string) (uint64, error) {
	if !isSpoolFileName(s) {
		return 0, fmt.Errorf("invalid spool file name: %s", s)
	}
	return strconv.ParseUint(s[6:19], 10, 64)
}

func formatSpoolName(id uint64) string {
	return fmt.Sprintf("spool-%013d.json", id)
}

func formatPendingTmpName(id uint64) string {
	return fmt.Sprintf("pending-%013d.tmp", id)
}

func writeSpoolFile(root *os.Root, name string, b []byte) error {
	f, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	n, err := f.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		_ = root.Remove(name)
		return err
	}
	return closeErr
}

func syncSpoolDir(root *os.Root) error {
	f, err := root.Open(".")
	if err != nil {
		return err
	}
	defer f.Close()
	return f.Sync()
}

func scanSpool(root *os.Root, cfg DurableDeliveryConfig, isLogs bool) ([]spooledBatch, int64, uint64, error) {
	maxEntries := cfg.MaxSpoolBatches*2 + 1024
	entries, err := readSpoolDirEntries(root, maxEntries)
	if err != nil {
		return nil, 0, 0, err
	}
	var removedPending bool
	for _, e := range entries {
		if isPendingTmpName(e.Name()) {
			_ = root.Remove(e.Name())
			removedPending = true
		}
	}
	if removedPending {
		_ = syncSpoolDir(root)
	}

	var batches []spooledBatch
	var totalBytes int64
	var maxID uint64

	for _, e := range entries {
		if !isSpoolFileName(e.Name()) {
			continue
		}
		if len(batches) >= cfg.MaxSpoolBatches {
			return nil, 0, 0, fmt.Errorf("telemetry delivery: spooled batches exceed configured capacity")
		}
		id, err := parseSpoolID(e.Name())
		if err != nil {
			return nil, 0, 0, fmt.Errorf("telemetry delivery: %w", err)
		}
		if id > maxID {
			maxID = id
		}
		f, err := root.Open(e.Name())
		if err != nil {
			return nil, 0, 0, fmt.Errorf("telemetry delivery: open spool batch: %w", err)
		}
		info, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, 0, 0, fmt.Errorf("telemetry delivery: stat spool batch: %w", err)
		}
		if info.Size() > int64(cfg.Delivery.MaxBatchBytes) {
			f.Close()
			return nil, 0, 0, fmt.Errorf("telemetry delivery: spool batch exceeds max batch bytes")
		}
		if totalBytes+info.Size() > cfg.MaxSpoolBytes {
			f.Close()
			return nil, 0, 0, fmt.Errorf("telemetry delivery: spooled batches exceed configured capacity")
		}
		payload, err := io.ReadAll(io.LimitReader(f, int64(cfg.Delivery.MaxBatchBytes)+1))
		f.Close()
		if err != nil {
			return nil, 0, 0, fmt.Errorf("telemetry delivery: read spool batch: %w", err)
		}
		if int64(len(payload)) > int64(cfg.Delivery.MaxBatchBytes) {
			return nil, 0, 0, fmt.Errorf("telemetry delivery: spool batch exceeds max batch bytes")
		}
		if isLogs {
			err = validateLogsDeliveryBatch(payload, cfg.Delivery.MaxBatchDataPoints)
		} else {
			err = validateDeliveryBatch(payload, cfg.Delivery.MaxBatchDataPoints)
		}
		if err != nil {
			return nil, 0, 0, fmt.Errorf("telemetry delivery: spool batch corrupt: %w", err)
		}
		totalBytes += int64(len(payload))
		batches = append(batches, spooledBatch{
			id:      id,
			name:    e.Name(),
			size:    int64(len(payload)),
			payload: payload,
		})
	}

	sort.Slice(batches, func(i, j int) bool {
		return batches[i].id < batches[j].id
	})

	if len(batches) > cfg.MaxSpoolBatches || totalBytes > cfg.MaxSpoolBytes {
		return nil, 0, 0, fmt.Errorf("telemetry delivery: spooled batches exceed configured capacity")
	}

	return batches, totalBytes, maxID, nil
}

// NewDurableDelivery starts a disk-backed OTLP HTTP metrics delivery worker.
// Batches enqueued to DurableDelivery are durably spooled to SpoolDir before
// admission is acknowledged. The spool directory must be exclusively owned by
// Owner and private to the calling user (0700). On startup, pending spooled
// records from any previous instance are cleanly reloaded and scheduled for
// delivery.
func NewDurableDelivery(c DurableDeliveryConfig) (*DurableDelivery, error) {
	return openDurableDelivery(c, false)
}

// NewLogsDurableDelivery starts a disk-backed OTLP HTTP logs delivery worker.
// It shares disk spooling, worker lifecycle, retry hysteresis, and recovery
// semantics with NewDurableDelivery while validating ExportLogsServiceRequest
// batches and reporting rejected log records.
func NewLogsDurableDelivery(c DurableDeliveryConfig) (*DurableDelivery, error) {
	return openDurableDelivery(c, true)
}

func openDurableDelivery(c DurableDeliveryConfig, isLogs bool) (*DurableDelivery, error) {
	if err := validateDurableDeliveryConfig(&c); err != nil {
		return nil, err
	}
	c.Delivery.Headers = c.Delivery.Headers.Clone()
	if c.Delivery.Headers == nil {
		c.Delivery.Headers = make(http.Header)
	}
	client := *c.Delivery.Client
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	c.Delivery.Client = &client

	root, err := openSpoolRoot(c.SpoolDir)
	if err != nil {
		return nil, err
	}
	if err := checkOrWriteOwnerMarker(root, c.Owner); err != nil {
		_ = root.Close()
		return nil, err
	}
	if err := root.Mkdir("lease", 0700); err != nil {
		_ = root.Close()
		return nil, fmt.Errorf("telemetry delivery writer lease: %w", err)
	}
	_ = syncSpoolDir(root)
	lockFile, err := root.OpenFile("lease/lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		_ = root.Remove("lease")
		_ = syncSpoolDir(root)
		_ = root.Close()
		return nil, fmt.Errorf("telemetry delivery writer lease: %w", err)
	}
	if err := lockFileExclusive(lockFile); err != nil {
		_ = lockFile.Close()
		_ = root.Close()
		return nil, fmt.Errorf("telemetry delivery writer lease: %w", err)
	}
	infoLock, err := lockFile.Stat()
	if err != nil {
		_ = unlockFile(lockFile)
		_ = lockFile.Close()
		_ = root.Close()
		return nil, fmt.Errorf("telemetry delivery writer lease: stat lock: %w", err)
	}
	infoPath, err := root.Stat("lease/lock")
	if err != nil || !os.SameFile(infoLock, infoPath) {
		_ = unlockFile(lockFile)
		_ = lockFile.Close()
		_ = root.Close()
		return nil, fmt.Errorf("telemetry delivery writer lease: unlinked lock")
	}
	batches, totalBytes, maxID, err := scanSpool(root, c, isLogs)
	if err != nil {
		_ = root.Remove("lease/lock")
		_ = root.Remove("lease")
		_ = syncSpoolDir(root)
		_ = unlockFile(lockFile)
		_ = lockFile.Close()
		_ = root.Close()
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())
	d := &DurableDelivery{
		config:          c,
		isLogs:          isLogs,
		root:            root,
		lockFile:        lockFile,
		spoolDir:        c.SpoolDir,
		queue:           batches,
		nextID:          maxID,
		wake:            make(chan struct{}, 1),
		done:            make(chan struct{}),
		flushWake:       make(chan struct{}, 1),
		shutdownExpired: make(chan struct{}),
		cancel:          cancel,
	}
	d.stats.PendingSpool = len(batches)
	d.stats.PendingSpoolBytes = totalBytes
	d.stats.PendingBatches = len(batches)
	d.stats.PendingBytes = int(totalBytes)
	d.stats.RecoveredBatches = uint64(len(batches))

	go d.run(ctx)
	return d, nil
}

// RecoverDurableDelivery cleans stale writer leases and temporary files after an
// ungraceful process termination. It validates the private spool directory and
// owner identity before taking action, and refuses recovery if the lease is actively
// held by a live writer.
func RecoverDurableDelivery(path, owner string) error {
	_, err := RecoverDurableDeliveryManifest(path, owner)
	return err
}

// RecoverDurableDeliveryManifest cleans stale writer leases and temporary files after an
// ungraceful process termination and returns a versioned recovery manifest describing the
// recovered spool state. It requires an existing owned marker and validates all
// committed OTLP requests and accounting bounds before removing anything.
// Payloads may be metrics or logs; opening a writer applies its smaller limits.
func RecoverDurableDeliveryManifest(path, owner string) (DurableDeliveryRecoveryManifest, error) {
	if len(owner) == 0 || len(owner) > 128 {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: invalid owner")
	}
	for _, b := range owner {
		if b < '!' || b > '~' {
			return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: invalid owner character")
		}
	}
	info, err := os.Lstat(path)
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: spool directory cannot be a symbolic link")
	}
	if !info.IsDir() || info.Mode()&0077 != 0 {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: spool directory must be private")
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, err
	}
	defer root.Close()
	if _, _, err := preflightDeliveryRecovery(root, owner); err != nil {
		return DurableDeliveryRecoveryManifest{}, err
	}

	_, leaseStatErr := root.Lstat("lease")
	hadLease := leaseStatErr == nil

	if err := root.Mkdir("lease", 0700); err != nil && !os.IsExist(err) {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: recovery lease mkdir: %w", err)
	}
	_ = syncSpoolDir(root)
	lockFile, err := root.OpenFile("lease/lock", os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: recovery lease open: %w", err)
	}
	defer lockFile.Close()
	if err := lockFileExclusive(lockFile); err != nil {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: writer lease is actively held")
	}
	infoLock, err := lockFile.Stat()
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: writer lease stat: %w", err)
	}
	infoPath, err := root.Stat("lease/lock")
	if err != nil || !os.SameFile(infoLock, infoPath) {
		return DurableDeliveryRecoveryManifest{}, fmt.Errorf("telemetry delivery: writer lease is actively held")
	}
	cleanupLease := !hadLease
	defer func() {
		infoLock, errLock := lockFile.Stat()
		infoPath, errPath := root.Stat("lease/lock")
		stillLinked := (errLock == nil && errPath == nil && os.SameFile(infoLock, infoPath))
		lockHeldByOther := false
		if err := lockFileExclusive(lockFile); err != nil {
			lockHeldByOther = true
		}
		if cleanupLease && stillLinked && !lockHeldByOther {
			_ = root.Remove("lease/lock")
			_ = root.Remove("lease")
			_ = syncSpoolDir(root)
		}
		_ = unlockFile(lockFile)
	}()

	// Recheck under the exclusive lease: a previous writer may have finished
	// between the read-only admission pass and acquisition.
	m, temporary, err := preflightDeliveryRecovery(root, owner)
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, err
	}
	for _, name := range temporary {
		if err := root.Remove(name); err != nil {
			return DurableDeliveryRecoveryManifest{}, err
		}
	}
	if len(temporary) > 0 {
		_ = syncSpoolDir(root)
	}
	cleanupLease = true
	m.RecoveredLease = hadLease

	return m, nil
}

// preflightDeliveryRecovery reads only. Recovery has no endpoint configuration,
// so accept either request kind supported by Enqueue, using its public hard
// limits. Configured smaller limits are checked again when a writer opens.
func preflightDeliveryRecovery(root *os.Root, owner string) (DurableDeliveryRecoveryManifest, []string, error) {
	m := DurableDeliveryRecoveryManifest{SchemaVersion: DurableDeliveryRecoverySchemaVersion, MarkerVersion: 1, Owner: owner}
	entries, err := readSpoolDirEntries(root, 65536)
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, nil, err
	}
	info, err := root.Lstat("owner.json")
	if err != nil || !info.Mode().IsRegular() {
		return DurableDeliveryRecoveryManifest{}, nil, fmt.Errorf("telemetry delivery: recovery requires an owned marker")
	}
	data, err := readSpoolMarker(root)
	if err != nil {
		return DurableDeliveryRecoveryManifest{}, nil, err
	}
	var marker spoolOwnerMarker
	if err := decodeOperationsJSON(data, &marker); err != nil || marker.Version != 1 || (marker.SchemaVersion != 0 && marker.SchemaVersion != DurableDeliveryRecoverySchemaVersion) || marker.Owner != owner {
		return DurableDeliveryRecoveryManifest{}, nil, fmt.Errorf("telemetry delivery: spool owner mismatch or corrupt marker")
	}
	var temporary []string
	for _, e := range entries {
		if e.Name() == "owner.json" || e.Name() == "lease" {
			continue
		}
		st, err := root.Lstat(e.Name())
		if err != nil {
			return DurableDeliveryRecoveryManifest{}, nil, err
		}
		if !st.Mode().IsRegular() {
			return DurableDeliveryRecoveryManifest{}, nil, fmt.Errorf("telemetry delivery: recovery entry is not a regular file")
		}
		if isPendingTmpName(e.Name()) {
			temporary = append(temporary, e.Name())
			continue
		}
		if _, err := parseSpoolID(e.Name()); err != nil {
			return DurableDeliveryRecoveryManifest{}, nil, err
		}
		if st.Size() <= 0 || st.Size() > 4<<20 || st.Size() > (1<<30)-m.PendingBytes {
			return DurableDeliveryRecoveryManifest{}, nil, fmt.Errorf("telemetry delivery: recovery spool exceeds byte bounds")
		}
		f, err := root.Open(e.Name())
		if err != nil {
			return DurableDeliveryRecoveryManifest{}, nil, err
		}
		opened, statErr := f.Stat()
		if statErr != nil || !os.SameFile(st, opened) {
			return DurableDeliveryRecoveryManifest{}, nil, errors.Join(fmt.Errorf("telemetry delivery: recovery spool identity changed"), statErr, f.Close())
		}
		payload, readErr := io.ReadAll(io.LimitReader(f, (4<<20)+1))
		after, statErr := f.Stat()
		closeErr := f.Close()
		if err := errors.Join(readErr, statErr, closeErr); err != nil {
			return DurableDeliveryRecoveryManifest{}, nil, err
		}
		if len(payload) != int(st.Size()) || after.Size() != st.Size() || after.ModTime() != st.ModTime() {
			return DurableDeliveryRecoveryManifest{}, nil, fmt.Errorf("telemetry delivery: recovery spool changed while reading")
		}
		if validateDeliveryBatch(payload, 4096) != nil && validateLogsDeliveryBatch(payload, 4096) != nil {
			return DurableDeliveryRecoveryManifest{}, nil, fmt.Errorf("telemetry delivery: recovery spool contains an invalid OTLP request")
		}
		m.PendingBatches++
		m.PendingBytes += int64(len(payload))
	}
	m.RemovedTempFiles = len(temporary) // intended count; reported only after removal
	if err := ValidateDurableDeliveryRecoveryManifest(m); err != nil {
		return DurableDeliveryRecoveryManifest{}, nil, err
	}
	return m, temporary, nil
}

// Enqueue validates and commits one encoded OTLP request to the disk spool.
// It reserves capacity under the writer lock before committing to disk,
// guaranteeing that concurrent admissions strictly observe spool limits.
// Caller cancellation refuses admission.
func (d *DurableDelivery) Enqueue(ctx context.Context, payload []byte) error {
	if ctx == nil {
		return fmt.Errorf("telemetry delivery: nil context")
	}
	if d == nil || d.done == nil {
		return ErrDeliveryClosed
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(payload) == 0 || len(payload) > d.config.Delivery.MaxBatchBytes {
		return fmt.Errorf("telemetry delivery: invalid batch size")
	}
	var err error
	if d.isLogs {
		err = validateLogsDeliveryBatch(payload, d.config.Delivery.MaxBatchDataPoints)
	} else {
		err = validateDeliveryBatch(payload, d.config.Delivery.MaxBatchDataPoints)
	}
	if err != nil {
		return err
	}

	d.mu.Lock()
	if d.closing || d.workerDone {
		d.mu.Unlock()
		return ErrDeliveryClosed
	}
	if d.nextID >= 9999999999999 {
		d.mu.Unlock()
		return fmt.Errorf("telemetry delivery: spool identifier exhausted")
	}
	if d.stats.PendingSpool >= d.config.MaxSpoolBatches || int64(len(payload)) > d.config.MaxSpoolBytes-d.stats.PendingSpoolBytes {
		d.stats.DroppedBatches++
		d.mu.Unlock()
		return ErrDeliveryFull
	}
	if err := ctx.Err(); err != nil {
		d.mu.Unlock()
		return err
	}

	d.nextID++
	id := d.nextID
	tmpName := formatPendingTmpName(id)
	finalName := formatSpoolName(id)

	// Reserve capacity under lock before releasing for disk I/O
	d.stats.PendingSpool++
	d.stats.PendingSpoolBytes += int64(len(payload))
	d.activeAdmissions++
	d.mu.Unlock()

	defer d.finishAdmission()

	if err := writeSpoolFile(d.root, tmpName, payload); err != nil {
		_ = d.root.Remove(tmpName)
		_ = syncSpoolDir(d.root)
		d.mu.Lock()
		d.stats.PendingSpool--
		d.stats.PendingSpoolBytes -= int64(len(payload))
		d.mu.Unlock()
		return fmt.Errorf("telemetry delivery: spool write: %w", err)
	}
	if err := d.root.Rename(tmpName, finalName); err != nil {
		_ = d.root.Remove(tmpName)
		_ = syncSpoolDir(d.root)
		d.mu.Lock()
		d.stats.PendingSpool--
		d.stats.PendingSpoolBytes -= int64(len(payload))
		d.mu.Unlock()
		return fmt.Errorf("telemetry delivery: spool commit: %w", err)
	}
	if err := syncSpoolDir(d.root); err != nil {
		rmErr := d.root.Remove(finalName)
		d.mu.Lock()
		if rmErr != nil {
			// Rollback deletion failed; the file remains on disk.
			// Retain the disk reservation so budgets are not exceeded.
			d.terminal = fmt.Errorf("telemetry delivery: sync spool dir: %w (rollback remove: %v)", err, rmErr)
			d.mu.Unlock()
			return fmt.Errorf("telemetry delivery: sync spool dir: %w", err)
		}
		_ = syncSpoolDir(d.root)
		d.stats.PendingSpool--
		d.stats.PendingSpoolBytes -= int64(len(payload))
		d.mu.Unlock()
		return fmt.Errorf("telemetry delivery: sync spool dir: %w", err)
	}

	d.mu.Lock()
	if d.closing {
		rmErr := d.root.Remove(finalName)
		if rmErr != nil {
			d.terminal = fmt.Errorf("telemetry delivery: closed during admission (rollback remove: %w)", rmErr)
			d.mu.Unlock()
			return ErrDeliveryClosed
		}
		_ = syncSpoolDir(d.root)
		d.stats.PendingSpool--
		d.stats.PendingSpoolBytes -= int64(len(payload))
		d.mu.Unlock()
		return ErrDeliveryClosed
	}
	d.queue = append(d.queue, spooledBatch{
		id:      id,
		name:    finalName,
		size:    int64(len(payload)),
		payload: append([]byte(nil), payload...),
	})
	d.stats.PendingBatches++
	d.stats.PendingBytes += len(payload)
	d.stats.EnqueuedBatches++
	d.stats.SpooledBatches++
	d.stats.SpooledBytes += uint64(len(payload))
	d.mu.Unlock()

	d.signal()
	return nil
}

func (d *DurableDelivery) finishAdmission() {
	d.mu.Lock()
	d.activeAdmissions--
	shouldRelease := d.tryReleaseLeaseLocked()
	d.mu.Unlock()
	d.signal()
	if shouldRelease {
		d.releaseLease()
	}
}

// Stats returns a coherent copy of delivery and spool counters without payload metadata.
func (d *DurableDelivery) Stats() DurableDeliveryStats {
	if d == nil {
		return DurableDeliveryStats{}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.stats
}

// Recover flushes pending spooled records to the endpoint until the spool queue is
// drained or ctx is done. It returns ErrDeliveryIncomplete if any batch failed delivery.
func (d *DurableDelivery) Recover(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("telemetry delivery: nil context")
	}
	if d == nil || d.done == nil {
		return ErrDeliveryClosed
	}
	d.signal()
	for {
		d.mu.Lock()
		closing := d.closing
		pending := d.stats.PendingSpool
		term := d.terminal
		d.mu.Unlock()
		if closing {
			if term != nil {
				return term
			}
			return ErrDeliveryClosed
		}
		if term != nil {
			return term
		}
		if pending == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-d.done:
			d.mu.Lock()
			term = d.terminal
			d.mu.Unlock()
			if term != nil {
				return term
			}
			return nil
		case <-d.flushWake:
		}
	}
}

// Close stops admission and drains spooled batches within ShutdownTimeout or
// caller's context deadline. If shutdown expires or is cancelled before draining,
// pending batches remain spooled on disk for restart recovery, and ErrDeliveryIncomplete
// is returned. Close always releases the directory writer lease.
func (d *DurableDelivery) Close(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("telemetry delivery: nil context")
	}
	if d == nil || d.done == nil {
		return ErrDeliveryClosed
	}
	d.closeOnce.Do(func() {
		d.mu.Lock()
		d.closing = true
		d.mu.Unlock()
		d.signal()
		go func() {
			timer := time.NewTimer(d.config.Delivery.ShutdownTimeout)
			defer timer.Stop()
			select {
			case <-d.done:
				return
			case <-ctx.Done():
			case <-timer.C:
			}
			d.mu.Lock()
			d.terminal = ErrDeliveryIncomplete
			d.mu.Unlock()
			d.cancel()
			close(d.shutdownExpired)
		}()
	})
	select {
	case <-d.done:
		d.mu.Lock()
		defer d.mu.Unlock()
		if (d.stats.PendingSpool > 0 || d.activeAdmissions > 0) && d.terminal == nil {
			return ErrDeliveryIncomplete
		}
		return d.terminal
	case <-ctx.Done():
		d.cancel()
		return errors.Join(ErrDeliveryIncomplete, ctx.Err())
	case <-d.shutdownExpired:
		return ErrDeliveryIncomplete
	}
}

func (d *DurableDelivery) tryReleaseLeaseLocked() bool {
	if !d.leaseReleased && d.workerDone && d.activeAdmissions == 0 {
		d.leaseReleased = true
		return true
	}
	return false
}

func (d *DurableDelivery) releaseLease() {
	d.mu.Lock()
	d.leaseReleased = true
	d.mu.Unlock()
	d.leaseOnce.Do(func() {
		if d.lockFile != nil {
			infoLock, errLock := d.lockFile.Stat()
			infoPath, errPath := d.root.Stat("lease/lock")
			stillLinked := (errLock == nil && errPath == nil && os.SameFile(infoLock, infoPath))
			lockHeldByOther := false
			if err := lockFileExclusive(d.lockFile); err != nil {
				lockHeldByOther = true
			}
			if stillLinked && !lockHeldByOther {
				_ = d.root.Remove("lease/lock")
				_ = d.root.Remove("lease")
				_ = syncSpoolDir(d.root)
			}
			_ = unlockFile(d.lockFile)
			_ = d.lockFile.Close()
			d.lockFile = nil
		}
		if d.root != nil {
			_ = d.root.Close()
			d.root = nil
		}
	})
}

func (d *DurableDelivery) signal() {
	select {
	case d.wake <- struct{}{}:
	default:
	}
}

func (d *DurableDelivery) notifyFlush() {
	select {
	case d.flushWake <- struct{}{}:
	default:
	}
}

func (d *DurableDelivery) run(ctx context.Context) {
	defer close(d.done)
	defer func() {
		d.mu.Lock()
		d.workerDone = true
		shouldRelease := d.tryReleaseLeaseLocked()
		d.mu.Unlock()
		if shouldRelease {
			d.releaseLease()
		}
	}()
	defer d.cancel()
	for {
		d.mu.Lock()
		if ctx.Err() != nil {
			if d.stats.PendingSpool > 0 || d.activeAdmissions > 0 {
				d.terminal = ErrDeliveryIncomplete
			}
			d.mu.Unlock()
			d.notifyFlush()
			return
		}
		if len(d.queue) == 0 {
			closing := d.closing
			active := d.activeAdmissions
			d.mu.Unlock()
			if closing && active == 0 {
				return
			}
			select {
			case <-ctx.Done():
			case <-d.wake:
			}
			continue
		}
		item := d.queue[0]
		d.mu.Unlock()

		ok := d.send(ctx, item.payload)

		d.mu.Lock()
		if ok {
			err := d.root.Remove(item.name)
			if err != nil {
				d.terminal = fmt.Errorf("telemetry delivery: remove spool batch: %w", err)
				d.mu.Unlock()
				d.notifyFlush()
				return
			}
			err = syncSpoolDir(d.root)
			if err != nil {
				d.terminal = fmt.Errorf("telemetry delivery: sync spool dir: %w", err)
				d.mu.Unlock()
				d.notifyFlush()
				return
			}
			d.queue[0] = spooledBatch{}
			d.queue = d.queue[1:]
			d.stats.SucceededBatches++
			d.stats.PendingSpool--
			d.stats.PendingSpoolBytes -= item.size
			d.stats.PendingBatches--
			d.stats.PendingBytes -= int(item.size)
		} else {
			if ctx.Err() != nil {
				d.terminal = ErrDeliveryIncomplete
				d.mu.Unlock()
				d.notifyFlush()
				return
			}
			err := d.root.Remove(item.name)
			if err != nil {
				d.terminal = fmt.Errorf("telemetry delivery: remove spool batch: %w", err)
				d.mu.Unlock()
				d.notifyFlush()
				return
			}
			err = syncSpoolDir(d.root)
			if err != nil {
				d.terminal = fmt.Errorf("telemetry delivery: sync spool dir: %w", err)
				d.mu.Unlock()
				d.notifyFlush()
				return
			}
			d.queue[0] = spooledBatch{}
			d.queue = d.queue[1:]
			d.stats.FailedBatches++
			d.stats.PendingSpool--
			d.stats.PendingSpoolBytes -= item.size
			d.stats.PendingBatches--
			d.stats.PendingBytes -= int(item.size)
			d.terminal = ErrDeliveryIncomplete
		}
		d.mu.Unlock()
		d.notifyFlush()
	}
}

func (d *DurableDelivery) send(parent context.Context, payload []byte) bool {
	ctx, cancel := context.WithTimeout(parent, d.config.Delivery.BatchTimeout)
	defer cancel()
	backoff := d.config.Delivery.InitialBackoff
	for attempt := 0; attempt < d.config.Delivery.MaxAttempts; attempt++ {
		if ctx.Err() != nil {
			return false
		}
		if attempt > 0 {
			d.mu.Lock()
			d.stats.RetryAttempts++
			d.mu.Unlock()
		}
		requestCtx, requestCancel := context.WithTimeout(ctx, d.config.Delivery.RequestTimeout)
		req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, d.config.Delivery.Endpoint, bytes.NewReader(payload))
		if err != nil {
			requestCancel()
			return false
		}
		req.Header = d.config.Delivery.Headers.Clone()
		req.Header.Set("Content-Type", "application/json")
		response, err := d.config.Delivery.Client.Do(req)
		retry := false
		retryAfter := time.Duration(0)
		if err != nil {
			requestCancel()
			d.mu.Lock()
			d.stats.TransportFailures++
			d.mu.Unlock()
			retry = true
		} else {
			body, readErr := io.ReadAll(io.LimitReader(response.Body, d.config.Delivery.MaxResponseBytes+1))
			closeErr := response.Body.Close()
			requestCancel()
			status := response.StatusCode
			retryAfter = deliveryRetryAfter(response.Header.Get("Retry-After"), time.Now())
			if status == http.StatusOK {
				d.mu.Lock()
				d.stats.SentBatches++
				d.mu.Unlock()
				if readErr != nil || closeErr != nil || int64(len(body)) > d.config.Delivery.MaxResponseBytes {
					d.mu.Lock()
					d.stats.InvalidResponses++
					d.mu.Unlock()
					return false
				}
				partial, rejected, valid := deliveryResponse(body, d.isLogs)
				if !valid {
					d.mu.Lock()
					d.stats.InvalidResponses++
					d.mu.Unlock()
					return false
				}
				if partial {
					d.mu.Lock()
					d.stats.PartialBatches++
					if d.isLogs {
						d.stats.RejectedLogRecords += rejected
					} else {
						d.stats.RejectedDataPoints += rejected
					}
					d.mu.Unlock()
					return false
				}
				return true
			}
			d.mu.Lock()
			if status == http.StatusUnauthorized || status == http.StatusForbidden {
				d.stats.AuthFailures++
			} else if status != 429 && status != 502 && status != 503 && status != 504 {
				d.stats.RejectedBatches++
			} else {
				d.stats.OutageResponses++
			}
			d.mu.Unlock()
			retry = status == 429 || status == 502 || status == 503 || status == 504
		}
		if !retry || attempt+1 >= d.config.Delivery.MaxAttempts {
			return false
		}
		wait := backoff
		if backoff > 1 {
			wait = backoff/2 + time.Duration(rand.Int64N(int64(backoff-backoff/2)))
		}
		if retryAfter > wait {
			wait = retryAfter
		}
		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return false
		case <-timer.C:
		}
		if backoff < d.config.Delivery.MaxBackoff {
			if backoff > d.config.Delivery.MaxBackoff/2 {
				backoff = d.config.Delivery.MaxBackoff
			} else {
				backoff *= 2
			}
		}
	}
	return false
}
