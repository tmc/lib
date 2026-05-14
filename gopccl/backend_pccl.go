//go:build gopccl_pccl

package gopccl

import (
	"context"
	"errors"
	"fmt"
	"net"
	"runtime"
	"strconv"
	"strings"
	"unsafe"

	pccl "github.com/tmc/lib/pccl"
)

func init() {
	registerBackend("pccl", newPCCLBackend)
}

type pcclBackend struct {
	comm          pccl.Comm
	master        pccl.MasterInstance
	masterStarted bool
	closed        bool
}

func newPCCLBackend(ctx context.Context, cfg Config) (backend, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := pcclValidateConfig(cfg); err != nil {
		return nil, err
	}
	addr, err := pcclParseAddress(cfg.Coordinator)
	if err != nil {
		return nil, err
	}
	if err := pccl.Init(); err != nil {
		return nil, fmt.Errorf("pccl init: %w", err)
	}
	b := &pcclBackend{}
	if cfg.StartMaster {
		if err := pccl.CreateMaster(addr, uintptr(unsafe.Pointer(&b.master))); err != nil {
			return nil, fmt.Errorf("pccl create master: %w", err)
		}
		if err := pccl.RunMaster(&b.master); err != nil {
			_ = pccl.DestroyMaster(&b.master)
			return nil, fmt.Errorf("pccl run master: %w", err)
		}
		b.masterStarted = true
	}
	params := pccl.CommCreateParams{
		MasterAddress:         addr,
		PeerGroup:             cfg.PeerGroup,
		P2pConnectionPoolSize: cfg.P2PPoolSize,
	}
	if err := pccl.CreateCommunicator(&params, uintptr(unsafe.Pointer(&b.comm))); err != nil {
		_ = b.Close()
		return nil, fmt.Errorf("pccl create communicator: %w", err)
	}
	if err := b.comm.Connect(); err != nil {
		_ = b.Close()
		return nil, fmt.Errorf("pccl connect: %w", err)
	}
	if err := ctx.Err(); err != nil {
		_ = b.Close()
		return nil, err
	}
	return b, nil
}

func (b *pcclBackend) Name() string { return "pccl" }

func (b *pcclBackend) Close() error {
	if b == nil || b.closed {
		return nil
	}
	b.closed = true
	var err error
	if b.comm != 0 {
		if e := b.comm.DestroyCommunicator(); e != nil {
			err = errors.Join(err, fmt.Errorf("pccl destroy communicator: %w", e))
		}
		b.comm = 0
	}
	if b.masterStarted && b.master != 0 {
		if e := pccl.InterruptMaster(&b.master); e != nil {
			err = errors.Join(err, fmt.Errorf("pccl interrupt master: %w", e))
		} else if e := pccl.MasterAwaitTermination(&b.master); e != nil {
			err = errors.Join(err, fmt.Errorf("pccl await master: %w", e))
		} else if e := pccl.DestroyMaster(&b.master); e != nil {
			err = errors.Join(err, fmt.Errorf("pccl destroy master: %w", e))
		}
		b.master = 0
	}
	return err
}

func (b *pcclBackend) Capabilities() Capabilities {
	ready := b != nil && !b.closed
	return Capabilities{
		Collectives: []string{"all_reduce"},
		DeviceKinds: []DeviceKind{DeviceHost, DeviceCUDA},
		Provider:    ProviderInfo{Name: "pccl", Runtime: "pccl"},
		Health:      HealthSnapshot{Ready: ready},
	}
}

func (b *pcclBackend) barrier(ctx context.Context) error { return ErrUnsupported }

func (b *pcclBackend) allReduce(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	if err := pcclContext(ctx); err != nil {
		return err
	}
	srcPtr, dstPtr, dtype, err := pcclCollectiveArgs(dst, src)
	if err != nil {
		return err
	}
	redop, err := pcclReduceOp(op)
	if err != nil {
		return err
	}
	desc := pccl.ReduceDescriptor{
		Count:         src.Count,
		Op:            redop,
		SrcDescriptor: pccl.ReduceOperandDescriptor{Datatype: dtype},
		QuantizationOptions: pccl.QuantizationOptions{
			QuantizedDatatype: dtype,
			Algorithm:         pccl.QuantNone,
		},
	}
	info, err := b.comm.AllReduce(srcPtr, dstPtr, &desc)
	runtime.KeepAlive(src.Owner)
	runtime.KeepAlive(dst.Owner)
	_ = info
	if err != nil {
		return fmt.Errorf("pccl all reduce: %w", err)
	}
	return pcclContext(ctx)
}

func (b *pcclBackend) allGather(ctx context.Context, dst Buffer, src Buffer) error {
	return ErrUnsupported
}
func (b *pcclBackend) broadcast(ctx context.Context, dst Buffer, src Buffer, root int) error {
	return ErrUnsupported
}
func (b *pcclBackend) allToAll(ctx context.Context, dst Buffer, src Buffer) error {
	return ErrUnsupported
}
func (b *pcclBackend) reduceScatter(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	return ErrUnsupported
}
func (b *pcclBackend) send(ctx context.Context, peer int, src Buffer) error { return ErrUnsupported }
func (b *pcclBackend) recv(ctx context.Context, peer int, dst Buffer) error { return ErrUnsupported }

func (b *pcclBackend) peersPending(ctx context.Context) (bool, error) {
	if err := pcclContext(ctx); err != nil {
		return false, err
	}
	pending, err := b.comm.ArePeersPending()
	if err != nil {
		return false, fmt.Errorf("pccl peers pending: %w", err)
	}
	if err := pcclContext(ctx); err != nil {
		return false, err
	}
	return pending, nil
}

func (b *pcclBackend) updateTopology(ctx context.Context) error {
	if err := pcclContext(ctx); err != nil {
		return err
	}
	if err := b.comm.UpdateTopology(); err != nil {
		return fmt.Errorf("pccl update topology: %w", err)
	}
	return pcclContext(ctx)
}

func (b *pcclBackend) optimizeTopology(ctx context.Context) error {
	if err := pcclContext(ctx); err != nil {
		return err
	}
	if err := b.comm.OptimizeTopology(); err != nil {
		return fmt.Errorf("pccl optimize topology: %w", err)
	}
	return pcclContext(ctx)
}

func (b *pcclBackend) synchronizeState(ctx context.Context, revision uint64, tensors []StateTensor, strategy SyncStrategy) (SyncInfo, error) {
	if err := pcclContext(ctx); err != nil {
		return SyncInfo{}, err
	}
	pcclStrategy, err := pcclSyncStrategy(strategy)
	if err != nil {
		return SyncInfo{}, err
	}
	state, pins, err := pcclSharedState(revision, tensors)
	if err != nil {
		return SyncInfo{}, err
	}
	info, err := b.comm.SynchronizeSharedState(&state, pcclStrategy)
	runtime.KeepAlive(pins)
	if err != nil {
		return SyncInfo{}, fmt.Errorf("pccl synchronize state: %w", err)
	}
	if err := pcclContext(ctx); err != nil {
		return SyncInfo{}, err
	}
	return SyncInfo{TxBytes: info.TxBytes, RxBytes: info.RxBytes}, nil
}

func pcclValidateConfig(cfg Config) error {
	if cfg.Size <= 0 {
		return fmt.Errorf("%w: size must be positive", ErrInvalidArgument)
	}
	if cfg.Rank < 0 || cfg.Rank >= cfg.Size {
		return fmt.Errorf("%w: rank %d out of range for size %d", ErrInvalidArgument, cfg.Rank, cfg.Size)
	}
	if cfg.Coordinator == "" {
		return fmt.Errorf("%w: coordinator is required", ErrInvalidArgument)
	}
	if _, err := pcclParseAddress(cfg.Coordinator); err != nil {
		return err
	}
	return nil
}

func pcclParseAddress(value string) (pccl.SocketAddressT, error) {
	host, portText, err := net.SplitHostPort(value)
	if err != nil {
		return pccl.SocketAddressT{}, fmt.Errorf("%w: coordinator address %q", ErrInvalidArgument, value)
	}
	port, err := strconv.Atoi(portText)
	if err != nil || port < 0 || port > 65535 {
		return pccl.SocketAddressT{}, fmt.Errorf("%w: coordinator port %q", ErrInvalidArgument, portText)
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return pccl.SocketAddressT{}, fmt.Errorf("%w: coordinator host %q", ErrInvalidArgument, host)
	}
	addr := pccl.SocketAddressT{Port: uint16(port)}
	if v4 := ip.To4(); v4 != nil {
		addr.Inet.Protocol = pccl.IPv4
		copy(addr.Inet.Ipv4.Data[:], v4)
		return addr, nil
	}
	v6 := ip.To16()
	if v6 == nil {
		return pccl.SocketAddressT{}, fmt.Errorf("%w: coordinator host %q", ErrInvalidArgument, host)
	}
	addr.Inet.Protocol = pccl.IPv6
	copy(addr.Inet.Ipv6.Data[:], v6)
	return addr, nil
}

type pcclStatePins struct {
	infos  []pccl.TensorInfo
	names  [][]byte
	owners []any
}

func pcclSharedState(revision uint64, tensors []StateTensor) (pccl.SharedState, pcclStatePins, error) {
	if len(tensors) == 0 {
		return pccl.SharedState{}, pcclStatePins{}, fmt.Errorf("%w: tensors are required", ErrInvalidArgument)
	}
	pins := pcclStatePins{
		infos:  make([]pccl.TensorInfo, len(tensors)),
		names:  make([][]byte, len(tensors)),
		owners: make([]any, len(tensors)),
	}
	for i, tensor := range tensors {
		if tensor.Name == "" {
			return pccl.SharedState{}, pcclStatePins{}, fmt.Errorf("%w: tensor name is required", ErrInvalidArgument)
		}
		if strings.ContainsRune(tensor.Name, 0) {
			return pccl.SharedState{}, pcclStatePins{}, fmt.Errorf("%w: tensor name contains nul", ErrInvalidArgument)
		}
		ptr, err := pcclBufferPtr(tensor.Buffer)
		if err != nil {
			return pccl.SharedState{}, pcclStatePins{}, err
		}
		dtype, err := pcclDataType(tensor.Buffer.DType)
		if err != nil {
			return pccl.SharedState{}, pcclStatePins{}, err
		}
		device, err := pcclDeviceType(tensor.Buffer.Device.Kind)
		if err != nil {
			return pccl.SharedState{}, pcclStatePins{}, err
		}
		name := append([]byte(tensor.Name), 0)
		pins.names[i] = name
		pins.owners[i] = tensor.Buffer.Owner
		pins.infos[i] = pccl.TensorInfo{
			Name:                   uintptr(unsafe.Pointer(&name[0])),
			Data:                   ptr,
			Count:                  tensor.Buffer.Count,
			Datatype:               dtype,
			DeviceType:             device,
			AllowContentInequality: tensor.AllowInequality,
		}
	}
	state := pccl.SharedState{
		Revision: revision,
		Count:    uint(len(pins.infos)),
		Infos:    uintptr(unsafe.Pointer(&pins.infos[0])),
	}
	return state, pins, nil
}

func pcclDeviceType(kind DeviceKind) (pccl.DeviceType, error) {
	switch kind {
	case DeviceHost:
		return pccl.DeviceCpu, nil
	case DeviceCUDA:
		return pccl.DeviceCuda, nil
	default:
		return 0, fmt.Errorf("%w: buffer device %d, want host or cuda", ErrInvalidArgument, kind)
	}
}

func pcclCollectiveArgs(dst Buffer, src Buffer) (uintptr, uintptr, pccl.DataType, error) {
	if src.Count == 0 {
		return 0, 0, 0, fmt.Errorf("%w: count must be positive", ErrInvalidArgument)
	}
	srcPtr, err := pcclBufferPtr(src)
	if err != nil {
		return 0, 0, 0, err
	}
	dstPtr, err := pcclBufferPtr(dst)
	if err != nil {
		return 0, 0, 0, err
	}
	dtype, err := pcclDataType(src.DType)
	if err != nil {
		return 0, 0, 0, err
	}
	return srcPtr, dstPtr, dtype, nil
}

func pcclBufferPtr(buf Buffer) (uintptr, error) {
	if buf.Count == 0 {
		return 0, fmt.Errorf("%w: count must be positive", ErrInvalidArgument)
	}
	if buf.Device.Kind != DeviceHost && buf.Device.Kind != DeviceCUDA {
		return 0, fmt.Errorf("%w: buffer device %d, want host or cuda", ErrInvalidArgument, buf.Device.Kind)
	}
	if buf.Ptr == nil {
		return 0, fmt.Errorf("%w: nil buffer pointer", ErrInvalidArgument)
	}
	return uintptr(buf.Ptr), nil
}

func pcclDataType(dtype DType) (pccl.DataType, error) {
	switch dtype {
	case Int8:
		return pccl.Int8, nil
	case Int16:
		return pccl.Int16, nil
	case Int32:
		return pccl.Int32, nil
	case Int64:
		return pccl.Int64, nil
	case Uint8:
		return pccl.Uint8, nil
	case Uint16:
		return pccl.Uint16, nil
	case Uint32:
		return pccl.Uint32, nil
	case Uint64:
		return pccl.Uint64, nil
	case Float32:
		return pccl.Float, nil
	case Float64:
		return pccl.Double, nil
	default:
		return 0, fmt.Errorf("%w: unsupported dtype %d", ErrInvalidArgument, dtype)
	}
}

func pcclSyncStrategy(strategy SyncStrategy) (pccl.SharedStateSyncStrategy, error) {
	switch strategy {
	case EnforcePopular:
		return pccl.SharedStateSyncStrategyEnforcePopular, nil
	case ReceiveOnly:
		return pccl.SharedStateSyncStrategyReceiveOnly, nil
	case SendOnly:
		return pccl.SharedStateSyncStrategySendOnly, nil
	default:
		return 0, fmt.Errorf("%w: unsupported sync strategy %d", ErrInvalidArgument, strategy)
	}
}

func pcclReduceOp(op ReduceOp) (pccl.RedOp, error) {
	switch op {
	case Sum:
		return pccl.Sum, nil
	case Max:
		return pccl.Max, nil
	case Min:
		return pccl.Min, nil
	case Prod:
		return pccl.Prod, nil
	default:
		return 0, fmt.Errorf("%w: unsupported reduce op %d", ErrInvalidArgument, op)
	}
}

func pcclContext(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
