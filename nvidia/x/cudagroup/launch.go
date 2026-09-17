package cudagroup

import "errors"

// LaunchRequirements describes grid and cluster dimensions in CTAs and block
// dimensions in threads. A zero Cluster means an ordinary, unclustered launch.
// Device-specific limits and cluster launch attributes need driver validation.
type LaunchRequirements struct{ Grid, Block, Cluster Dim3 }

// ValidateLaunch checks membership against a launch description. It does not
// establish device capability, residency, synchronization, or forward progress.
func ValidateLaunch(req LaunchRequirements, group Group) error {
	if group.kind == 0 {
		return errors.New("launch requires an execution group")
	}
	if _, err := req.Grid.count(); err != nil {
		return err
	}
	block, err := New(CTA, req.Block)
	if err != nil {
		return err
	}
	if req.Cluster != (Dim3{}) {
		if _, err := New(Cluster, req.Cluster); err != nil {
			return err
		}
		if req.Grid.X%req.Cluster.X != 0 || req.Grid.Y%req.Cluster.Y != 0 || req.Grid.Z%req.Cluster.Z != 0 {
			return errors.New("grid dimensions are not divisible by cluster dimensions")
		}
	}
	switch group.kind {
	case Thread, Warp, WarpGroup:
		if group.base+group.Size() > block.Size() {
			return errors.New("group extends beyond launch block")
		}
		if group.parent != nil && group.parent.dims != req.Block {
			return errors.New("subgroup parent differs from launch block")
		}
	case CTA:
		if group.dims != req.Block {
			return errors.New("cta shape differs from launch block")
		}
	case Cluster:
		if group.dims != req.Cluster {
			return errors.New("cluster shape differs from launch cluster")
		}
	default:
		return errors.New("unsupported execution group")
	}
	return nil
}
