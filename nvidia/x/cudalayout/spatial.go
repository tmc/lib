package cudalayout

import (
	"fmt"
	"math"
)

// Validate checks that each CTA mesh factor divides the number of shared
// tiles along its problem dimension, and each warp mesh factor divides the
// number of register tiles within a shared tile. Omitted dimensions have
// factor one. Products must fit int. This checks logical subdivisions, not
// launch geometry, reduction synchronization or device residency.
func (s SpatialTiling) Validate(factors IndexFactorization, problem map[string]int) error {
	if err := factors.Validate(problem); err != nil {
		return err
	}
	shared := factors.TileShape(LevelShared)
	register := factors.TileShape(LevelRegister)
	for _, mesh := range []struct {
		name  string
		parts map[string]int
		outer map[string]int
		inner map[string]int
	}{
		{"CTA", s.CTAMesh, problem, shared},
		{"warp", s.WarpMesh, shared, register},
	} {
		if _, err := spatialProduct(mesh.parts); err != nil {
			return fmt.Errorf("%s mesh: %w", mesh.name, err)
		}
		for dim, count := range mesh.parts {
			outer, inner := mesh.outer[dim], mesh.inner[dim]
			if outer <= 0 || inner <= 0 || outer%inner != 0 || (outer/inner)%count != 0 {
				return fmt.Errorf("%s mesh factor %d for %s does not divide its tile count: %w", mesh.name, count, dim, ErrUnsupported)
			}
		}
	}
	return nil
}

func spatialProduct(mesh map[string]int) (int, error) {
	product := 1
	for dim, count := range mesh {
		if count <= 0 {
			return 0, fmt.Errorf("mesh factor for %s must be positive: %w", dim, ErrUnsupported)
		}
		if product > math.MaxInt/count {
			return 0, ErrOverflow
		}
		product *= count
	}
	return product, nil
}
