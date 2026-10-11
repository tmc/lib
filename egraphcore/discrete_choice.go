package egraphcore

import "fmt"

// IntnSource supplies the bounded random integers used by discrete choice
// operators. math/rand.Rand implements this interface.
type IntnSource interface {
	Intn(n int) int
}

// DiscreteChoiceGene identifies one mutable position in an integer genome and
// the alleles permitted at that position.
type DiscreteChoiceGene struct {
	Index   int
	Alleles []int
}

// MutateDiscreteChoice applies mutationCount independent mutations to genome.
// Each mutation chooses one supplied gene uniformly, then chooses one of its
// allowed alleles uniformly. Genes and allele pools should be supplied in
// deterministic order when reproducible searches are required. The returned
// genome is a copy; the input is unchanged.
func MutateDiscreteChoice(genome []int, genes []DiscreteChoiceGene, mutationCount int, rng IntnSource) ([]int, error) {
	if mutationCount < 0 {
		return nil, fmt.Errorf("egraphcore: negative mutation count")
	}
	if mutationCount > 0 && len(genes) != 0 && rng == nil {
		return nil, fmt.Errorf("egraphcore: nil discrete choice random source")
	}
	seen := make(map[int]bool, len(genes))
	for _, gene := range genes {
		if gene.Index < 0 || gene.Index >= len(genome) {
			return nil, fmt.Errorf("egraphcore: discrete choice gene index %d out of range", gene.Index)
		}
		if seen[gene.Index] {
			return nil, fmt.Errorf("egraphcore: duplicate discrete choice gene index %d", gene.Index)
		}
		seen[gene.Index] = true
		if len(gene.Alleles) == 0 {
			return nil, fmt.Errorf("egraphcore: discrete choice gene %d has no alleles", gene.Index)
		}
	}
	result := append([]int(nil), genome...)
	for n := 0; n < mutationCount && len(genes) != 0; n++ {
		geneIndex := rng.Intn(len(genes))
		if geneIndex < 0 || geneIndex >= len(genes) {
			return nil, fmt.Errorf("egraphcore: discrete choice random gene index %d out of range", geneIndex)
		}
		gene := genes[geneIndex]
		alleleIndex := rng.Intn(len(gene.Alleles))
		if alleleIndex < 0 || alleleIndex >= len(gene.Alleles) {
			return nil, fmt.Errorf("egraphcore: discrete choice random allele index %d out of range", alleleIndex)
		}
		result[gene.Index] = gene.Alleles[alleleIndex]
	}
	return result, nil
}
