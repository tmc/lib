// Bounded stable expert-major grouping. Invalid ids are counted separately,
// never used as expert indices, matching upstream sorted-pair exclusion.
// The consumer must reject invalid_count != 0 before evaluating experts.
extern "C" __global__ void ds4_shape_group_csr(
    const int *ids, int *offsets, int *pairs, int *active, int *count,
    int *invalid_count, unsigned npairs, unsigned experts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int invalid = 0;
  for (unsigned p = 0; p < npairs; p++)
    if (ids[p] < 0 || unsigned(ids[p]) >= experts) invalid++;
  *invalid_count = invalid;
  int total = 0, nactive = 0;
  for (unsigned e = 0; e < experts; e++) {
    offsets[e] = total;
    const int before = total;
    for (unsigned p = 0; p < npairs; p++)
      if (ids[p] == int(e)) pairs[total++] = int(p);
    if (total != before) active[nactive++] = int(e);
  }
  offsets[experts] = total;
  *count = nactive;
}
