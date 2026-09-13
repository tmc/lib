
struct ds4_q8_activation { float d; signed char qs[32]; };
static_assert(sizeof(ds4_q8_activation) == 36, "Q8 activation block size");

