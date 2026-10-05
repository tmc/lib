// vec.cu exercises cudabind: padding, unnamed structs, toolkit types,
// enums, macros, C++ linkage, by-value structs and dynamic shared memory.

#define VEC_WARP 32
#define VEC_LANES (VEC_WARP / 4)
#define VEC_HALF 0.5f
#define VEC_MIN(a, b) ((a) < (b) ? (a) : (b))

// Pair needs padding in Go before b, which C aligns to 16 bytes.
struct Pair {
    int a;
    uint4 b;
};

// Point is a typedef of an unnamed struct.
typedef struct {
    float x;
    double y;
} Point;

enum Op { OP_ADD, OP_MUL = 4 };

namespace geom {
enum class Side { LEFT = -1, RIGHT = 1 };
}

// scale is the factor of scale_all.
__constant__ float scale;

// origin is the point that shift moves points from.
__device__ Point origin;

// scale_all multiplies each of the n values of x by scale.
extern "C" __global__ void scale_all(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= scale;
    }
}

// shift moves each of the n points by origin and records op.
__global__ void shift(Point* p, Pair* tags, int n, Op op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += origin.x;
        p[i].y += origin.y;
        tags[i].a = op;
        tags[i].b = make_uint4(i, i, i, i);
    }
}

// fill sets out to pair, whose Go name is also that of its type.
extern "C" __global__ void fill(Pair pair, Pair* out) {
    *out = pair;
}

// sum adds the n values of in into *out, one block, through shared memory.
extern "C" __global__ void sum(const int* in, int n, int* out) {
    extern __shared__ int s[];
    int t = threadIdx.x;
    s[t] = t < n ? in[t] : 0;
    __syncthreads();
    if (t == 0) {
        int total = 0;
        for (int i = 0; i < blockDim.x; i++) {
            total += s[i];
        }
        *out = total;
    }
}
