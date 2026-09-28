// golden.c writes testdata/golden.txt: the network outputs PufferLib's own
// CPU forward pass computes for the cases in golden_test.go. Build it against
// a PufferLib 5.0 checkout:
//
//	gcc -O2 -ffp-contract=off -I$PUFFERLIB/src -o $HOME/tmp/puffernet-golden testdata/golden.c -lm
//	$HOME/tmp/puffernet-golden > testdata/golden.txt
//
// Weights, observations and terminals come from the same LCG as the test.
#include "puffercpu.c"

static unsigned int seed;

static float next(void) {
    seed = seed * 1664525u + 1013904223u;
    return (float)((int)(seed >> 9) % 2001 - 1000) / 2000.0f;
}

static int align8(int n) { return (n + 7) & ~7; }

static void run(const char* name, int in, int hidden, int layers,
        int* acts, int nacts, int batch, int steps) {
    int atn = 0, cont = 1;
    for (int i = 0; i < nacts; i++) {
        atn += acts[i];
        if (acts[i] != 1) cont = 0;
    }
    int n = align8(hidden * in);
    n = align8(n + (atn + 1) * hidden);
    if (cont) n = align8(n + nacts);
    for (int l = 0; l < layers; l++) n = align8(n + 3 * hidden * hidden);

    seed = 1;
    Weights* w = calloc(1, sizeof(Weights) + (n + 7) * sizeof(float));
    w->data = (float*)(w + 1);
    w->size = n + 7;
    for (int i = 0; i < n; i++) w->data[i] = next();

    PufferNet* net = make_puffernet(w, batch, in, hidden, layers, acts, nacts);
    float* obs = calloc(batch * in, sizeof(float));
    float* actions = calloc(batch * nacts, sizeof(float));
    float* term = calloc(batch, sizeof(float));
    printf("case %s\n", name);
    for (int t = 0; t < steps; t++) {
        for (int i = 0; i < batch * in; i++) obs[i] = 2 * next();
        for (int b = 0; b < batch; b++) term[b] = (t * batch + b) % 5 == 4;
        forward_puffernet(net, obs, actions, NULL, term);
        for (int i = 0; i < batch * (atn + 1); i++) {
            printf("%s%.9g", i ? " " : "", net->decoder->output[i]);
        }
        printf("\n");
    }
    free_puffernet(net);
    free(obs);
    free(actions);
    free(term);
    free(w);
}

int main(void) {
    int breakout[] = {3};
    int continuous[] = {1, 1};
    int multi[] = {2, 4};
    run("breakout", 118, 32, 4, breakout, 1, 3, 6);
    run("continuous", 5, 12, 2, continuous, 2, 2, 6);
    run("multidiscrete", 9, 8, 3, multi, 2, 2, 5);
    return 0;
}
