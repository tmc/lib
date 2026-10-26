// Race test for ready.cuh: producer threads ("blocks") write their slots
// of a dependency, then signal it; one consumer per dependency waits for
// all of them, then checks every slot holds this epoch's value. Threads
// start in a random order with random delays, the number of producers per
// dependency changes every epoch, and the counters are never reset.
// Built with -DNO_WAIT (the consumer skips the wait) it must report
// failures; see ready_test.go.
//
// It checks the counting, not memory ordering: the host's fences stand in
// for the device's, and dropping them does not make this test fail.
#include "ready_host.h"
#include "../ready.cuh"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

int main() {
  constexpr int DEPS = 8, MAXP = 6, EPOCHS = 300;
  std::vector<unsigned long long> ctr(DEPS, 0);
  std::vector<std::atomic<unsigned>> data(DEPS * MAXP);
  std::mt19937 rng(12345);
  long checked = 0, bad = 0;
  for (unsigned epoch = 1; epoch <= EPOCHS; epoch++) {
    int np[DEPS], units[DEPS] = {};
    for (int d = 0; d < DEPS; d++) np[d] = 1 + rng() % MAXP;
    std::vector<std::function<void()>> work;
    for (int d = 0; d < DEPS; d++) {
      for (int i = 0; i < np[d]; i++) {
        const int us = rng() % 300;
        // Each producer signals 1 or 2 units, so counts are not producer
        // counts and a wait one unit short misses a producer.
        const unsigned n = 1 + rng() % 2;
        units[d] += n;
        work.push_back([&, d, i, us, n] {
          std::this_thread::sleep_for(std::chrono::microseconds(us));
          data[d * MAXP + i].store(epoch, std::memory_order_relaxed);
          ready::signal(ctr.data(), d, epoch, n);
        });
      }
      work.push_back([&, d] {
#ifndef NO_WAIT
        ready::wait(ctr.data(), d, units[d], epoch);
#endif
        for (int i = 0; i < np[d]; i++) {
          checked++;
          if (data[d * MAXP + i].load(std::memory_order_relaxed) != epoch) bad++;
        }
      });
    }
    std::shuffle(work.begin(), work.end(), rng);
    std::vector<std::thread> th;
    for (auto& f : work) th.emplace_back(f);
    for (auto& t : th) t.join();
    // Every counter used this epoch ends at exactly its signaled count.
    for (int d = 0; d < DEPS; d++)
      if (ctr[d] != ((unsigned long long)epoch << 32 | (unsigned)units[d])) {
        std::printf("FAIL epoch %u dep %d: counter %#llx\n", epoch, d, ctr[d]);
        return 1;
      }
  }
  std::printf("%s: %ld slots checked, %ld stale\n", bad ? "FAIL" : "ok", checked, bad);
  return bad ? 1 : 0;
}
