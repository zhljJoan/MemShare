#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using A = py::array_t<int64_t, py::array::c_style>;
uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
void sample(A ptr, A nbr, A ts, A remote, A roots, A times, A ids, A out,
            int fanout, int threads, bool coordinated, double beta, int group,
            uint64_t seed, uint64_t batch) {
  auto p = ptr.unchecked<1>(), n = nbr.unchecked<1>(), t = ts.unchecked<1>();
  auto r = remote.unchecked<1>(), q = roots.unchecked<1>(),
       qt = times.unchecked<1>();
  auto qi = ids.unchecked<1>();
  auto o = out.mutable_unchecked<2>();
  if (fanout < 1 || threads < 1 || group < 1 || o.shape(0) != q.shape(0) ||
      o.shape(1) != fanout + 1 || qi.shape(0) != q.shape(0) ||
      qt.shape(0) != q.shape(0))
    throw std::invalid_argument(
        "invalid query/output shapes or sampling options");
  for (int64_t i = 0; i < q.shape(0); ++i)
    if (q(i) < 0 || q(i) + 1 >= p.shape(0) || qi(i) < 0)
      throw std::invalid_argument("invalid root/query id");
  py::gil_scoped_release release;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t i = 0; i < q.shape(0); ++i) {
    for (int j = 0; j <= fanout; ++j)
      o(i, j) = -1;
    int64_t lo = p(q(i)), hi = p(q(i) + 1),
            end = std::lower_bound(t.data(0) + lo, t.data(0) + hi, qt(i)) -
                  t.data(0);
    if (!coordinated) {
      int64_t start = std::max(lo, end - fanout);
      for (int64_t j = start; j < end; ++j)
        o(i, j - start) = j;
      continue;
    }
    int64_t first = std::max(lo, end - fanout);
    double sum = 0;
    int rem = 0;
    double delta = end > lo ? double((qt(i) - t(end - 1)) * fanout) : 1.;
    for (int64_t j = end - 1; j >= first; --j)
      if (r(j)) {
        sum += std::exp(double(t(j) - qt(i)) / delta);
        ++rem;
      }
    if (sum < 1e-6)
      sum = 1;
    int count = 0;
    for (int64_t j = end - 1; j >= lo && count <= fanout; --j) {
      if (r(j)) {
        if (j < first)
          continue;
        uint64_t k = seed ^ mix(n(j)) ^
                     mix(uint64_t(qi(i) / group) + 0xD1B54A32D192ED03ULL) ^
                     mix(batch + 0x94D049BB133111EBULL) ^
                     mix(0xBF58476D1CE4E5B9ULL);
        double u = double(mix(k) >> 11) / double(1ULL << 53);
        if (u > beta * std::exp(double(t(j) - qt(i)) / delta) / sum * rem)
          continue;
      }
      // Preserve the production sampler's fanout+1 stopping rule for boundary
      // policies.
      o(i, count++) = j;
    }
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sample", &sample); }
