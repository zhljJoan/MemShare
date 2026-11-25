#include "parallel_hashmap/phmap.h"
#include <algorithm>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py = pybind11;
using I = int64_t;
using Array = py::array_t<I, py::array::c_style>;
using Vec = std::vector<I>;
struct Objects {
  phmap::flat_hash_map<I, I> slot;
  Vec keys;
  I add(I key) {
    auto entry = slot.emplace(key, keys.size());
    if (entry.second)
      keys.push_back(key);
    return entry.first->second;
  }
  void sort_and_remap(Vec &references) {
    std::sort(keys.begin(), keys.end());
    Vec remap(keys.size());
    for (size_t i = 0; i < keys.size(); ++i)
      remap[slot.at(keys[i])] = i;
    for (I &x : references)
      x = remap[x];
  }
};
py::object compact(Array roots, Array rt, Array nbr, Array nt, Array edge,
                  Array owner, Array node_map, Array edge_map, bool reverse,
                  int rank) {
  auto q = roots.unchecked<1>(), t = rt.unchecked<1>(), n = nbr.unchecked<1>(),
       ts = nt.unchecked<1>(), e = edge.unchecked<1>(),
       o = owner.unchecked<1>(), nm = node_map.unchecked<1>(),
       em = edge_map.unchecked<1>();
  I nr = q.shape(0), ne = n.shape(0);
  if (nr != t.shape(0) || ne != ts.shape(0) || ne != e.shape(0) ||
      ne != o.shape(0))
    throw std::invalid_argument("unaligned sampled fields");
  Objects nodes, edges;
  phmap::flat_hash_map<std::pair<I, I>, I> states;
  // Order states by first occurrence, including roots before sampled neighbors.
  Vec uid, stamp, root(nr), row(ne), col(ne), edge_ref, delta;
  I root_states = 0, local_roots = 0;
  bool exact_in_double = true;
  {
    py::gil_scoped_release release;
    auto mapped = [&](I id) {
      if (id < 0 || id >= nm.shape(0))
        throw std::out_of_range("node id");
      return nm(id);
    };
    auto state = [&](I node, I time) {
      constexpr I limit = I{1} << 53;
      exact_in_double &= time >= -limit && time <= limit;
      auto entry = states.emplace(std::make_pair(node, time), uid.size());
      if (entry.second) {
        uid.push_back(nodes.add(node));
        stamp.push_back(time);
      }
      return entry.first->second;
    };
    for (I i = 0; i < nr; ++i) {
      I node = mapped(q(i));
      root[i] = state(node, t(i));
      local_roots += ((node >> 50) & 65535) == rank;
    }
    root_states = uid.size();
    if (!reverse) {
      edge_ref.resize(ne);
      delta.resize(ne);
    }
    for (I i = 0; i < ne; ++i) {
      I src = o(i);
      if (src < 0 || src >= nr)
        throw std::out_of_range("root occurrence");
      I node = mapped(n(i));
      if (reverse) {
        row[i] = root[src];
        col[i] = node;
      } else {
        if (e(i) < 0 || e(i) >= em.shape(0))
          throw std::out_of_range("edge id");
        row[i] = state(node, ts(i));
        col[i] = root[src];
        edge_ref[i] = edges.add(em(e(i)));
        if (exact_in_double)
          delta[i] = t(src) - ts(i);
      }
    }
    nodes.sort_and_remap(uid);
    edges.sort_and_remap(edge_ref);
  }
  // The original GPU builder compares float64 timestamps.
  if (!exact_in_double)
    return py::none();
  std::vector<const Vec *> fields = {&nodes.keys, &edges.keys, &uid,
                                     &stamp,      &root,       &row,
                                     &col,        &edge_ref,   &delta};
  Vec lengths;
  for (auto f : fields)
    lengths.push_back(f->size());
  Array packed(std::accumulate(lengths.begin(), lengths.end(), I{0}));
  auto dst = packed.mutable_data();
  for (auto f : fields)
    dst = std::copy(f->begin(), f->end(), dst);
  py::list sizes;
  for (auto size : lengths)
    sizes.append(size);
  return py::make_tuple(packed, sizes, root_states, local_roots);
}
PYBIND11_MODULE(memshare_packed_block, m) { m.def("compact", &compact); }
