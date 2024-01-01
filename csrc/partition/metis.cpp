#include <torch/all.h>
#include <metis.h>
#include "utils.h"


at::Tensor metis_partition(
    at::Tensor rowptr,
    at::Tensor col,
    at::optional<at::Tensor> opt_value,
    at::optional<at::Tensor> opt_vtx_w,
    at::optional<at::Tensor> opt_vtx_s,
    int64_t num_parts,
    bool recursive,
    bool min_edge_cut
) {
    static_assert(sizeof(idx_t) == sizeof(int64_t));
    static_assert(sizeof(real_t) == sizeof(float));

    CHECK_CPU(rowptr);
    AT_ASSERT(rowptr.dim() == 1);
    AT_ASSERT(rowptr.is_contiguous());

    CHECK_CPU(col);
    AT_ASSERT(col.dim() == 1);
    AT_ASSERT(col.is_contiguous());

    idx_t nvtxs = rowptr.numel() - 1;
    idx_t ncon = 1;

    idx_t *xadj = rowptr.data_ptr<idx_t>();
    idx_t *adjncy = col.data_ptr<idx_t>();

    // edge weights
    idx_t *adjwgt = nullptr;
    if (opt_value.has_value()) {
        CHECK_CPU(opt_value.value());
        AT_ASSERT(opt_value.value().dim() == 1);
        AT_ASSERT(opt_value.value().numel() == col.numel());
        AT_ASSERT(opt_value.value().is_contiguous());

        adjwgt = opt_value.value().data_ptr<idx_t>();
    }

    // node weights
    idx_t *vwgt = nullptr;
    if (opt_vtx_w.has_value()) {
        CHECK_CPU(opt_vtx_w.value());
        AT_ASSERT(opt_vtx_w.value().dim() <= 2);
        AT_ASSERT(opt_vtx_w.value().size(0) == nvtxs);
        AT_ASSERT(opt_vtx_w.value().is_contiguous());

        vwgt = opt_vtx_w.value().data_ptr<idx_t>();
        if (opt_vtx_w.value().dim() == 2) {
            ncon = opt_vtx_w.value().size(1);
        }
    }

    idx_t *vsize = nullptr;
    if (opt_vtx_s.has_value()) {
        CHECK_CPU(opt_vtx_s.value());
        AT_ASSERT(opt_vtx_s.value().dim() == 1);
        AT_ASSERT(opt_vtx_s.value().numel() == nvtxs);
        AT_ASSERT(opt_vtx_s.value().is_contiguous());

        vsize = opt_vtx_s.value().data_ptr<idx_t>();
    }

    idx_t nparts = num_parts;
    idx_t objval = -1;

    auto part = at::empty({nvtxs}, rowptr.options());
    idx_t *part_data = part.data_ptr<idx_t>();

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    if (min_edge_cut) {
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    } else {
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    }

    int ret;
    if (recursive) {
        ret = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
                             &nparts, NULL, NULL, options, &objval, part_data);
    } else {
        ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
                        &nparts, NULL, NULL, options, &objval, part_data);
    }

    AT_ASSERT(ret == METIS_OK);
    return part;
}


at::Tensor metis_cache_friendly_reordering(
    at::Tensor rowptr,
    at::Tensor col,
    at::Tensor part
) {
    static_assert(sizeof(idx_t) == sizeof(int64_t));
    static_assert(sizeof(real_t) == sizeof(float));

    CHECK_CPU(rowptr);
    AT_ASSERT(rowptr.dim() == 1);
    AT_ASSERT(rowptr.is_contiguous());

    CHECK_CPU(col);
    AT_ASSERT(col.dim() == 1);
    AT_ASSERT(col.is_contiguous());

    CHECK_CPU(part);
    AT_ASSERT(part.dim() == 1);
    AT_ASSERT(part.is_contiguous());

    idx_t nvtxs = rowptr.numel() - 1;

    idx_t *xadj = rowptr.data_ptr<idx_t>();
    idx_t *adjncy = col.data_ptr<idx_t>();

    idx_t *part_data = part.data_ptr<idx_t>();

    auto old2new = at::empty({nvtxs}, rowptr.options());
    idx_t *old2new_data = old2new.data_ptr<idx_t>();

    int ret = METIS_CacheFriendlyReordering(nvtxs, xadj, adjncy, part_data, old2new_data);
    AT_ASSERT(ret == METIS_OK);
    return old2new;
}