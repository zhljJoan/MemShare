#include <torch/all.h>
#include "utils.h"

#ifdef WITH_MTMETIS

#include <mtmetis.h>

at::Tensor mt_metis_partition(
    at::Tensor rowptr,
    at::Tensor col,
    at::optional<at::Tensor> opt_value,
    at::optional<at::Tensor> opt_vtx_w,
    int64_t num_parts,
    int64_t num_workers,
    bool recursive
) {
    static_assert(sizeof(mtmetis_vtx_type) == sizeof(int64_t));
    static_assert(sizeof(mtmetis_adj_type) == sizeof(int64_t));
    static_assert(sizeof(mtmetis_wgt_type) == sizeof(int64_t));
    static_assert(sizeof(mtmetis_pid_type) == sizeof(int64_t));

    CHECK_CPU(rowptr);
    AT_ASSERT(rowptr.dim() == 1);
    AT_ASSERT(rowptr.is_contiguous());

    CHECK_CPU(col);
    AT_ASSERT(col.dim() == 1);
    AT_ASSERT(col.is_contiguous());

    mtmetis_vtx_type nvtxs = rowptr.numel() - 1;
    mtmetis_vtx_type ncon = 1;

    mtmetis_adj_type *xadj = reinterpret_cast<mtmetis_adj_type*>(rowptr.data_ptr<int64_t>());
    mtmetis_vtx_type *adjncy = reinterpret_cast<mtmetis_vtx_type*>(col.data_ptr<int64_t>());

    // edge weights
    mtmetis_wgt_type *adjwgt = NULL;
    if (opt_value.has_value()) {
        CHECK_CPU(opt_value.value());
        AT_ASSERT(opt_value.value().dim() == 1);
        AT_ASSERT(opt_value.value().numel() == col.numel());
        AT_ASSERT(opt_value.value().is_contiguous());

        adjwgt = reinterpret_cast<mtmetis_wgt_type*>(opt_value.value().data_ptr<int64_t>());
    }

    // node weights
    mtmetis_wgt_type *vwgt = NULL;
    if (opt_vtx_w.has_value()) {
        CHECK_CPU(opt_vtx_w.value());
        AT_ASSERT(opt_vtx_w.value().dim() <= 2);
        AT_ASSERT(opt_vtx_w.value().size(0) == nvtxs);
        AT_ASSERT(opt_vtx_w.value().is_contiguous());

        vwgt = reinterpret_cast<mtmetis_wgt_type*>(opt_vtx_w.value().data_ptr<int64_t>());
        if (opt_vtx_w.value().dim() == 2) {
            ncon = opt_vtx_w.value().size(1);
        }
    }

    mtmetis_pid_type nparts = num_parts;
    mtmetis_wgt_type objval = -1;

    auto part = at::empty({static_cast<int64_t>(nvtxs)}, rowptr.options());
    mtmetis_pid_type *part_data = reinterpret_cast<mtmetis_pid_type*>(part.data_ptr<int64_t>());

    double *options = mtmetis_init_options();
    options[MTMETIS_OPTION_NTHREADS] = num_workers;

    int ret;
    if (recursive) {
        ret = MTMETIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                                &nparts, NULL, NULL, options, &objval, part_data);
    } else {
        ret = MTMETIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                            &nparts, NULL, NULL, options, &objval, part_data);
    }

    AT_ASSERT(ret == MTMETIS_SUCCESS);
    return part;
}

#endif