# import torch
# from starrygl.core.acopy import get_executor, AsyncCopyWorkBase, List


# if __name__ == "__main__":
#     data = torch.arange(1024*1024*1024, dtype=torch.float32).view(-1, 1).repeat(1, 8)
#     # buffer = torch.arange(1000, dtype=torch.float32).view(-1, 1).repeat(1, 8).pin_memory()
#     index = torch.arange(1024*1024 * 256, dtype=torch.long).cuda()
#     values = torch.ones(1024*1024 * 256, 8, dtype=torch.float32).cuda()

#     pull_works: List[AsyncCopyWorkBase] = []
#     push_works: List[AsyncCopyWorkBase] = []
#     for s in range(0, index.size(0), 1024 * 1024 * 256):
#         t = min(s + 1024 * 1024 * 256, index.size(0))
#         idx = index[s:t]
#         val = values[idx]
#         pullw = get_executor().async_pull(data, idx)
#         pushw = get_executor().async_push(data, idx, val)
#         pull_works.append(pullw)
#         push_works.append(pushw)
    
#     pull_time = 0.0
#     for w in pull_works:
#         val = w.get()
#         pull_time += w.time_used()
#         # print(val)
    
#     push_time = 0.0
#     for w in push_works:
#         w.get()
#         push_time += w.time_used()
#     # print(data)

#     print(pull_time, push_time)
        