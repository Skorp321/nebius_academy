
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def ring_allreduce(x: torch.Tensor, pg: dist.ProcessGroup):
    group_rank = dist.get_rank(group=pg)
    group_size = dist.get_world_size(group=pg)

    if group_size == 1:
        return

    left = (group_rank - 1 + group_size) % group_size
    right = (group_rank + 1) % group_size

    flat = x.view(-1)
    chunks = list(flat.chunk(group_size))

    # Phase 1: reduce-scatter.
    # Each rank circulates chunks around the ring and accumulates partial sums.
    for step in range(group_size - 1):
        send_idx = (group_rank - step) % group_size
        recv_idx = (group_rank - step - 1) % group_size

        recv_buf = torch.empty_like(chunks[recv_idx])

        req = dist.isend(chunks[send_idx], group=pg, group_dst=right)
        dist.recv(recv_buf, group=pg, group_src=left)
        req.wait()

        chunks[recv_idx].add_(recv_buf)

    # Phase 2: all-gather.
    # Now every rank owns one fully reduced chunk; circulate those chunks
    # so every rank reconstructs the full reduced tensor.
    for step in range(group_size - 1):
        send_idx = (group_rank + 1 - step) % group_size
        recv_idx = (group_rank - step) % group_size

        recv_buf = torch.empty_like(chunks[recv_idx])

        req = dist.isend(chunks[send_idx], group=pg, group_dst=right)
        dist.recv(recv_buf, group=pg, group_src=left)
        req.wait()

        chunks[recv_idx].copy_(recv_buf)


def main():
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cpu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    rank = dist.get_rank()
    x = (torch.arange(4) + rank * 4).float()
    print(f"[rank {rank}] allreduce input: {x}", flush=True)

    pg = mesh.get_group("dp")
    ring_allreduce(x, pg)

    print(f"[rank {rank}] allreduce output: {x}", flush=True)

if __name__ == "__main__":
    main()
