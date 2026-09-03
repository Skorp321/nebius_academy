
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def naive_allreduce(x: torch.Tensor, pg: dist.ProcessGroup):
    group_rank = dist.get_rank(group=pg)
    group_size = dist.get_world_size(group=pg)
    root = 0

    if group_rank == root:
        total = x.clone()
        for src in range(1, group_size):
            buf = torch.empty_like(x)
            dist.recv(buf, group=pg, group_src=src)
            total += buf

        x.copy_(total)

        for dst in range(1, group_size):
            dist.send(total, group=pg, group_dst=dst)
    else:
        dist.send(x, group=pg, group_dst=root)
        dist.recv(x, group=pg, group_src=root)

def main():
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cpu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    rank = dist.get_rank()
    x = (torch.arange(4) + rank * 4).float()
    print(f"[rank {rank}] allreduce input: {x}", flush=True)

    pg = mesh.get_group("dp")
    naive_allreduce(x, pg)

    print(f"[rank {rank}] allreduce output: {x}", flush=True)

if __name__ == "__main__":
    main()
