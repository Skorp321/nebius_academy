
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

def main():
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cpu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    rank = dist.get_rank()
    x = (torch.arange(4) + rank * 4).float()
    print(f"[rank {rank}] allreduce input: {x}", flush=True)

    pg = mesh.get_group("dp")
    dist.all_reduce(x, group=pg)
    print(f"[rank {rank}] allreduce output: {x}", flush=True)

if __name__ == "__main__":
    main()
