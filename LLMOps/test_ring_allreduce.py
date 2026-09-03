
import os, torch, torch.distributed as dist, torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import init_device_mesh
sent = 0
def wrap(f):
    def g(*a, **k):
        global sent; sent += (a[0] if a else k["tensor"]).numel(); return f(*a, **k)
    return g
def nope(*a, **k):
    raise AssertionError("built-in collective")
w, r = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
pg = init_device_mesh("cpu", (w,), mesh_dim_names=("dp",)).get_group("dp")
pg2 = [dist.new_group([0, 1]), dist.new_group([2, 3])][r // 2] if w == 4 else None
dist.send = c10d.send = wrap(c10d.send)
dist.isend = c10d.isend = wrap(c10d.isend)
for m in (dist, c10d):
    for n in "all_reduce reduce broadcast all_gather all_gather_into_tensor gather scatter reduce_scatter reduce_scatter_tensor all_to_all all_to_all_single".split():
        if hasattr(m, n): setattr(m, n, nope)
from ring_allreduce import ring_allreduce
def check(b, s, c, pg):
    global sent
    ranks, g = dist.get_process_group_ranks(pg), dist.get_world_size(pg)
    x = b * s(r) + c(r)
    y = sum(b * s(i) + c(i) for i in ranks)
    ptr, old = x.data_ptr(), sent
    ring_allreduce(x, pg)
    assert x.data_ptr() == ptr and x.shape == y.shape and x.dtype == y.dtype
    assert torch.allclose(x, y, rtol=1e-5, atol=1e-7), (r, x, y)
    assert 0 < sent - old <= 2 * x.numel() * (g - 1) // g, "wrong communication pattern"
for b, s, c in [
    (torch.arange(4*w, dtype=torch.float32).reshape(w, 4) / 3 - 2, lambda i: i + 1, lambda i: i - 2),
    (torch.linspace(-1, 1, 2*w, dtype=torch.float64), lambda i: i*i + 1, lambda i: -i),
    (torch.arange(w*(w-1), dtype=torch.float32).reshape(w-1, w) / 7, lambda i: i - 1, lambda i: 2 - i),
]:
    check(b, s, c, pg)
if w == 4:
    check(torch.arange(8, dtype=torch.float32), lambda i: i + 1, lambda i: -i, pg2)
print(f"[rank {r}] Tests for Puzzle A passed!", flush=True)
