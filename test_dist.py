import datetime
import torch
import torch.distributed as dist

backend = 'nccl'
dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=30))
dist.barrier()

print (f"rank={dist.get_rank()}")

print ("test broadcast")
a = torch.ones(3).cuda()
dist.broadcast(a, src=0)
print (a)
