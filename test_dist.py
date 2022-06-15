import datetime
import os
import torch
import torch.distributed as dist

backend = 'nccl'
dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=30))

local_rank = os.environ["LOCAL_RANK"]

print ("test all-reduce")
a = torch.ones(3).to(f"cuda:{local_rank}")
dist.all_reduce(a)
print (a)


print ("test broadcast")
dist.broadcast(a, src=0)
dist.barrier()
print (a)
