import datetime

import torch

import composer.utils.dist as dist

dist_backend = 'nccl'
dist.initialize_dist(backend=dist_backend, timeout=datetime.timedelta(seconds=30))
dist.barrier()

print (f"rank={dist.get_global_rank()}")

a = torch.ones(3).cuda()
print ("test all-reduce")
dist.all_reduce(a, reduce_operation='SUM')
print (a)

print ("test broadcast")
dist.broadcast(a, src=0)
print (a)
