import torch
import torch.distributed as dist

""" utils from openseg codebase """
def is_distributed():
    return dist.is_initialized()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def all_reduce_numpy(array):
    tensor = torch.from_numpy(array).cuda()
    dist.all_reduce(tensor)
    return tensor.cpu().numpy()

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def barrier():
    """Synchronizes all processes.

    This collective blocks processes until the whole group enters this
    function.
    """
    if dist.is_initialized():
        dist.barrier() # processes in global group wait here until all processes reach this point
    return

@torch.no_grad()
def concat_all_gather(tensor, concat_dim=0):
    """ from moco
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=concat_dim)
    return output
