import torch

def uniform_sample(input, sample_cnt, add_dataptr=True):
    num_elem = input.numel()
    sample_cnt = min(num_elem, sample_cnt)
    key = []
    if add_dataptr:
        key.append(input.data_ptr())
    key += input.ravel()[torch.arange(0, sample_cnt).to(torch.long) *
                          (num_elem // sample_cnt)].tolist()
    return key

def get_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    if print_info:
        print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
        print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)
    return allocated

def compute_tensor_bytes(tensors):
    """Compute the bytes used by a list of tensors"""
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    ret = 0
    for x in tensors:
        if type(x)== int:
            ret += 4
        elif x.dtype in [torch.long]:
            ret += np.prod(x.size()) * 8
        elif x.dtype in [torch.float32, torch.int]:
            ret += np.prod(x.size()) * 4
        elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
            ret += np.prod(x.size()) * 2
        elif x.dtype in [torch.int8, torch.uint8]:
            ret += np.prod(x.size()) * 1
        else:
            print("[Error] unsupport datatype ", x.dtype)
            exit(0)

    return ret
