import torch
from config import config
from handler import op_quantize, op_dequantize, op_quantize_mask, op_dequantize_mask, op_dequantize_cvbn, op_quantize_cvbn
from utils import uniform_sample


class Quantizer:
    def __init__(self, bh, bl, rh, rl):
        self.unrelated_tensors = set()  
        self.default_bit = default_bit

        self.ptr_qtensor_map = {}
        self.layer_key_map = {}

        self.seeds = {}
        self.bits = {}
        self.dims = {}

        self.iter = 0  
        self.seed_iter = 0

    def filter_tensors(self, pairs):
        for _, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    def check_quantize(self, input_tensor):
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False, False
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            if (input_tensor.max() == 1) and (input_tensor.min() == 0):
                return True, True
            return False, False
        if input_tensor.dtype not in [torch.float32, torch.float16]:
            return False, False
        if input_tensor.requires_grad is False:
            return False, False
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
            ):
            return False, False
        return True, False

    def __del__(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        del self.unrelated_tensors

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.iter += 1

    def generate_tensor_key(self, t, tid):
        if config.check_dup:
            sample_cnt = min(100, t.numel())
            key = uniform_sample(t, sample_cnt, add_dataptr=True)
            key.append(t.sum().item())
            return tuple(key)
        else:
            return (tid)

    def quantize(self, input):
        quantize, is_dropout_mask = self.check_quantize(input)

        if not quantize:
            return False, input

        if is_dropout_mask:
            q_inputs = op_quantize_mask(input)
            return True, is_dropout_mask, q_inputs

        tid = self.tid
        self.tid += 1
        input_shape = input.shape

        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map

        if not skip_quantize:
            if self.iter == 0:
                bit = self.default_bit
                self.bits[tid] = bit
                self.dims[tid] = input.numel()
                self.seeds[tid] = tid
            else:
                bit = self.bits[tid]
            if input.dim() == 4:
                q_inputs = op_quantize_cvbn(input, bit, self.seeds[tid] + self.seed_iter)
            else:
                q_inputs = op_quantize(input, bit, self.seeds[tid] + self.seed_iter)
            self.ptr_qtensor_map[key] = [q_inputs, 1, tid]
        else:
            self.ptr_qtensor_map[key][1] += 1
        return True, is_dropout_mask, key, input_shape, tid

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            return input[1]
        is_dropout_mask = input[1]
        if is_dropout_mask:
            _, is_dropout_mask, q_inputs = input
            ret = op_dequantize_mask(q_inputs)
            return ret

        _, _, key, input_shape, tid = input
        q_inputs, ref_cnt, key_tid = self.ptr_qtensor_map[key]

        if not q_inputs[0].is_cuda:
            q_inputs[0] = q_inputs[0].cuda(non_blocking=False)
                    
        if len(input_shape) == 4:
            ret = op_dequantize_cvbn(q_inputs, input_shape)
        else:
            ret = op_dequantize(q_inputs, input_shape)

        ref_cnt -= 1
        if ref_cnt < 0:
            print("[Error] Ref count < 0", key, ref_cnt)
            exit(-1)
        elif ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret
