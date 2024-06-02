import torch
from config import config
from quantizer import Quantizer

class Controller:
    def __init__(self, model) -> None:
        self.model = model
        self.Assert()
        
        self.quantizer = Quantizer(
            bh=config.bh, bl=config.bl, rh=config.rh, rl=config.rl)
        self.quantizer.filter_tensors(model.named_parameters())
        self.auto_prec = config.auto_prec
        if self.auto_prec:
            self.ap = AutoPrecision(
                self.model, self.quantizer, config.bit, config.max_bit,
                config.work_dir, config.adapt_interval, config.log_interval)
            
        self.iter = 0
        
    def __del__(self) -> None:
        pass
    
    def Assert() -> None:
        assert config.bh < 16 and config.bh > 1
        assert config.bl < 16 and config.bl > 1
        assert config.bh < config.bl
        assert config.rh < 1 and config.rh > 0
        assert config.rl < 1 and config.rl > 0
        assert config.rh > config.rl
    
    def install_hook(self):
        def pack_hook(x):
            r = self.quantize(x)
            del x
            return r

        def unpack_hook(x):
            r = self.dequantize(x)
            del x
            return r

        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._register_saved_tensors_default_hooks(
                pack_hook, unpack_hook)
        else:
            torch._C._autograd._push_saved_tensors_default_hooks(
                pack_hook, unpack_hook)
            
    def uninstall_hook(self):
        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._reset_saved_tensors_default_hooks()
        else:
            torch._C._autograd._pop_saved_tensors_default_hooks()

    def iterate(self, get_grad):
        self.quantizer.iterate()
        if self.auto_prec:
            self.ap.iterate_wrapper(get_grad)
        self.iter += 1
        self.quantizer.seed_iter = self.iter
            
    def quantize(self, input):
        return self.quantizer.quantize(input)

    def dequantize(self, input):
        return self.quantizer.dequantize(input)
    