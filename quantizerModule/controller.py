import torch
from config import config

class Controller:
    def __init__(self, model) -> None:
        self.model = model
        self.Assert()
        
    
    def __del__(self) -> None:
        pass
    
    def Assert() -> None:
        assert config.bh < 16 and config.bh > 1
        assert config.bl < 16 and config.bl > 1
        assert config.bh < config.bl
        assert config.rh < 1 and config.rh > 0
        assert config.rl < 1 and config.rl > 0
        assert config.rh > config.rl
    
    