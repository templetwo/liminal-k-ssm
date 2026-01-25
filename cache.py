import mlx.core as mx

class MambaCache:
    def __init__(self):
        self.conv_state = None
        self.ssm_state = None

    def __getitem__(self, idx):
        if idx == 0: return self.conv_state
        if idx == 1: return self.ssm_state
        raise IndexError

    def __setitem__(self, idx, value):
        if idx == 0: self.conv_state = value
        elif idx == 1: self.ssm_state = value
        else: raise IndexError
