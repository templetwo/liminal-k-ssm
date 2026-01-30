from dataclasses import dataclass

@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})
