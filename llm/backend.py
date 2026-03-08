from abc import ABC, abstractmethod
from typing import List, Dict

BACKEND_REGISTRY: dict[str, type] = {}

def register_backend(name: str):
    def decorator(cls):
        BACKEND_REGISTRY[name] = cls
        return cls
    return decorator

def create_backend(args) -> "LLMBackend":
    try:
        backend_cls = BACKEND_REGISTRY[args.backend]
    except KeyError as exc:
        valid = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown backend {args.backend!r}. Valid backends: {valid}"
        ) from exc
    return backend_cls.from_args(args)

class LLMBackend(ABC):

    @classmethod
    def from_args(cls, args) -> "Backend":
        raise NotImplementedError(
            f"{cls.__name__}.from_args() must be implemented."
        )

    @abstractmethod
    async def batch_decide(self, payloads: List[dict]) -> Dict:
        """
        Given agent payloads, return decisions.
        """
        pass
