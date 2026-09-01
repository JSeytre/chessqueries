"""Shared argument validation for the released training entry points."""
import argparse


def single_gpu_devices(value: str) -> int:
    """Accept only the one-GPU configuration used by the released recipes."""
    devices = int(value)
    if devices != 1:
        raise argparse.ArgumentTypeError(
            "the released training recipe supports one GPU; --devices must be 1"
        )
    return devices
