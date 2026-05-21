# env_init.py
import os
import subprocess


def pick_free_gpu():
    # Query GPU memory usage
    output = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True
    )
    usage = [int(x) for x in output.decode().strip().split("\n")]
    free_gpus = [i for i, u in enumerate(usage) if u < 10]

    if not free_gpus:
        print("[env_init] No free GPUs available")
        return None

    # Return first two indices from free_gpus
    # If only one free GPU is available, return that one.
    return free_gpus[:2]


gpus = pick_free_gpu()

if gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    print(f"[env_init] Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    # Optional: fall back to CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("[env_init] Falling back to CPU")
