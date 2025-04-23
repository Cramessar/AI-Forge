import psutil
import platform
try:
    import GPUtil
except ImportError:
    GPUtil = None


def get_system_profile():
    ram = round(psutil.virtual_memory().total / (1024**3), 2)
    cpu = platform.processor()
    gpu_name = "None"
    gpu_mem_gb = 0

    if GPUtil:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_name = gpus[0].name
            gpu_mem_gb = round(gpus[0].memoryTotal / 1024, 2)

    return {
        "total_ram_gb": ram,
        "cpu_name": cpu,
        "gpu_name": gpu_name,
        "gpu_mem_gb": gpu_mem_gb,
    }


def get_tuned_generation_settings(profile=None):
    """
    Returns optimal generation settings (temperature and max_tokens) 
    based on system specs like RAM, CPU, and GPU.
    """
    if profile is None:
        profile = get_system_profile()

    ram = profile.get("total_ram_gb", 8)
    cpu = profile.get("cpu_name", "").lower()
    gpu = profile.get("gpu_name", "").lower()
    gpu_mem = profile.get("gpu_mem_gb", 0)

    #advanced logic
    if "intel" in cpu and ram < 8:
        return {"temperature": 0.6, "max_tokens": 512}
    if "ryzen" in cpu and ram < 16:
        return {"temperature": 0.7, "max_tokens": 768}
    if "nvidia" in gpu and ram >= 16 and gpu_mem >= 8:
        return {"temperature": 0.8, "max_tokens": 1024}
    if "apple" in cpu or "m1" in cpu or "m2" in cpu:
        return {"temperature": 0.75, "max_tokens": 1024}

    # fallback tiered logic
    if gpu_mem >= 12 and ram >= 32:
        return {"temperature": 0.9, "max_tokens": 1536}
    elif gpu_mem >= 8 and ram >= 16:
        return {"temperature": 0.8, "max_tokens": 1024}
    elif gpu_mem >= 4 or ram >= 12:
        return {"temperature": 0.7, "max_tokens": 768}
    else:
        return {"temperature": 0.6, "max_tokens": 512}

