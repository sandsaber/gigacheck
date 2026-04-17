import os
import subprocess

import torch
from loguru import logger


def _setup_rocm_env():
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return

    rocm_path = os.environ.get("ROCM_PATH", "")
    if not rocm_path:
        for candidate in [
            "/opt/rocm",
            os.path.expanduser("~/rocm"),
        ]:
            if os.path.isdir(candidate):
                rocm_path = candidate
                break

        therock = os.path.expanduser(
            "~/Documents/HomeBrew/git/Tools/therock-tarball/install"
        )
        if os.path.isdir(therock):
            rocm_path = therock

    if rocm_path:
        os.environ["ROCM_PATH"] = rocm_path
        lib_path = os.path.join(rocm_path, "lib")
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_path not in existing_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{existing_ld}"

    gfx = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if not gfx:
        try:
            result = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Name:") and "gfx" in stripped:
                    arch = stripped.split("gfx")[1].split()[0].strip()
                    major = arch[0]
                    minor = arch[1] if len(arch) > 1 else "0"
                    patch = arch[2] if len(arch) > 2 else "0"
                    gfx = f"{major}.{minor}.{patch}"
                    break
        except Exception:
            pass

    if gfx:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", gfx)


_setup_rocm_env()


def detect_device() -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            logger.info(f"ROCm GPU detected: {device_name} (HIP {torch.version.hip})")
            return "cuda:0"
        logger.info(f"CUDA GPU detected: {device_name} (CUDA {torch.version.cuda})")
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS (Metal) GPU detected")
        return "mps"
    logger.info("No GPU detected, using CPU")
    return "cpu"


def get_backend_name() -> str:
    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return "rocm"
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_attn_implementation(requested: str) -> str:
    fallback_chain = {
        "flash_attention_2": ["flash_attention_2", "sdpa", "eager"],
        "flex_attention": ["flex_attention", "sdpa", "eager"],
        "sdpa": ["sdpa", "eager"],
        "eager": ["eager"],
    }

    chain = fallback_chain.get(requested, ["sdpa", "eager"])

    for impl in chain:
        if impl == "flash_attention_2":
            try:
                import flash_attn

                return "flash_attention_2"
            except ImportError:
                logger.warning("flash-attn not installed, skipping flash_attention_2")
                continue
        if impl == "flex_attention":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                return impl
            continue
        if impl == "sdpa":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                return impl
            continue
        if impl == "eager":
            return "eager"

    return "eager"


def validate_rocm_training() -> bool:
    if not torch.cuda.is_available():
        logger.error("No GPU available for ROCm training")
        return False
    if not (hasattr(torch.version, "hip") and torch.version.hip is not None):
        logger.error(
            "ROCm training requested but PyTorch was not built with ROCm/HIP support"
        )
        return False
    try:
        logger.info(
            f"ROCm device: {torch.cuda.get_device_name(0)}, HIP {torch.version.hip}"
        )
        return True
    except Exception as e:
        logger.error(f"ROCm validation failed: {e}")
        return False
