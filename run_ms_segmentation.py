# ==================================================
# AUTO VENV + DEPENDENCY SETUP
# ==================================================
import os
import sys
import subprocess
import importlib.util
import platform

# --------------------------------------------------
# 0. Python version safety check
# --------------------------------------------------
if sys.version_info >= (3, 12):
    raise RuntimeError(
        "Python 3.12 is NOT supported by PyTorch / nnU-Net.\n"
        "Please install Python 3.10 or 3.11 (64-bit CPython)."
    )

# --------------------------------------------------
# 1. Ensure virtual environment
# --------------------------------------------------
def ensure_venv(venv_dir=".venv311"):
    if sys.prefix != sys.base_prefix:
        return

    print("[SETUP] No virtual environment detected.")

    if not os.path.exists(venv_dir):
        print("[SETUP] Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    if os.name == "nt":
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(python_exe):
        raise RuntimeError("Virtual environment python not found.")

    print(f"[SETUP] Restarting using: {python_exe}")
    subprocess.check_call([python_exe] + sys.argv)
    sys.exit(0)

# --------------------------------------------------
# 2. Install PyTorch (CPU first, CUDA optional)
# --------------------------------------------------
def install_torch():
    try:
        import torch
        print("[SETUP] Torch already installed.")
        return
    except ImportError:
        pass

    system = platform.system().lower()

    print("[SETUP] Installing PyTorch (CPU baseline)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio"
    ])

    # ---- Optional CUDA upgrade (Windows / Linux only) ----
    if system in ("windows", "linux"):
        try:
            import torch
            if torch.cuda.is_available():
                print("[SETUP] CUDA detected – upgrading to CUDA build")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--upgrade",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ])
        except Exception:
            print("[WARN] CUDA upgrade skipped")

# --------------------------------------------------
# 3. Install remaining dependencies
# --------------------------------------------------
def install_dependencies():
    REQUIRED = {
        "nnunetv2": "nnunetv2",
        "nibabel": "nibabel",
        "numpy": "numpy",
        "SimpleITK": "SimpleITK",
        "yaml": "pyyaml"
    }

    missing = []
    for module, pkg in REQUIRED.items():
        if importlib.util.find_spec(module) is None:
            missing.append(pkg)

    if missing:
        print("[SETUP] Installing missing packages:", missing)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing
        )
    else:
        print("[SETUP] All required packages installed.")

# --------------------------------------------------
# RUN SETUP
# --------------------------------------------------
ensure_venv()
install_torch()
install_dependencies()

# ==================================================
# SAFE TO IMPORT HEAVY LIBRARIES
# ==================================================
import shutil
import json
import torch

# ==================================================
# 4. Setup nnU-Net environment
# ==================================================
def setup_nnunet_env(base_dir):
    raw = os.path.join(base_dir, "nnUNet_raw")
    pre = os.path.join(base_dir, "nnUNet_preprocessed")
    res = os.path.join(base_dir, "nnUNet_results")

    os.environ["nnUNet_raw"] = raw
    os.environ["nnUNet_preprocessed"] = pre
    os.environ["nnUNet_results"] = res

    os.makedirs(raw, exist_ok=True)
    os.makedirs(pre, exist_ok=True)
    os.makedirs(res, exist_ok=True)

# ==================================================
# 5. Register pretrained model
# ==================================================
def register_model(model_src, results_dir, dataset_name, trainer_name):
    auto_trainer = trainer_name + "__nnUNetPlans__3d_fullres"
    src = os.path.join(model_src, trainer_name)
    dst = os.path.join(results_dir, dataset_name, auto_trainer)

    if not os.path.exists(dst):
        print("[SETUP] Registering pretrained model...")
        shutil.copytree(src, dst)
    else:
        print("[SETUP] Pretrained model already registered.")

# ==================================================
# 6. Prepare input
# ==================================================
def prepare_input(flair_path, dataset_name):
    imagesTs = os.path.join(
        os.environ["nnUNet_raw"],
        dataset_name,
        "imagesTs"
    )
    os.makedirs(imagesTs, exist_ok=True)

    target = os.path.join(imagesTs, "case_0000.nii.gz")
    shutil.copy(flair_path, target)

# ==================================================
# 7. Run inference
# ==================================================
def run_inference(cfg):
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running on {device.upper()}")

    nnunet_exec = os.path.join(
        os.path.dirname(sys.executable),
        "nnUNetv2_predict.exe" if os.name == "nt" else "nnUNetv2_predict"
    )

    cmd = [
        nnunet_exec,
        "-d", cfg["nnunet"]["dataset_name"],
        "-i", os.path.join(
            os.environ["nnUNet_raw"],
            cfg["nnunet"]["dataset_name"],
            "imagesTs"
        ),
        "-o", output_dir,
        "-tr", cfg["nnunet"]["trainer"],
        "-c", cfg["nnunet"]["configuration"],
        "-device", device
    ]

    subprocess.check_call(cmd)

# ==================================================
# 8. PUBLIC API
# ==================================================
def segment_ms_lesions(flair_image_path, output_mask_path):
    with open("config.json") as f:
        cfg = json.load(f)

    setup_nnunet_env(cfg["paths"]["runtime_dir"])

    register_model(
        model_src=cfg["paths"]["model_dir"],
        results_dir=os.environ["nnUNet_results"],
        dataset_name=cfg["nnunet"]["dataset_name"],
        trainer_name=cfg["nnunet"]["trainer"]
    )

    prepare_input(flair_image_path, cfg["nnunet"]["dataset_name"])
    run_inference(cfg)

    pred = os.path.join(cfg["paths"]["output_dir"], "case.nii.gz")
    if not os.path.exists(pred):
        raise FileNotFoundError("Inference failed: output not found")

    shutil.move(pred, output_mask_path)
    print(f"[DONE] Lesion mask saved to {output_mask_path}")

# ==================================================
# 9. CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python run_ms_segmentation.py input_flair.nii.gz output_mask.nii.gz")
        sys.exit(1)

    segment_ms_lesions(sys.argv[1], sys.argv[2])
