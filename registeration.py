# ==================================================
# AUTO VENV + DEPENDENCY SETUP
# ==================================================
import os
import sys
import subprocess
import importlib.util
import platform

# --------------------------------------------------
# 0. Python version safety
# --------------------------------------------------
if sys.version_info >= (3, 12):
    raise RuntimeError(
        "Python 3.12 is NOT supported by some neuroimaging tools.\n"
        "Please use Python 3.10 or 3.11."
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

    print(f"[SETUP] Restarting using: {python_exe}")
    subprocess.check_call([python_exe] + sys.argv)
    sys.exit(0)

# --------------------------------------------------
# 2. Install required Python dependencies
# --------------------------------------------------
def install_dependencies():
    REQUIRED = {
        "nibabel": "nibabel",
        "matplotlib": "matplotlib",
        "nilearn": "nilearn",
    }

    missing = []
    for module, pkg in REQUIRED.items():
        if importlib.util.find_spec(module) is None:
            missing.append(pkg)

    if missing:
        print("[SETUP] Installing:", missing)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing
        )
    else:
        print("[SETUP] All Python dependencies installed.")

# --------------------------------------------------
# RUN SETUP
# --------------------------------------------------
ensure_venv()
install_dependencies()

# ==================================================
# SAFE IMPORTS
# ==================================================
import shutil
import nibabel as nib

# ==================================================
# 3. Check FSL installation
# ==================================================
def check_fsl():
    try:
        subprocess.check_output(["flirt", "-version"])
    except Exception:
        raise RuntimeError(
            "FSL is not installed or not in PATH.\n"
            "Install FSL and ensure 'flirt' command works in terminal."
        )

# ==================================================
# 4. Longitudinal Registration
# ==================================================
def longitudinal_flirt(moving, reference, out_img):
    """
    Rigid registration (6 DOF) of moving -> reference
    """

    check_fsl()

    cmd = [
        "flirt",
        "-in", moving,
        "-ref", reference,
        "-out", out_img,
        "-dof", "6",
        "-cost", "normmi",
        "-interp", "spline"
    ]

    print("[INFO] Running FLIRT registration...")
    subprocess.check_call(cmd)

    if not os.path.exists(out_img):
        raise FileNotFoundError("Registration failed: output image not found")

    print("[DONE] Registration completed.")
    print(f"[OUTPUT] Registered image: {out_img}")


# ==================================================
# 5. PUBLIC API
# ==================================================
def register_longitudinal(time2_image, time1_image, output_registered):
    longitudinal_flirt(
        moving=time2_image,
        reference=time1_image,
        out_img=output_registered
    )


# ==================================================
# 6. CLI ENTRY POINT
# ==================================================
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage:")
        print("  python run_longitudinal_registration.py")
        print("      moving.nii.gz reference.nii.gz output_registered.nii.gz")
        sys.exit(1)

    register_longitudinal(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3]
    )