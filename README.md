# LesionTracking

A tool for tracking and analyzing multiple sclerosis lesions between longitudinal MRI scans.

## 🍎 macOS Installation Guide

Follow these steps to set up and run the project on macOS.

### Prerequisites

1.  **Terminal**: Open the **Terminal** app (Command + Space, type "Terminal").
2.  **Homebrew** (Optional but recommended): If you don't have python or git installed, Homebrew makes it easy.
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
3.  **Python & Git**:
    ```bash
    brew install python git
    ```

### Installation

1.  **Clone the Repository**
    Download the code to your machine.
    ```bash
    git clone https://github.com/Menna1812/LesionTracking.git
    cd LesionTracking
    ```

2.  **Create a Virtual Environment**
    It's best practice to isolate project dependencies.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the Environment**
    ```bash
    source venv/bin/activate
    ```
    (You should see `(venv)` appear in your terminal prompt).

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Alternatively, install the package in editable mode:*
    ```bash
    pip install -e .
    ```

### Usage

You need two NIfTI files (`.nii.gz`) to run the tracking: a **baseline** mask and a **follow-up** mask.

Run the tool from the project root:

```bash
python -m lesion_tracking.main <path_to_baseline.nii.gz> <path_to_followup.nii.gz>
```

**Example:**
```bash
python -m lesion_tracking.main data/patient1_baseline.nii.gz data/patient1_followup.nii.gz
```

### Output

The tool will generate CSV summary tables in the `output/` directory:
- `{PatientID} Baseline summary table.csv`
- `{PatientID} Follow up summary table.csv`