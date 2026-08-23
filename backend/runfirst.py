"""
runfirst.py
============
DepthWizard — Fully Automated Team Onboarding & Setup Script
SIH26175 — Single-View Height Estimation & 3D Flythrough

Team:
    Soniya Singh   - Team Lead
    Piyush Golan
    Anushka Soni
    Shivam Jha
    Srishti Jain

WHAT THIS SCRIPT DOES (fully automatic, no manual venv steps needed):
    1. Explains what DepthWizard actually is, in plain language.
    2. Checks your Python version and warns about known issues.
    3. Creates a venv automatically if one doesn't exist yet.
    4. Re-launches itself INSIDE that venv automatically.
    5. Auto-detects your GPU/CUDA setup and picks the right torch build.
    6. Installs everything with a live progress bar, narrating each package.
    7. Explains the AI model download that happens on first server run.
    8. Runs a real end-to-end self-test so you know setup actually worked.

HOW TO RUN THIS (this is now the ONLY command anyone needs):

        python runfirst.py

    That's it. No manual venv creation, no manual activation. The script
    handles all of that itself.
"""

import subprocess
import sys
import os
import time
import platform
import venv as venv_module


# ===========================================================================
# TEST MODE
# ===========================================================================
# Flip this to True while YOU are testing/debugging this script itself.
# It skips the slow parts (real pip installs, real model-download wait,
# the "press enter" pauses) so you can run through the whole flow in a
# few seconds and check the narration/logic works before handing this
# off to the team. Leave it False for the actual teammate-facing version.

TEST_MODE = True


# ---------------------------------------------------------------------------
# Color support (stdlib only — no colorama dependency)
# ---------------------------------------------------------------------------
# Red    = failure / error / blocking warning
# Yellow = caution / heads-up / non-blocking warning
# Green  = success / good / all-clear
# Blue   = information / explanation / educational content
# Cyan   = special highlights / commands to run
# Magenta = team/contact information
#
# Windows cmd/PowerShell needs VT100 processing explicitly enabled before it
# will render ANSI codes instead of printing raw escape junk — we turn that
# on below. If it can't be enabled (very old Windows) or output isn't a real
# terminal (piped to a file), we fall back to no color rather than break.

def _supports_color():
    if not sys.stdout.isatty():
        return False
    if platform.system() == "Windows":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            if not kernel32.SetConsoleMode(handle, new_mode):
                return False
            return True
        except Exception:
            return False
    return True


_COLOR_ENABLED = _supports_color()

_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _colorize(text, code):
    if not _COLOR_ENABLED:
        return text
    return f"{code}{text}{_RESET}"


def red(text):
    return _colorize(text, _RED)


def green(text):
    return _colorize(text, _GREEN)


def yellow(text):
    return _colorize(text, _YELLOW)


def blue(text):
    return _colorize(text, _BLUE)


def cyan(text):
    return _colorize(text, _CYAN)


def magenta(text):
    return _colorize(text, _MAGENTA)


def bold(text):
    return _colorize(text, _BOLD)


# Dictionary mapping color names to color functions
COLOR_FUNCTIONS = {
    "blue": blue,
    "green": green,
    "red": red,
    "yellow": yellow,
    "cyan": cyan,
    "magenta": magenta,
}


# ---------------------------------------------------------------------------
# Pretty printing helpers (stdlib only — this script must never fail
# before pip even runs, so no external deps here)
# ---------------------------------------------------------------------------

def banner(text, color="blue"):
    """Display a prominent banner with colored borders"""
    line = "=" * 70
    color_func = COLOR_FUNCTIONS.get(color, blue)
    print(f"\n{color_func(line)}")
    print(f"{bold(color_func(text))}")
    print(f"{color_func(line)}")


def step(n, total, text, color="blue"):
    """Display a step header with progress indication"""
    color_func = COLOR_FUNCTIONS.get(color, blue)
    print(f"\n{bold(blue(f'[Step {n}/{total}]'))} {bold(color_func(text))}")
    time.sleep(0.5)


def pause(prompt=">> Press Enter to continue..."):
    """Pause execution until user presses Enter (skipped in TEST_MODE)"""
    if TEST_MODE:
        print(f"{yellow(prompt)} {blue('(auto-skipped — TEST_MODE=True)')}")
        return
    input(cyan(prompt))


# ---------------------------------------------------------------------------
# Live progress bar (stdlib only)
# ---------------------------------------------------------------------------

def print_progress_bar(current, total, prefix="", width=40):
    """Display a colored progress bar"""
    fraction = current / total if total else 1
    filled = int(width * fraction)
    bar_raw = "#" * filled + "-" * (width - filled)
    
    # Color the bar based on completion percentage
    if fraction < 0.33:
        bar = red(bar_raw)
    elif fraction < 0.66:
        bar = yellow(bar_raw)
    elif fraction < 1.0:
        bar = blue(bar_raw)
    else:
        bar = green(bar_raw)
    
    percent = fraction * 100
    sys.stdout.write(f"\r    {prefix} [{bar}] {percent:5.1f}%")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def fake_timed_progress(label, seconds):
    """Used for narrating steps that don't have real byte-level progress
    (e.g. waiting on a subprocess) — ticks a bar over an estimated duration."""
    ticks = 30
    for i in range(ticks + 1):
        print_progress_bar(i, ticks, prefix=label)
        time.sleep(0.05 if TEST_MODE else seconds / ticks)


# ---------------------------------------------------------------------------
# Step: Explain the project (with timed color explanation)
# ---------------------------------------------------------------------------

def explain_project():
    banner("DepthWizard — SIH26175", "cyan")
    print("\n" + magenta(bold("TEAM MEMBERS:")))
    print(magenta("    Soniya Singh (Lead), Piyush Golan, Anushka Soni, Shivam Jha, Srishti Jain"))
    
    time.sleep(1)
    
    print("\n" + blue(bold("WHAT WE'RE BUILDING:")))
    print(blue("    Upload ONE flat 2D photo -> get back an interactive 3D scene you can"))
    print(blue("    rotate, zoom, and fly through in a browser. No stereo camera, no LIDAR,"))
    print(blue("    just AI predicting depth from a single image."))
    
    time.sleep(1)
    
    print("\n" + cyan(bold("HOW IT WORKS (high level):")))
    print(cyan("    1. A pretrained AI model called 'Depth Anything V2' looks at your photo"))
    print(cyan("       and estimates how far away every pixel is (a 'depth map')."))
    time.sleep(0.8)
    print(cyan("    2. We turn that depth map + the original photo into a 3D point cloud"))
    print(cyan("       (thousands of colored dots placed in 3D space)."))
    time.sleep(0.8)
    print(cyan("    3. A webpage (using Three.js) renders that point cloud so you can drag"))
    print(cyan("       to rotate and scroll to zoom, live, right in the browser."))
    
    time.sleep(1)
    
    print("\n" + green(bold("THIS SCRIPT NOW DOES EVERYTHING AUTOMATICALLY:")))
    print(green("    - Creates your venv (you don't run 'python -m venv' yourself anymore)"))
    print(green("    - Activates it for you behind the scenes"))
    print(green("    - Detects your GPU and installs the matching torch build"))
    print(green("    - Installs all other packages with a live progress bar"))
    print(green("    - Runs a real self-test at the end"))
    
    time.sleep(1)
    
    print("\n" + yellow(bold("WHAT THIS SCRIPT DOES NOT DO:")))
    print(yellow("    - It does not touch the frontend (frontend/index.html) — that just"))
    print(yellow("      needs a browser, nothing to install there."))
    print(yellow("    - It does not require a GPU. It runs on CPU too, just slower."))
    
    print()
    pause(">> Press Enter once you've read this to continue setup...")


# ---------------------------------------------------------------------------
# Step: Python version check (with color-coded results)
# ---------------------------------------------------------------------------

def check_python_version():
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    
    print(blue(f"    You're running Python {version_str}"))
    time.sleep(0.5)

    if (major, minor) >= (3, 13):
        print(yellow("""
    !! WARNING: Python 3.13+ detected.

    Some ML packages don't ship prebuilt installers for 3.13 yet. This
    script installs newer package versions that DO support 3.13, but if
    installation still fails, the fix is switching to Python 3.11/3.12.
"""))
        time.sleep(1.5)
    elif (major, minor) < (3, 10):
        print(yellow(f"""
    !! WARNING: Python {version_str} may be too OLD for some packages here.
    Recommended: Python 3.10, 3.11, or 3.12.
"""))
        time.sleep(1.5)
    else:
        print(green("    Good — this version should work smoothly."))
        time.sleep(0.5)


# ---------------------------------------------------------------------------
# Step: Auto-create + auto-relaunch inside venv
# ---------------------------------------------------------------------------
#
# This is what makes "python runfirst.py" the ONLY command anyone needs.
# If we're not already running inside backend/venv, we create it (if
# missing) and then re-execute this same script using the venv's own
# python interpreter.

VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")


def _venv_python_path():
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def _is_running_in_our_venv():
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    return in_venv and os.path.abspath(sys.prefix) == os.path.abspath(VENV_DIR)


def ensure_venv_and_relaunch():
    if TEST_MODE:
        print(blue("    TEST_MODE=True — skipping real venv creation/relaunch,"))
        print(blue("    running self-test logic in the current interpreter instead."))
        time.sleep(0.5)
        return

    if _is_running_in_our_venv():
        print(green(f"    Already running inside the project venv: {sys.prefix}"))
        time.sleep(0.5)
        return

    if not os.path.exists(VENV_DIR):
        print(blue(f"    No venv found at {VENV_DIR} — creating one now..."))
        time.sleep(0.5)
        venv_module.EnvBuilder(with_pip=True).create(VENV_DIR)
        print(green("    Venv created."))
        time.sleep(0.5)
    else:
        print(green(f"    Found existing venv at {VENV_DIR}"))
        time.sleep(0.5)

    venv_python = _venv_python_path()
    print(cyan("    Switching into the venv's Python and re-running setup...\n"))
    time.sleep(1)
    result = subprocess.run([venv_python, os.path.abspath(__file__)])
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Step: GPU / CUDA auto-detection
# ---------------------------------------------------------------------------
#
# We can't import torch yet (not installed), so we detect NVIDIA GPU
# presence via `nvidia-smi` and pick a matching CUDA wheel index.
# Falls back to CPU-only wheels if nothing is detected — never blocks setup.

def detect_gpu_and_pick_torch_index():
    print(blue("    Checking for an NVIDIA GPU (via nvidia-smi)..."))
    time.sleep(1)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_line = result.stdout.strip().splitlines()[0]
            print(green(f"    GPU detected: {gpu_line}"))
            time.sleep(0.5)
            print(green("    Using CUDA 12.8 wheel index (cu128) — broad current-GPU support."))
            time.sleep(0.5)
            return "https://download.pytorch.org/whl/cu128", True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print(yellow("    No NVIDIA GPU detected (or nvidia-smi unavailable)."))
    time.sleep(0.5)
    print(yellow("    Installing CPU-only torch — the app still works, just slower."))
    time.sleep(0.5)
    return "https://download.pytorch.org/whl/cpu", False


# ---------------------------------------------------------------------------
# Step: Install requirements with live progress + narration per package
# ---------------------------------------------------------------------------

PACKAGE_GROUPS = [
    ("Core web/server libs", ["fastapi", "uvicorn", "pillow", "numpy", "opencv-python"], 30),
    ("PyTorch (large download)", ["torch==2.7.1", "torchvision==0.22.1"], 240),
    ("AI model libraries", ["transformers==4.46.3", "accelerate==1.1.1"], 45),
]


def install_requirements():
    torch_index, has_gpu = detect_gpu_and_pick_torch_index()

    if TEST_MODE:
        print(blue("\n    TEST_MODE=True — simulating install with fake progress bars,"))
        print(blue("    no real packages will be downloaded.\n"))
        time.sleep(0.5)
        for name, pkgs, _ in PACKAGE_GROUPS:
            print(cyan(f"    Installing: {name} ({', '.join(pkgs)})"))
            fake_timed_progress(name, 1)
            time.sleep(0.3)
        print(green("\n    (Simulated) all packages installed successfully."))
        time.sleep(0.5)
        return

    total_groups = len(PACKAGE_GROUPS)
    for i, (name, pkgs, est_seconds) in enumerate(PACKAGE_GROUPS, start=1):
        print(f"\n    {bold(blue(f'[{i}/{total_groups}]'))} {cyan(name)}")
        print(f"        {blue('Packages:')} {', '.join(pkgs)}")
        print(f"        {blue('Estimated time:')} ~{est_seconds}s {yellow('(depends on your internet speed)')}")
        time.sleep(0.5)

        cmd = [sys.executable, "-m", "pip", "install"]
        if "torch" in pkgs[0]:
            cmd += pkgs + ["--index-url", torch_index]
        else:
            cmd += pkgs

        start = time.time()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        tick = 0
        while process.poll() is None:
            elapsed = time.time() - start
            fraction = min(elapsed / est_seconds, 0.97)
            print_progress_bar(int(fraction * 100), 100, prefix=name[:24].ljust(24))
            time.sleep(0.2)
            tick += 1

        print_progress_bar(100, 100, prefix=name[:24].ljust(24))
        output = process.stdout.read() if process.stdout else ""

        if process.returncode != 0:
            print(red(f"\n    !! FAILED installing {name} after {time.time()-start:.0f}s"))
            print(red("    ---- pip output (last 40 lines) ----"))
            for line in output.strip().splitlines()[-40:]:
                print(red(f"    {line}"))
            print(red("    -------------------------------------"))
            print(yellow("""
    Common fixes:
        1. Python version issue — try Python 3.11/3.12 (see version check above).
        2. Internet dropped mid-download — just re-run: python runfirst.py
        3. Paste the error above to your team lead or Piyush if unsure.
"""))
            time.sleep(2)
            sys.exit(1)

        print(green(f"    Done in {time.time()-start:.0f}s."))
        time.sleep(0.5)

    status = green("YES") if has_gpu else yellow("NO (CPU mode)")
    print(f"\n{green('    All packages installed successfully.')} {blue('GPU acceleration:')} {status}")
    time.sleep(1)


# ---------------------------------------------------------------------------
# Step: Explain the model download (with blue informational coloring)
# ---------------------------------------------------------------------------

def explain_model_download():
    print(blue("""
    This script does NOT download the AI model itself — that happens
    automatically the FIRST time you run:

    """), end="")
    print(cyan(bold("        python server.py")))
    print(blue("""
    What to expect on that first run:
        - Model: Depth Anything V2 (Small)
        - Download size: roughly 100-200 MB
        - Estimated time: 1-5 minutes on decent wifi
        - Downloads ONCE and is cached locally — every run after is instant.

    On slow/limited internet, do this download BEFORE you need to demo,
    not five minutes before a meeting.
"""))
    time.sleep(2)


# ---------------------------------------------------------------------------
# Step: Self-test (with color-coded results)
# ---------------------------------------------------------------------------

def run_self_test():
    try:
        from generate_sample_depth import fake_depth_from_brightness
        from pointcloud import image_and_depth_to_pointcloud
        from PIL import Image
        import numpy as np

        print(blue("    Creating test image..."))
        time.sleep(0.5)
        test_array = np.random.default_rng(0).integers(
            0, 255, size=(64, 64, 3), dtype=np.uint8
        )
        test_image = Image.fromarray(test_array, mode="RGB")

        print(blue("    Generating depth map..."))
        time.sleep(0.5)
        depth = fake_depth_from_brightness(test_image)

        print(blue("    Creating point cloud..."))
        time.sleep(0.5)
        pointcloud = image_and_depth_to_pointcloud(test_image, depth, max_points=1000)

        assert pointcloud["count"] == 1000
        assert len(pointcloud["points"]) == 1000
        assert len(pointcloud["colors"]) == 1000

        print(green("    Self-test PASSED — core pipeline (image -> depth -> point cloud) works."))
        time.sleep(1)
    except ModuleNotFoundError as exc:
        if TEST_MODE:
            print(yellow(f"    Self-test SKIPPED in TEST_MODE (missing project module: {exc})."))
            print(yellow("    This is expected if you're running runfirst.py standalone, outside"))
            print(yellow("    the real backend/ folder — the automation logic above is what matters."))
            time.sleep(1)
        else:
            print(red(f"\n    !! Self-test FAILED: {exc}\n    Share this error with your team."))
            time.sleep(2)
            sys.exit(1)
    except Exception as exc:
        print(red(f"""
    !! Self-test FAILED: {exc}

    Package installation technically succeeded, but something in the
    pipeline code isn't working. Share this error with your team.
"""))
        time.sleep(2)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step: Final instructions (with color-coded sections)
# ---------------------------------------------------------------------------

def print_next_steps():
    title = "SETUP COMPLETE" + ("  (TEST_MODE — nothing was actually installed)" if TEST_MODE else "")
    print(f"\n{bold(green('=' * 70))}")
    print(f"{bold(green(title))}")
    print(f"{bold(green('=' * 70))}")
    time.sleep(1)
    
    print(cyan("""
You're ready. Here's what to do next:

    1. Start the backend server (in this same terminal):
"""))
    time.sleep(0.5)
    print(cyan(bold("           python server.py")))
    time.sleep(0.5)
    print(blue("""
       First run downloads the AI model (see above) — be patient.

    2. In a SEPARATE terminal, start the frontend:
"""))
    time.sleep(0.5)
    print(cyan(bold("           cd ../frontend")))
    print(cyan(bold("           python -m http.server 5500")))
    time.sleep(0.5)
    
    print(blue("""
    3. Open your browser to:
"""))
    time.sleep(0.5)
    print(cyan(bold("           http://localhost:5500")))
    time.sleep(0.5)
    
    print(blue("""
    4. Upload a photo, click "Generate 3D View", and drag to rotate.

"""))
    
    print(yellow(bold("WANT TO SKIP THE AI MODEL WHILE DEVELOPING (faster iteration)?")))
    time.sleep(0.5)
    print(yellow("    Windows (PowerShell):   $env:USE_FAKE_DEPTH=\"1\"; python server.py"))
    print(yellow("    Mac/Linux:               USE_FAKE_DEPTH=1 python server.py"))
    time.sleep(1)
    
    print(magenta(bold("""
QUESTIONS OR STUCK?
    Ping the team:
        Soniya Singh (Team Lead)
        Piyush Golan
        Anushka Soni
        Shivam Jha
        Srishti Jain
""")))
    time.sleep(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOTAL_STEPS = 7


def main():
    if TEST_MODE:
        banner("RUNNING IN TEST_MODE — nothing real will be installed/downloaded", "red")

    explain_project()

    step(1, TOTAL_STEPS, "Checking your Python version...")
    check_python_version()

    step(2, TOTAL_STEPS, "Setting up your virtual environment (fully automatic)...", "cyan")
    ensure_venv_and_relaunch()

    step(3, TOTAL_STEPS, "Detecting GPU + installing packages (live progress)...", "cyan")
    install_requirements()

    step(4, TOTAL_STEPS, "About the AI model download...", "cyan")
    explain_model_download()

    step(5, TOTAL_STEPS, "Running self-test...", "cyan")
    run_self_test()

    step(6, TOTAL_STEPS, "Done — printing next steps...", "green")
    print_next_steps()


if __name__ == "__main__":
    main()