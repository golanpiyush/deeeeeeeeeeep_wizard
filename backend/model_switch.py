"""
model_switch.py
=================
ONE PLACE to change which "Depth Anything V2" checkpoint the whole app uses.

Why this file exists:
    Before, the model size was buried in an environment variable
    (DEPTHWIZARD_MODEL_SIZE) that nobody on the team would remember to
    set. Now it's a plain Python constant at the top of this file —
    change one line, restart the server, done.

    The environment variable still works too (it OVERRIDES this file,
    so CI/demo-day scripting can still force a size without editing code),
    but for day-to-day dev, just edit ACTIVE_MODEL below.

HOW TO SWITCH MODELS:
    Change ACTIVE_MODEL to one of: "small", "base", "large"
    Then restart server.py. That's it.

WHICH ONE SHOULD YOU USE?
    "small"  -> fastest, lowest VRAM (~1GB), best for iterating/demoing on
                CPU or a laptop GPU. Recommended default.
    "base"   -> noticeably sharper depth edges, ~400MB heavier download,
                needs a decent GPU (4GB+ VRAM) to stay fast.
    "large"  -> best quality, but slow on CPU and needs 6GB+ VRAM to run
                smoothly. Use this only for the final polished demo run
                on Piyush's RTX 4060, not for everyday dev iteration.
"""

import os


# ---------------------------------------------------------------------------
# <<< CHANGE THIS LINE TO SWITCH MODELS >>>
# ---------------------------------------------------------------------------
ACTIVE_MODEL = "large"   # "small" | "base" | "large"

# ---------------------------------------------------------------------------
# Colorama Setup (cross-platform colored terminal output)
# ---------------------------------------------------------------------------

try:
    from colorama import init, Fore, Back, Style
    
    # Initialize Colorama for Windows compatibility
    init(autoreset=True)
    
    # Define color constants for easy use
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    
    # Bright/bold variants
    BRIGHT_RED = Style.BRIGHT + Fore.RED
    BRIGHT_GREEN = Style.BRIGHT + Fore.GREEN
    BRIGHT_YELLOW = Style.BRIGHT + Fore.YELLOW
    BRIGHT_BLUE = Style.BRIGHT + Fore.BLUE
    BRIGHT_MAGENTA = Style.BRIGHT + Fore.MAGENTA
    BRIGHT_CYAN = Style.BRIGHT + Fore.CYAN
    BRIGHT_WHITE = Style.BRIGHT + Fore.WHITE
    
    BOLD = Style.BRIGHT
    DIM = Style.DIM
    RESET_ALL = Style.RESET_ALL
    
    COLORAMA_AVAILABLE = True
    
except ImportError:
    # Fallback if Colorama is not installed
    RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
    BRIGHT_RED = BRIGHT_GREEN = BRIGHT_YELLOW = BRIGHT_BLUE = ""
    BRIGHT_MAGENTA = BRIGHT_CYAN = BRIGHT_WHITE = ""
    BOLD = DIM = RESET_ALL = ""
    COLORAMA_AVAILABLE = False


def cprint(text, color=WHITE, style=""):
    """Print colored text (safe fallback if Colorama not available)"""
    if COLORAMA_AVAILABLE:
        print(f"{style}{color}{text}{RESET_ALL}")
    else:
        print(text)



# ---------------------------------------------------------------------------

# Environment variable always wins if set — lets you override without
# touching code, e.g.:  DEPTHWIZARD_MODEL_SIZE=large python server.py
ACTIVE_MODEL = os.environ.get("DEPTHWIZARD_MODEL_SIZE", ACTIVE_MODEL)

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

# Rough download sizes + VRAM guidance, shown in the terminal view / README
# so the team knows what they're choosing before they wait on a download.
MODEL_INFO = {
    "small": {"download_mb": "~100 MB", "min_vram_gb": 1, "quality": "Good"},
    "base": {"download_mb": "~400 MB", "min_vram_gb": 4, "quality": "Better"},
    "large": {"download_mb": "~1.3 GB", "min_vram_gb": 6, "quality": "Best"},
}


def get_active_model():
    """Returns (size_name, hf_model_id, info_dict) for whatever is currently
    configured — the single source of truth every other file imports from."""
    if ACTIVE_MODEL not in MODEL_IDS:
        raise ValueError(
            f"Invalid ACTIVE_MODEL={ACTIVE_MODEL!r} in model_switch.py. "
            f"Must be one of {list(MODEL_IDS)}."
        )
    return ACTIVE_MODEL, MODEL_IDS[ACTIVE_MODEL], MODEL_INFO[ACTIVE_MODEL]


def print_model_info():
    """Pretty-print model configuration with colors"""
    size, model_id, info = get_active_model()
    
    # Color-code model size based on which one it is
    size_colors = {
        "small": GREEN,
        "base": YELLOW,
        "large": RED
    }
    size_color = size_colors.get(size, WHITE)
    
    # Color-code VRAM requirement
    vram = info['min_vram_gb']
    if vram <= 1:
        vram_color = GREEN
    elif vram <= 4:
        vram_color = YELLOW
    else:
        vram_color = RED
    
    # Color-code quality
    quality_colors = {
        "Good": GREEN,
        "Better": YELLOW,
        "Best": RED
    }
    quality_color = quality_colors.get(info['quality'], WHITE)
    
    # Print header
    cprint("=" * 70, CYAN, BOLD)
    cprint("ACTIVE MODEL CONFIGURATION", CYAN, BOLD)
    cprint("=" * 70, CYAN, BOLD)
    print()
    
    # Print details with colors
    cprint(f"  Active model:      ", BLUE, BOLD, end="")
    cprint(size.upper(), size_color, BOLD)
    
    cprint(f"  Hugging Face ID:   ", BLUE, BOLD, end="")
    cprint(model_id, CYAN)
    
    cprint(f"  Download size:     ", BLUE, BOLD, end="")
    cprint(info['download_mb'], YELLOW)
    
    cprint(f"  Min VRAM:          ", BLUE, BOLD, end="")
    cprint(f"{info['min_vram_gb']} GB", vram_color)
    
    cprint(f"  Quality:           ", BLUE, BOLD, end="")
    cprint(info['quality'], quality_color, BOLD)
    
    print()
    cprint("=" * 70, CYAN, BOLD)
    
    # Show available options
    print()
    cprint("AVAILABLE MODELS:", MAGENTA, BOLD)
    for model_size in ["small", "base", "large"]:
        model_info = MODEL_INFO[model_size]
        model_id_str = MODEL_IDS[model_size]
        
        if model_size == size:
            # Highlight the currently active model
            cprint(f"  ► {model_size.upper():<8} ", size_colors.get(model_size, WHITE), BOLD, end="")
            cprint(f"{model_id_str}", CYAN, BOLD)
            cprint(f"    {'':8} Download: {model_info['download_mb']}, "
                  f"VRAM: {model_info['min_vram_gb']}GB, Quality: {model_info['quality']}", 
                  DIM)
        else:
            cprint(f"    {model_size.upper():<8} ", WHITE, end="")
            cprint(f"{model_id_str}", WHITE)
            cprint(f"    {'':8} Download: {model_info['download_mb']}, "
                  f"VRAM: {model_info['min_vram_gb']}GB, Quality: {model_info['quality']}", 
                  DIM)


if __name__ == "__main__":
    print_model_info()
    
    # Show how to switch models
    print()
    cprint("HOW TO SWITCH MODELS:", YELLOW, BOLD)
    cprint("  1. Edit ACTIVE_MODEL in model_switch.py", WHITE)
    cprint("  2. Restart server.py", WHITE)
    cprint("  3. Or use environment variable:", WHITE)
    cprint("     DEPTHWIZARD_MODEL_SIZE=large python server.py", CYAN)
    
    print()
    cprint("TIP: 'small' is recommended for daily development.", GREEN)