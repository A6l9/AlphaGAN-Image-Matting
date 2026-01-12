COLORS = {
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "white":   "\033[37m",
    "reset":   "\033[0m",
}


def color(text: str, name: str) -> str:
    """Colorizes the terminal output

    Args:
        text (str): Terminal output
        name (str): The name of the color

    Returns:
        str: The colorized teriminal output
    """
    return f"{COLORS.get(name, COLORS['reset'])}{text}{COLORS['reset']}"
