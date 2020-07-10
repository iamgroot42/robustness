from colorama import Fore, Style


def log_string(x, ttype="log"):
	color_mapping = {"train": Fore.YELLOW, "val": Fore.GREEN}
	return color_mapping.get(ttype, Fore.MAGENTA) + x + Style.RESET_ALL


def log_statement(x, ttype="log"):
	print(log_string(x, ttype))
