"""figures subpackage. See individual modules for CLI usage."""

import matplotlib

# Embed fonts as TrueType (Type 42): NeurIPS rejects PDFs with Type 3 fonts.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
