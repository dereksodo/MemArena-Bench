"""Allow running MASim as a module: python -m MASim"""

from memarena.runtime import configure_live_output
from MASim.pipeline.cli import main

configure_live_output()
main()
