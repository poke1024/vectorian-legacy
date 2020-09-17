print("importing cpp core... ", flush=True, end="")
import cppimport.import_hook
from .vcore import *
init_pyarrow()
print("done.", flush=True)
