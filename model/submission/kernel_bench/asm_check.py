"""¿Vectoriza LLVM el producto punto del kernel actual? Cuenta instrucciones AVX en el asm generado."""
import re, sys
from pathlib import Path
import numpy as np
import numba
from numba import njit
import llvmlite.binding as llvm

CH = Path(r"C:\Users\Joaquin\Desktop\UM\semestre 8\Machine learning\proyecto final\wnn_connectome_starterpack\wnn_connectome_starterpack\challenge")
sys.path.insert(0, str(CH / "submissions" / "ens3_A_B_Cswa"))
import solution as sol

print("host cpu:", llvm.get_host_cpu_name(), " features AVX2:", "avx2" in llvm.get_host_cpu_features().flatten())
sol.PredictionModel()
for name, fn in (("_gru_step", sol._gru_step), ("step_row", sol.step_row)):
    for sig, asm in fn.inspect_asm().items():
        cnt = {}
        for ins in ("vfmadd", "vmulps", "vaddps", "vmulss", "vaddss", "vfmadd231ss", "vfmadd231ps", "ymm", "xmm", "vcvtdq2ps", "vpmovsxbd"):
            cnt[ins] = len(re.findall(r"\b" + ins, asm))
        print(f"{name}: {len(asm.splitlines())} líneas asm; ", ", ".join(f"{k}={v}" for k, v in cnt.items()))
