# -*- mode: python ; coding: utf-8 -*-


import os
import sys
import sysconfig
import torch_geometric

# Add project root to path to find scheduler
project_root = os.path.dirname(os.path.abspath(SPEC))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

torch_geometric_path = os.path.dirname(torch_geometric.__file__)
scheduler_path = os.path.join(project_root, 'scheduler')

# Find _multiprocessing.so
dynload_dir = os.path.join(sysconfig.get_path('stdlib'), 'lib-dynload')
multiprocessing_so = os.path.join(dynload_dir, '_multiprocessing.cpython-310-darwin.so')

a = Analysis(
    ['scripts/eval_heuristics_multi_cases.py'],
    pathex=[project_root],
    binaries=[(multiprocessing_so, '.')] if os.path.exists(multiprocessing_so) else [],
    datas=[
        (torch_geometric_path, 'torch_geometric'),
        (scheduler_path, 'scheduler'),
    ],
    hiddenimports=['scheduler', '_multiprocessing', 'multiprocessing.connection'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='eval_heuristics',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
