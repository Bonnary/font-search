# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:\\Code\\Python\\font-search\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('D:\\Code\\Python\\font-search\\assets', 'assets')],
    hiddenimports=['cv2', 'skimage.feature', 'skimage.filters', 'skimage.metrics', 'skimage.transform', 'fonttools', 'fonttools.ttLib'],
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
    [],
    exclude_binaries=True,
    name='FontSearch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['D:\\Code\\Python\\font-search\\assets\\logo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FontSearch',
)
