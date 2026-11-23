# PyInstaller recipe for the BATMAN desktop app (main.py, PySide6).
#
# Build from the project root (Linux, inside the project's virtualenv):
#     pyinstaller --noconfirm packaging/batman.spec
# The result is dist/BATMAN/, which packaging/build_appimage.sh wraps into an
# AppImage.
#
# src/config.py resolves every path from the location of the src package. In
# a PyInstaller build that is the bundle's _internal folder, so the data files
# below are added at the same relative paths they have in the repository.

from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(ROOT / "data" / "elliptic_plus"), "data/elliptic_plus"),   # precomputed scores (Git LFS)
    (str(ROOT / "outputs" / "evaluation"), "outputs/evaluation"),   # metrics shown in the app
    (str(ROOT / "outputs" / "graphs"), "outputs/graphs"),
    (str(ROOT / "src" / "resources" / "fonts"), "src/resources/fonts"),
    (str(ROOT / "resources" / "images"), "resources/images"),
]

# The desktop app never imports these; leaving them out keeps the bundle smaller.
excludes = ["streamlit", "pytest", "IPython", "jupyter", "notebook", "tkinter"]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BATMAN",
    console=False,
    strip=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="BATMAN")
