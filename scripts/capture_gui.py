import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from src.desktop_gui.main_window import MainWindow
from src.desktop_gui.utils.theme_setup import setup_theme

def capture():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    setup_theme(app)
    window = MainWindow()
    window.resize(1400, 850)
    window.show()
    app.processEvents()

    os.makedirs('outputs/reference_screenshots', exist_ok=True)

    for idx, page_name in enumerate(window.sidebar.NAV_ITEMS):
        window.stacked.setCurrentIndex(idx)
        window.sidebar.list_widget.setCurrentRow(idx)
        app.processEvents()
        clean_name = page_name.lower().replace(" ", "_")
        window.grab().save(f'outputs/reference_screenshots/page_{clean_name}.png')

    print("Desktop GUI screenshots saved to outputs/reference_screenshots/")
    app.quit()

if __name__ == "__main__":
    capture()
