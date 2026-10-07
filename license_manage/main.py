import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
try:
    from .ui import LicenseManager
    from .storage import init_db
except ImportError:
    try:
        from license_manage.ui import LicenseManager
        from license_manage.storage import init_db
    except ImportError:
        from ui import LicenseManager
        from storage import init_db

def main():
    init_db()
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("rsna.medical.imaging.workflow.licensemanager.1.0")
        except Exception:
            pass
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    else:
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

    app = QApplication(sys.argv)
    try:
        from PyQt5.QtGui import QIcon

        icon_candidates = []
        if getattr(sys, "frozen", False):
            base_dir = os.path.dirname(sys.executable)
            internal_dir = getattr(sys, "_MEIPASS", os.path.join(base_dir, "_internal"))
            icon_candidates.extend(
                [
                    os.path.join(base_dir, "_internal", "assets", "icon.ico"),
                    os.path.join(internal_dir, "assets", "icon.ico"),
                    os.path.join(base_dir, "assets", "icon.ico"),
                ]
            )
        icon_candidates.extend(
            [
                os.path.join(os.getcwd(), "assets", "icon.ico"),
                os.path.join(os.path.dirname(__file__), "assets", "icon.ico"),
            ]
        )
        for p in icon_candidates:
            if p and os.path.exists(p):
                app.setWindowIcon(QIcon(p))
                break
    except Exception:
        pass

    w = LicenseManager()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
