import traceback, sys, os
from core.utils.local_search_manager import LocalSearchManager
from PyQt6.QtWidgets import QFileDialog, QMessageBox

def run_import_dialog(parent, search_mgr: LocalSearchManager):
    """
    Pops up a file dialog, then uses the given LocalSearchManager to
    import & index the chosen file. Reports success or a full traceback on error.
    """
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Import Content",
        "",
        "Supported Files (*.txt *.md *.markdown *.pdf);;All Files (*)"
    )
    if not file_path:
        return

    try:
        imported = search_mgr.import_document(file_path)
        QMessageBox.information(
            parent,
            "Import Successful",
            f"✅ {os.path.basename(imported)} indexed into search database!"
        )
    except Exception as e:
        # print full traceback so you can debug easily
        traceback.print_exc(file=sys.stderr)
        QMessageBox.critical(
            parent,
            "Import Failed",
            f"⚠️ Error during import:\n{e}"
        )
