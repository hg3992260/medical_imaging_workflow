import argparse
import importlib
import json
import os
import sys


class _FakeButton:
    def __init__(self, label: str, role: int):
        self.label = label
        self.role = role


class _CaptureMessageBox:
    Warning = 2
    NoButton = 0
    AcceptRole = 0
    RejectRole = 1
    last_box = None
    next_click = "accept"

    def __init__(self, parent=None):
        self.parent = parent
        self.icon = None
        self.window_title = ""
        self.text = ""
        self.informative_text = ""
        self.standard_buttons = None
        self.default_button = None
        self.buttons = []
        self._clicked = None
        type(self).last_box = self

    @classmethod
    def information(cls, parent, title: str, text: str):
        cls.last_box = {
            "kind": "information",
            "title": title,
            "text": text,
        }
        return 0

    def setIcon(self, icon):
        self.icon = icon

    def setWindowTitle(self, title: str):
        self.window_title = title

    def setText(self, text: str):
        self.text = text

    def setInformativeText(self, text: str):
        self.informative_text = text

    def setStandardButtons(self, buttons):
        self.standard_buttons = buttons

    def addButton(self, label: str, role: int):
        btn = _FakeButton(label, role)
        self.buttons.append(btn)
        return btn

    def setDefaultButton(self, button):
        self.default_button = button

    def exec_(self):
        if type(self).next_click == "accept":
            for btn in self.buttons:
                if btn.label == "覆盖全部项目":
                    self._clicked = btn
                    return 0
        for btn in self.buttons:
            if btn.label == "取消":
                self._clicked = btn
                return 0
        self._clicked = None
        return 0

    def clickedButton(self):
        return self._clicked


def _load_dist_module(dist_dir: str):
    internal_dir = os.path.join(dist_dir, "_internal")
    sys.path.insert(0, internal_dir)
    return importlib.import_module("ui.widgets.template_wizard_dialog")


def _build_dummy(name: str, selected_pdf_path: str):
    class _DummyDialog:
        def __init__(self):
            self.selected_template_name = name
            self.selected_pdf_path = selected_pdf_path
            self.accepted = False

        def accept(self):
            self.accepted = True

    return _DummyDialog()


def verify_dialog(dist_dir: str):
    module = _load_dist_module(dist_dir)
    original_box = module.QMessageBox
    module.QMessageBox = _CaptureMessageBox
    try:
        dummy_accept = _build_dummy("GlobalTemplate", r"F:\fake\template.pdf")
        _CaptureMessageBox.next_click = "accept"
        module.TemplateWizardDialog._accept(dummy_accept)
        box = _CaptureMessageBox.last_box
        assert dummy_accept.accepted is True
        assert isinstance(box, _CaptureMessageBox)
        assert box.window_title == "确认应用全局模板"
        assert box.text == "你正在覆盖全部项目的模板配置：GlobalTemplate"
        assert "这会覆盖所有项目当前使用的模板配置。" in box.informative_text
        assert "同时会替换历史项目留下的 template_pdf 风格记录，并写入当前模板内容。" in box.informative_text
        assert [btn.label for btn in box.buttons] == ["覆盖全部项目", "取消"]
        assert box.default_button.label == "取消"

        dummy_cancel = _build_dummy("GlobalTemplate", "")
        _CaptureMessageBox.next_click = "cancel"
        module.TemplateWizardDialog._accept(dummy_cancel)
        box_cancel = _CaptureMessageBox.last_box
        assert dummy_cancel.accepted is False
        assert "这会覆盖所有项目当前使用的模板配置。" in box_cancel.informative_text
        assert "template_pdf" not in box_cancel.informative_text
        return {
            "ok": True,
            "window_title": box.window_title,
            "text": box.text,
            "informative_text": box.informative_text,
            "buttons": [btn.label for btn in box.buttons],
            "default_button": box.default_button.label,
        }
    finally:
        module.QMessageBox = original_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dist",
            "MedicalImagingWorkflow_v41_Final",
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = verify_dialog(args.dist_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print("PASS")
        print(result["window_title"])
        print(result["text"])
        print(result["informative_text"])
        print(" / ".join(result["buttons"]))
        print(result["default_button"])


if __name__ == "__main__":
    main()
