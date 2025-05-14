# plugins/image_gen/image_settings_dialog.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox
from PyQt6.QtCore import Qt
import json
import os

CONFIG_PATH = "plugins/image_gen/image_gen_config.json"

class ImageGenSettingsDialog(QDialog):
    def __init__(self, parent=None, config_path=CONFIG_PATH):
        super().__init__(parent)
        self.setWindowTitle("Image Generation Settings")
        self.setMinimumWidth(400)
        self.config_path = config_path

        self.layout = QVBoxLayout()

        # ── Model Dropdown ──────────────────────────────
        self.layout.addWidget(QLabel("Model:"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
        "stabilityai/stable-diffusion-1-5",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/sdxl-base-1.0",
        "hakurei/waifu-diffusion",
        "Freepik/F-Lite",
        "andite/anything-v4.0",
        "Linaqruf/anything-v3.0",
        "cagliostrolab/animagine-xl",
        "DGSpitzer/animepastel-v2"
        ])

        self.layout.addWidget(self.model_dropdown)

        # ── Guidance Scale ──────────────────────────────
        self.layout.addWidget(QLabel("Guidance Scale (prompt strength):"))
        self.guidance_slider = QSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setMinimum(5)
        self.guidance_slider.setMaximum(20)
        self.guidance_slider.setTickInterval(1)
        self.guidance_slider.setValue(9)
        self.layout.addWidget(self.guidance_slider)
        self.guidance_label = QLabel("9")
        self.layout.addWidget(self.guidance_label)
        self.guidance_slider.valueChanged.connect(lambda v: self.guidance_label.setText(str(v)))

        # ── Inference Steps ─────────────────────────────
        self.layout.addWidget(QLabel("Inference Steps (detail level):"))
        self.steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.steps_slider.setMinimum(10)
        self.steps_slider.setMaximum(100)
        self.steps_slider.setTickInterval(5)
        self.steps_slider.setValue(50)
        self.layout.addWidget(self.steps_slider)
        self.steps_label = QLabel("50")
        self.layout.addWidget(self.steps_label)
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))

        # ── Resolution Presets ──────────────────────────
        self.layout.addWidget(QLabel("Image Resolution:"))
        self.resolution_dropdown = QComboBox()
        self.resolution_dropdown.addItems([
            "512x512", "768x768", "1024x1024", "1024x1536", "1536x1024"
        ])
        self.layout.addWidget(self.resolution_dropdown)

        # ── Save Button ────────────────────────────────
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_button)

        self.setLayout(self.layout)
        self.load_settings()

    def get_settings(self):
        width, height = map(int, self.resolution_dropdown.currentText().split("x"))
        return {
            "guidance_scale": self.guidance_slider.value(),
            "num_inference_steps": self.steps_slider.value(),
            "model_id": self.model_dropdown.currentText(),
            "width": width,
            "height": height
        }

    def save_settings(self):
        settings = self.get_settings()
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        self.accept()

    def load_settings(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)

                # Load sliders
                self.guidance_slider.setValue(settings.get("guidance_scale", 9))
                self.steps_slider.setValue(settings.get("num_inference_steps", 50))
                self.guidance_label.setText(str(settings.get("guidance_scale", 9)))
                self.steps_label.setText(str(settings.get("num_inference_steps", 50)))

                # Load resolution
                res = f'{settings.get("width", 512)}x{settings.get("height", 512)}'
                index = self.resolution_dropdown.findText(res)
                if index != -1:
                    self.resolution_dropdown.setCurrentIndex(index)

                # Load model
                model_id = settings.get("model_id", "stabilityai/stable-diffusion-2-1")
                index = self.model_dropdown.findText(model_id)
                if index != -1:
                    self.model_dropdown.setCurrentIndex(index)

            except Exception as e:
                print(f"[ImageGenSettingsDialog] Failed to load settings: {e}")
