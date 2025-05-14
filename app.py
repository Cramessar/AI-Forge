import sys
import json
import re
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QSizePolicy,
    QDialog,
    QSpinBox,
    QTabWidget,
    QListWidgetItem,
    QCheckBox,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
import webbrowser
from model_loader import ModelLoader
import markdown2
from pygments import highlight
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.formatters import HtmlFormatter
from PyQt6.QtWebEngineCore import QWebEnginePage
from db import init_db
from hardware_profile import get_system_profile, get_tuned_generation_settings
from prompt_tools import load_templates, apply_template, chain_prompts, update_template_selector_state, load_prompt_template, save_prompt_template, chain_prompts
from local_hf_runner import HFRunner, HF_MODEL_MAP
from core.plugin_loader import load_plugins
from plugins.image_gen.settings_dialog import ImageGenSettingsDialog
from core.utils.local_search_manager import LocalSearchManager
from core.utils.file_importer import run_import_dialog



class ExternalLinkPage(QWebEnginePage):
    def acceptNavigationRequest(self, url, nav_type, is_main_frame):
        if nav_type == QWebEnginePage.NavigationType.NavigationTypeLinkClicked:
            webbrowser.open(url.toString())
            return False
        return super().acceptNavigationRequest(url, nav_type, is_main_frame)


class GenerationThread(QThread):
    result_ready = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, model_loader, prompt):
        super().__init__()
        self.model_loader = model_loader
        self.prompt = prompt

    def run(self):
        try:
            backend = self.model_loader.config["default_model"]["type"]
            if backend == "ollama":
                for chunk in self.model_loader.generate_with_ollama_stream(self.prompt):
                    self.result_ready.emit(chunk)
                self.finished.emit(self.prompt)
            else:
                self.result_ready.emit(
                    "[Streaming only supported for Ollama right now]"
                )
                self.finished.emit(self.prompt)
        except Exception as e:
            self.result_ready.emit(f"[Error] {str(e)}")
            self.finished.emit(self.prompt)




class SettingsDialog(QDialog):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.config = config
        self.resize(400, 200)

        tabs = QTabWidget()

        # === GENERAL TAB ===
        general_tab = QWidget()
        general_layout = QVBoxLayout()

        # === PERFORMANCE TAB ===
        performance_tab = QWidget()
        performance_layout = QVBoxLayout()

        backend_label = QLabel("Execution Backend")
        backend_label.setToolTip(
            "Select CPU, GPU, or let the app decide automatically."
        )

        self.backend_selector = QComboBox()
        self.backend_selector.addItems(["auto", "cpu", "gpu"])
        self.backend_selector.setCurrentText(
            self.config.get("performance", {}).get("backend", "auto")
        )

        performance_layout.addWidget(backend_label)
        performance_layout.addWidget(self.backend_selector)

        self.benchmark_button = QPushButton("Run Benchmark")
        self.benchmark_button.clicked.connect(self.run_benchmark)
        performance_layout.addWidget(self.benchmark_button)

        last_benchmark = self.config.get("performance", {}).get("last_benchmark", {})
        cpu_time = last_benchmark.get("cpu", "Not run")
        gpu_time = last_benchmark.get("gpu", "Not run")
        self.benchmark_result = QLabel(
            f"Last CPU: {cpu_time} sec | GPU: {gpu_time} sec"
        )
        performance_layout.addWidget(self.benchmark_result)

        performance_tab.setLayout(performance_layout)
        tabs.addTab(performance_tab, "Performance")

        # temp settings
        temp_row = QHBoxLayout()
        temp_label = QLabel("Temperature")
        temp_label.setToolTip(
            "Controls randomness. Lower = more focused, higher = more creative."
        )
        self.temp_box = QSpinBox()
        self.temp_box.setRange(0, 100)
        self.temp_box.setValue(
            int(config.get("generation", {}).get("temperature", 0.7) * 100)
        )
        temp_row.addWidget(temp_label)
        temp_row.addWidget(self.temp_box)

        # token settings
        token_row = QHBoxLayout()
        token_label = QLabel("Max Tokens")
        token_label.setToolTip(
            "Maximum number of tokens (words/pieces) the model can generate."
        )
        self.token_box = QSpinBox()
        self.token_box.setRange(32, 2048)
        self.token_box.setValue(config.get("generation", {}).get("max_tokens", 512))
        token_row.addWidget(token_label)
        token_row.addWidget(self.token_box)

        general_layout.addLayout(temp_row)
        general_layout.addLayout(token_row)
        general_tab.setLayout(general_layout)

        tabs.addTab(general_tab, "General")

        # === MAIN DIALOG ===
        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)

        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        main_layout.addWidget(self.save_button)
        main_layout.addWidget(self.reset_button)

        self.setLayout(main_layout)

    def save_settings(self):
        self.config.setdefault("generation", {})
        self.config["generation"]["temperature"] = self.temp_box.value() / 100
        self.config["generation"]["max_tokens"] = self.token_box.value()
        self.config.setdefault("performance", {})
        self.config["performance"]["backend"] = self.backend_selector.currentText()
        self.accept()

    def reset_to_defaults(self):
        self.temp_box.setValue(70)  # default 0.7
        self.token_box.setValue(512)  # default 512 tokens
        QMessageBox.information(
            self, "Settings Reset", "Settings have been reset to their default values."
        )

    def run_benchmark(self):
        sample_prompt = "Tell me a fantasy story about a lost sword in a cursed forest."

        self.benchmark_button.setEnabled(False)
        self.benchmark_button.setText("🔄 Running Benchmark...")
        QApplication.processEvents()

        from model_loader import ModelLoader

        loader = ModelLoader()
        times = loader.run_performance_test()

        cpu_time = times.get("cpu", "Error")
        gpu_time = times.get("gpu", "Error")

        # determine best backend based on timing
        if isinstance(cpu_time, (int, float)) and isinstance(gpu_time, (int, float)):
            best = "gpu" if gpu_time < cpu_time else "cpu"
        elif isinstance(gpu_time, (int, float)):
            best = "gpu"
        else:
            best = "cpu"

        self.config["performance"]["last_benchmark"] = times
        self.benchmark_result.setText(f"Last CPU: {cpu_time} sec | GPU: {gpu_time} sec")

        reply = QMessageBox.question(
            self,
            "Apply Best Performance Setting?",
            f"Benchmark complete.\n\n"
            f"CPU: {cpu_time} sec\nGPU: {gpu_time} sec\n\n"
            f"Use {best.upper()} for best performance?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.config["performance"]["backend"] = best
            from hardware_profile import get_system_profile, get_tuned_generation_settings
            profile = get_system_profile()
            recommended = get_tuned_generation_settings(profile)
            self.config.setdefault("generation", {})
            self.config["generation"]["temperature"] = recommended["temperature"]
            self.config["generation"]["max_tokens"] = recommended["max_tokens"]
            if hasattr(self.parent(), "model_loader"):
                self.parent().model_loader.config["generation"]["temperature"] = recommended["temperature"]
                self.parent().model_loader.config["generation"]["max_tokens"] = recommended["max_tokens"]
            
            self.temp_box.setValue(int(recommended["temperature"] * 100))
            self.token_box.setValue(recommended["max_tokens"])

            QMessageBox.information(
                self,
                "Backend & Settings Applied",
                f"Processing will now use {best.upper()}.\n"
                f"Settings updated: Temperature = {recommended['temperature']}, Max Tokens = {recommended['max_tokens']}"
            )
            loader.config = self.config
            loader.save_config()
            if hasattr(self.parent(), "update_model_display"):
                model_name = self.config.get("default_model", {}).get("model_name", "Unknown")
                self.parent().update_model_display(model_name)
        self.benchmark_button.setEnabled(True)
        self.benchmark_button.setText("Run Benchmark")


class AIForgeUI(QWidget):
    def __init__(self):
        super().__init__()
        print("👷‍♂️ AIForgeUI.__init__() starting...")

        # Load plugins early
        from core.plugin_loader import load_plugins
        print("🔌 Calling load_plugins()...")
        self.plugins = load_plugins()
        print(f"🔌 Plugins loaded: {len(self.plugins)}")

        # Start with all plugins disabled; user will toggle them in the sidebar
        self.enabled_plugins = {plugin.get_name(): False for plugin in self.plugins}

        # Local search manager and mode flags
        self.local_search_manager = LocalSearchManager()
        

        # Window setup
        self.setWindowTitle("AI Forge")
        self.setWindowIcon(QIcon("Lulu-X.ico"))
        self.setMinimumSize(1000, 700)

        # Model loader & HF runner
        self.model_loader = ModelLoader()
        self.model_loader.load_model()
        self.hf_runner = HFRunner()
        self.backend_used = self.model_loader.config["performance"].get("backend", "cpu")

        # Tune generation settings based on hardware
        profile = get_system_profile()
        recommended = get_tuned_generation_settings(profile)
        self.model_loader.config.setdefault("generation", {})
        self.model_loader.config["generation"]["temperature"] = recommended["temperature"]
        self.model_loader.config["generation"]["max_tokens"] = recommended["max_tokens"]
        self.model_loader.save_config()

        # Conversation history and theme
        self.history = []
        self.ui_color = self.model_loader.config.get("ui_color", "#009EEB")

        # Build the UI and apply theming
        self.init_ui()
        self.apply_theme_color()




    def init_ui(self):
        from local_hf_runner import HF_MODEL_MAP  # ensures HF models are accessible

        main_layout = QHBoxLayout()
        self.toggle_sidebar_button = QPushButton()
        self.toggle_sidebar_button.clicked.connect(self.toggle_sidebar)
        self.toggle_sidebar_button.setFixedWidth(100)
        main_layout.addWidget(self.toggle_sidebar_button)

        self.sidebar = QVBoxLayout()
        content_area = QVBoxLayout()

        # Model selector
        self.model_selector = QComboBox()
        ollama_models = self.model_loader.list_ollama_models()
        hf_models = list(HF_MODEL_MAP.keys())
        all_models = ollama_models + hf_models
        self.model_selector.addItems(all_models)

        current = self.model_loader.config["default_model"].get("model_name")
        if current in all_models:
            self.model_selector.setCurrentText(current)
        else:
            self.model_selector.setCurrentIndex(0)

        self.sidebar.addWidget(QLabel("🧠 Model Selector"))
        self.sidebar.addWidget(self.model_selector)
        self.model_selector.currentTextChanged.connect(self.on_model_changed)

        # Sidebar buttons
        self.settings_button = QPushButton("🛠️ Settings")
        self.settings_button.clicked.connect(self.open_settings)
        self.sidebar.addWidget(self.settings_button)
        
        self.image_settings_button = QPushButton("🖌️ Image Settings")
        self.image_settings_button.clicked.connect(self.open_image_settings)
        self.sidebar.addWidget(self.image_settings_button)

        self.save_button = QPushButton("🧱 Save Session")
        self.save_button.clicked.connect(self.save_session)
        self.sidebar.addWidget(self.save_button)

        self.load_button = QPushButton("📜 Load Session")
        self.load_button.clicked.connect(self.load_session)
        self.sidebar.addWidget(self.load_button)

        self.import_button = QPushButton("📥 Add Content")
        self.import_button.clicked.connect(
            lambda: run_import_dialog(self, self.local_search_manager)
        )
        self.sidebar.addWidget(self.import_button)

        self.sidebar.addWidget(self.import_button)

        self.sidebar.addSpacing(10)

        # Prompt Template
        self.template_selector = QComboBox()
        self.templates = load_templates()
        self.template_selector.addItem("None")
        self.template_selector.addItems(self.templates.keys())
        self.template_selector.currentTextChanged.connect(self.on_template_selected)
        self.sidebar.addWidget(QLabel("🧩 Prompt Template"))
        self.sidebar.addWidget(self.template_selector)

        self.save_template_button = QPushButton("💾 Save as Template")
        self.save_template_button.clicked.connect(self.save_prompt_as_template)
        self.sidebar.addWidget(self.save_template_button)

        self.delete_template_button = QPushButton("🗑️ Delete Template")
        self.delete_template_button.clicked.connect(self.delete_selected_template)
        self.sidebar.addWidget(self.delete_template_button)

        self.sidebar.addSpacing(10)

        # Prompt Chaining
        self.chain_list = QListWidget()
        self.chain_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.chain_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for name in self.templates.keys():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.chain_list.addItem(item)
        self.chain_list.itemChanged.connect(self.update_template_selector_state)
        self.load_chain_state()

        self.sidebar.addWidget(QLabel("🔗 Chain Prompts"))
        self.sidebar.addWidget(self.chain_list)

        self.sidebar.addStretch()

        # History
        self.history_list = QListWidget()
        self.history_list.setMaximumWidth(250)
        self.history_list.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.history_list.itemClicked.connect(self.restore_prompt_from_history)
        self.sidebar.addWidget(QLabel("📚 History"))
        self.sidebar.addWidget(self.history_list)

        # Sidebar container
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setLayout(self.sidebar)
        self.toggle_sidebar_button.setText("Close" if self.sidebar_widget.isVisible() else "Close")
        main_layout.addWidget(self.sidebar_widget)

        # Output box
        self.output_box = QWebEngineView()
        self.output_box.setPage(ExternalLinkPage(self.output_box))
        content_area.addWidget(self.output_box, 5)
        self.output_box.setHtml("""
            <html>
            <head>
            <style>
                body {
                    background-color: #1a1a2e;
                    color: #f0f0f0;
                    font-family: Consolas, monospace;
                    text-align: center;
                    padding-top: 100px;
                }
                h3 {
                    color: #8be9fd;
                }
                p {
                    color: #bd93f9;
                }
            </style>
            </head>
            <body>
                <h3>Welcome to AI Forge ✨</h3>
                <p>Enter a prompt below to get started!</p>
            </body>
            </html>
        """)

        # Buttons row
        button_row = QHBoxLayout()
        self.clear_button = QPushButton("🧹 Clear")
        self.clear_button.clicked.connect(self.clear_output)
        button_row.addWidget(self.clear_button)

        self.copy_button = QPushButton("📋 Copy Output")
        self.copy_button.clicked.connect(self.copy_output)
        button_row.addWidget(self.copy_button)

        self.regenerate_button = QPushButton("🔁 Regenerate")
        self.regenerate_button.clicked.connect(self.regenerate_last)
        button_row.addWidget(self.regenerate_button)

        self.search_files_button = QPushButton("🔍 Search Files")
        self.search_files_button.clicked.connect(self.handle_search)
        button_row.addWidget(self.search_files_button)

        self.image_gen_button = QPushButton("🎨 Generate Image")
        self.image_gen_button.clicked.connect(self.handle_image_gen)
        button_row.addWidget(self.image_gen_button)

        button_row.addStretch()

        
        self.model_display = QLabel()
        self.update_model_display(current)
        button_row.addWidget(self.model_display)
        content_area.addLayout(button_row)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.installEventFilter(self)
        content_area.addWidget(self.prompt_input, 2)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("GenerateButton")
        self.generate_button.clicked.connect(self.handle_generate)
        content_area.addWidget(self.generate_button)

        main_layout.addLayout(content_area)
        self.setLayout(main_layout)


    def update_template_selector_state(self):
        """
        Disable the single-template dropdown if any chained templates are selected.
        """
        any_checked = any(
            self.chain_list.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.chain_list.count())
        )
        self.template_selector.setEnabled(not any_checked)

    def open_image_settings(self):
        dialog = ImageGenSettingsDialog(self)
        dialog.exec()


    def apply_theme_color(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a2e;
                color: #FF5BFF;
            }
            QPushButton {
                background-color: #D34DEE;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #BF4BF8;
            }
            QTextBrowser, QTextEdit {
                background-color: #1a1a2e;
                color: #56F1FF;
                border: 1px solid #56F1FF;
                font-family: Consolas, monospace;
            }
            QListWidget {
                background-color: #1a1a2e;
                color: #FF5BFF;
                border: 1px solid #56F1FF;
            }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #BF4BF8;
                min-height: 20px;
                border-radius: 4px;
            }
            QLabel#Heading {
                color: #FF5BFF;
                font-size: 16px;
                font-weight: bold;
            }
        """
        )

    def on_model_changed(self, new_model):
        current_model = self.model_loader.config["default_model"].get("model_name")

        if new_model != current_model:
            self.model_loader.config["default_model"]["model_name"] = new_model
            self.model_loader.save_config()

        self.update_model_display(new_model)

    def toggle_sidebar(self):
        if self.sidebar_widget.isVisible():
            self.sidebar_widget.hide()
            self.toggle_sidebar_button.setText("Open")
        else:
            self.sidebar_widget.show()
            self.toggle_sidebar_button.setText("Close")

    def eventFilter(self, source, event):
        if source == self.prompt_input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers():
                self.handle_generate()
                return True
        return super().eventFilter(source, event)

    def open_settings(self):
        dialog = SettingsDialog(self, self.model_loader.config)
        if dialog.exec():
            self.model_loader.save_config()
            self.apply_theme_color()

    def save_prompt_as_template(self):
        from PyQt6.QtWidgets import QInputDialog
        from prompt_tools import save_prompt_template

        prompt_text = self.prompt_input.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "Empty Prompt", "There is no prompt to save as a template.")
            return

        name, ok = QInputDialog.getText(self, "Save Template", "Enter a name for this template:")
        if ok and name:
            try:
                save_prompt_template(name, prompt_text)
                self.templates = load_templates()
                self.template_selector.clear()
                self.template_selector.addItem("None")
                self.template_selector.addItems(self.templates.keys())
                QMessageBox.information(self, "Template Saved", f"'{name}' saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save template:\n{str(e)}")
    
    def on_template_selected(self, template_name: str):
        if template_name and template_name != "None":
            template = self.templates.get(template_name)
            if template:
                preview = template.replace("{{input}}", "[your input here]").replace("{{previous}}", "[previous output]")
                self.prompt_input.setPlainText(preview)
        else:
            self.prompt_input.clear()
            
    def delete_selected_template(self):
        selected = self.template_selector.currentText()
        if selected == "None":
            QMessageBox.information(self, "Delete Template", "Please select a template to delete.")
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the template '{selected}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if confirm == QMessageBox.StandardButton.Yes:
            from prompt_tools import delete_prompt_template
            delete_prompt_template(selected)
            self.templates = load_templates()
            self.template_selector.clear()
            self.template_selector.addItem("None")
            self.template_selector.addItems(self.templates.keys())
            self.prompt_input.clear()
            QMessageBox.information(self, "Template Deleted", f"'{selected}' has been removed.")



    def handle_generate(self):
        # Basic LLM generation only — no local search or image
        self.generate_button.setText("Generate")
        self.prompt_input.setPlaceholderText("Enter your prompt here...")

        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a prompt before generating.")
            return

        # ── Prompt‐Chaining / Template Mode ─────────────────────────────
        chain_templates = [
            self.chain_list.item(i).text()
            for i in range(self.chain_list.count())
            if self.chain_list.item(i).checkState() == Qt.CheckState.Checked
        ]

        if chain_templates:
            prompt = chain_prompts(chain_templates, prompt, self.model_loader)
        else:
            selected = self.template_selector.currentText()
            if selected != "None" and selected in self.templates:
                prompt = apply_template(self.templates[selected], prompt)

        # ── Model Dispatch ──────────────────────────────────────────────
        self.generate_button.setEnabled(False)
        self.output_box.repaint()

        dropdown_model = self.model_selector.currentText()

        if dropdown_model in HF_MODEL_MAP:
            self.model_loader.config["default_model"]["type"] = "huggingface"
            self.model_loader.config["default_model"]["model_name"] = HF_MODEL_MAP[dropdown_model]
            self.model_loader.save_config()
            self.hf_runner = HFRunner(HF_MODEL_MAP[dropdown_model])
            result = self.hf_runner.generate(prompt)
            self.display_result(prompt, result)
            return

        current_cfg = self.model_loader.config["default_model"].get("model_name")
        if dropdown_model != current_cfg:
            self.model_loader.config["default_model"]["model_name"] = dropdown_model
            self.model_loader.config["default_model"]["type"] = "ollama"
            self.model_loader.save_config()

        self.update_model_display(dropdown_model)

        self.generated_text = ""
        self.thread = GenerationThread(self.model_loader, prompt)
        self.thread.result_ready.connect(self.append_stream_chunk)
        self.thread.finished.connect(self.finish_stream)
        self.thread.start()




    def preview_chained_prompt(self):
        user_input = self.prompt_input.toPlainText().strip()
        if not user_input:
            QMessageBox.information(self, "Empty Input", "Enter a prompt before previewing.")
            return

        chain_templates = [
            self.chain_list.item(i).text()
            for i in range(self.chain_list.count())
            if self.chain_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        if not chain_templates:
            QMessageBox.information(self, "No Templates", "No templates selected for chaining.")
            return

        try:
            preview = chain_prompts(chain_templates, user_input)
            QMessageBox.information(self, "Chain Preview", f"Resulting chained prompt:\n\n{preview}")
        except ValueError as e:
            QMessageBox.warning(self, "Chain Error", str(e))


    def append_stream_chunk(self, chunk):
        self.generated_text += chunk

    def finish_stream(self, prompt):
        self.display_result(prompt, self.generated_text)

    def display_result(self, prompt: str, result: str):
        from PyQt6.QtCore import QUrl
        from pygments.formatters import HtmlFormatter
        from db import get_connection

        plugin_input = {
            "text": result,
            "original_prompt": prompt  # ✅ Needed for keyword detection in plugins
        }

        # 🔌 Run plugin pipeline
        for plugin in self.plugins:
            name = plugin.get_name()
            if self.enabled_plugins.get(name, True):
                try:
                    if plugin.plugin_type() == "post_proc":
                        plugin_input = plugin.run(plugin_input)
                    elif plugin.plugin_type() == "image_gen" and prompt.lower().startswith("image:"):
                        plugin.run(plugin_input)
                except Exception as e:
                    print(f"[Plugin Error] {name}: {e}")

        result = plugin_input.get("text", result)

        # 📄 Convert to HTML
        raw_html = markdown2.markdown(
            result, extras=["fenced-code-blocks", "break-on-newline", "code-friendly"]
        )

        highlighted = self.highlight_code_blocks(raw_html)
        block = f"""
        <div class="ai-output">
            <b style="color:#ff79c6;">Prompt:</b><br><i>{prompt}</i><hr>
            <b style="color:#8be9fd;">Response:</b><br>{highlighted}
            <hr><br>
        </div>
        """

        if not hasattr(self, "html_history"):
            self.html_history = ""

        self.html_history += block
        html = f"""
        <html>
        <head>
        <style>
            .highlight {{
                background-color: #282a36;
                padding: 12px;
                border-radius: 6px;
                overflow-x: auto;
            }}
            pre, code {{
                font-family: Consolas, monospace;
                font-size: 14px;
                white-space: pre-wrap;
                background: none;
            }}
            b, i {{
                color: #bd93f9;
            }}
            hr {{
                border: 0;
                height: 1px;
                background: #444;
            }}
        </style>
        </head>
        <body style="background-color: #1a1a2e; color: #f0f0f0; font-family: Consolas, monospace;">
            {self.html_history}
        </body>
        </html>
        """

        self.output_box.setHtml(html, baseUrl=QUrl("about:blank"))
        self.history.append((prompt, result))
        self.history_list.addItem(prompt[:40] + "...")

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO history (prompt, response) VALUES (?, ?)", (prompt, result)
            )
            conn.commit()

        self.generate_button.setEnabled(True)
        self.output_box.page().runJavaScript("window.scrollTo(0, document.body.scrollHeight);")


    def update_model_display(self, model_name):
        backend = self.model_loader.config["performance"].get("backend", "cpu")

        if backend == "auto":
            backend = self.model_loader.choose_best_backend()

        self.model_display.setText(
            f"<b>Model:</b> <span style='color:#56F1FF;'>{model_name}</span><br>"
            f"<b>Processing with:</b> <span style='color:#56F1FF;'>{backend.upper()}</span>"
        )

    def clear_output(self):
        self.html_history = ""
        self.output_box.setHtml("")

    def copy_output(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_box.toPlainText())
        QMessageBox.information(self, "Copied", "Output copied to clipboard.")

    def regenerate_last(self):
        if self.history:
            last_prompt, _ = self.history[-1]
            self.prompt_input.setPlainText(last_prompt)
            self.handle_generate()

    def save_session(self):
        file_path = QFileDialog.getSaveFileName(
            self, "Save Session", "session.json", "JSON Files (*.json)"
        )[0]
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            QMessageBox.information(self, "Saved", "Session saved successfully.")

    def load_session(self):
        file_path = QFileDialog.getOpenFileName(
            self, "Load Session", "", "JSON Files (*.json)"
        )[0]
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
            self.history_list.clear()
            self.output_box.clear()
            for prompt, response in self.history:
                html = markdown2.markdown(
                    response, extras=["fenced-code-blocks", "break-on-newline"]
                )
                self.display_result(prompt, response)
                self.history_list.addItem(prompt[:40] + "...")
            QMessageBox.information(self, "Loaded", "Session loaded successfully.")
            
    def save_chain_state(self):
        """Save the current chain prompt selection and order."""
        state = []
        for i in range(self.chain_list.count()):
            item = self.chain_list.item(i)
            state.append({
                "name": item.text(),
                "checked": item.checkState() == Qt.CheckState.Checked
            })
        with open("chain_state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_chain_state(self):
        """Load the saved chain prompt selection and order."""
        try:
            with open("chain_state.json", "r", encoding="utf-8") as f:
                state = json.load(f)
            self.chain_list.clear()
            for entry in state:
                item = QListWidgetItem(entry["name"])
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if entry["checked"] else Qt.CheckState.Unchecked)
                self.chain_list.addItem(item)
        except FileNotFoundError:
            pass  

    def handle_search(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing Query", "Please enter something to search for.")
            return

        print(f"[LocalSearch Triggered] Searching for: {prompt}")

        try:
            plugin = next(p for p in self.plugins if p.get_name() == "Local Document Search")
        except StopIteration:
            QMessageBox.critical(self, "Search Plugin Missing", "Local Document Search plugin not found.")
            return

        plugin_input = {"text": "", "original_prompt": prompt}
        output = plugin.run(plugin_input)
        result = output.get("text", "⚠️ No documents returned.")

        self.display_result(prompt, result)


    def handle_image_gen(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a description for your image.")
            return

        print(f"[ImageGeneration Triggered] Generating image for: {prompt}")

        try:
            plugin = next(p for p in self.plugins if p.plugin_type() == "image_gen")
        except StopIteration:
            QMessageBox.critical(self, "Image Plugin Missing", "Image Generation plugin not found.")
            return

        plugin_input = {"original_prompt": prompt}
        result = plugin.run(plugin_input)

        if "error" in result:
            QMessageBox.critical(self, "Image Error", result["error"])
            return

        img_src = result.get("image_path") or result.get("url") or result.get("text")
        if not img_src or not str(img_src).lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            QMessageBox.critical(self, "Image Error", "Plugin didn’t return a valid image path or URL.")
            return

        html = f'''
        <html><body style="margin:0; background:#000;">
            <img src="file:///{img_src}" style="width:100%;height:auto;"/>
        </body></html>'''
        self.output_box.setHtml(html)



        
    def restore_prompt_from_history(self, item):
        index = self.history_list.row(item)
        if 0 <= index < len(self.history):
            self.prompt_input.setPlainText(self.history[index][0])

    def highlight_code_blocks(self, html_text):
        code_block_pattern = re.compile(
            r'<pre><code(?: class="language-(\w+)")?>(.*?)</code></pre>', re.DOTALL
        )

        def replacer(match):
            lang = match.group(1) or "text"
            code = match.group(2)

            # unescape HTML
            code = (
                code.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
                .replace("&quot;", '"')
            )

            try:
                lexer = get_lexer_by_name(lang)
            except Exception:
                lexer = guess_lexer(code)

            formatter = HtmlFormatter(noclasses=True, style="colorful", nowrap=True)
            highlighted = highlight(code, lexer, formatter)
            return f'<div class="highlight">{highlighted}</div>'

        return code_block_pattern.sub(replacer, html_text)


if __name__ == "__main__":
    init_db()
    print("🚀 Creating AIForgeUI window...")
    app = QApplication(sys.argv)
    window = AIForgeUI()
    window.show()
    sys.exit(app.exec())
