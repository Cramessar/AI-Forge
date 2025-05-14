# plugin_loader.py
import importlib.util
import os
from core.plugin_base import AIForgePlugin

PLUGIN_FOLDER = "plugins"

def load_plugins():
    print("🧩 Scanning plugins folder...")
    plugins = []
    for folder in os.listdir(PLUGIN_FOLDER):
        plugin_path = os.path.join(PLUGIN_FOLDER, folder, "plugin.py")
        if os.path.exists(plugin_path):
            print(f"🧩 Found plugin: {folder}")
            spec = importlib.util.spec_from_file_location(f"{folder}_plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugin_class = getattr(module, "Plugin", None)
            if plugin_class and issubclass(plugin_class, AIForgePlugin):
                plugin_instance = plugin_class()
                plugins.append(plugin_instance)
            else:
                print(f"⚠️ Warning: No valid Plugin class found in {folder}")
    print(f"🧩 Finished loading {len(plugins)} plugin(s).")
    return plugins
