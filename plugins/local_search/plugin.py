# Example minimal plugin.py
from core.plugin_base import AIForgePlugin
from core.utils.local_search_manager import LocalSearchManager

class Plugin(AIForgePlugin):
    def __init__(self, cfg={}):
        super().__init__(cfg)
        self.manager = LocalSearchManager()

    def get_name(self):   return "Local Document Search"
    def plugin_type(self):return "post_proc"

    def run(self, input_data):
        q = input_data.get("original_prompt","").split("search:",1)[-1].strip()
        result = self.manager.search(q)
        input_data["text"] += "\n\n" + result
        return input_data
