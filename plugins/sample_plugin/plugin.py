from core.plugin_base import AIForgePlugin

class Plugin(AIForgePlugin):
    def __init__(self, config={}):
        super().__init__(config)

    def get_name(self):
        return "Sample Plugin"

    def plugin_type(self):
        return "post_proc"  # or whatever type you want ("image_gen", "pre_proc", etc)

    def run(self, input_data: dict):
        print("[SamplePlugin] run() called")
        return input_data
