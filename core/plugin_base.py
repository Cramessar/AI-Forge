class AIForgePlugin:
    def __init__(self, config: dict = {}):
        self.config = config

    def get_name(self) -> str:
        raise NotImplementedError("Plugin must implement get_name")

    def run(self, input_data: dict) -> dict:
        raise NotImplementedError("Plugin must implement run")

    def plugin_type(self) -> str:
        return "generic"  # Override with 'tts', 'image_gen', etc.
