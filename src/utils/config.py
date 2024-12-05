class Settings:
    def __init__(self):
        # Example configuration settings
        self.api_base_url = "https://api.example.com"
        self.timeout = 30

    def get_api_base_url(self):
        return self.api_base_url

    def get_timeout(self):
        return self.timeout
