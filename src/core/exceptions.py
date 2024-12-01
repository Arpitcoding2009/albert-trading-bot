class ExchangeError(Exception):
    """Exception raised for errors related to exchange operations."""
    def __init__(self, message="An error occurred with the exchange operation"):
        self.message = message
        super().__init__(self.message)
