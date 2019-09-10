class DobuleInitalizationException(Exception):
    def __init__(self, message, error):

        message = message + " | Check the following parameter: " + error + "."

        super().__init__(message)
