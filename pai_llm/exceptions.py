class PaiLLMError(Exception):
    pass


class DataConvertError(PaiLLMError):
    pass


class ConfigurationError(PaiLLMError):
    pass


class ValidationError(PaiLLMError):
    pass
