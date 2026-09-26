class InvalidLayerSizeError(ValueError):
    """Raised when a layer size is not a positive integer."""


class MismatchedDataError(ValueError):
    """Raised when data dimensions do not match model expectations."""


class DataNotLoadedError(RuntimeError):
    """Raised when an operation requires a dataset that is not loaded."""


class ModelNotInitializedError(RuntimeError):
    """Raised when an operation requires a model that has not been created."""
