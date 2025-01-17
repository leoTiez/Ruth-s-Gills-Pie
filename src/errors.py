class InternalException(Exception):
    def __init__(self, message: str = ''):
        msg = 'Internal error. Please report a bug. ' + message
        super(InternalException, self).__init__(msg)


class PossibleInternalException(Exception):
    def __init__(self, message: str = ''):
        msg = 'This could be an internal error. Before reporting a bug, please check the following: ' + message
        super(PossibleInternalException, self).__init__(msg)


class ValueNotInitialized(Exception):
    def __init__(self, value_str: str = '', function_str: str = ''):
        msg = 'Value has not been yet initialized: %s. The value must be set before running function: %s' % (
            value_str, function_str)
        super(ValueNotInitialized, self).__init__(msg)


class ReactionError(Exception):
    def __init__(self, message: str = ''):
        msg = 'Reaction could not be sampled. That means that there is in incoherence in the reaction path.' \
              'Please check your reactions. If the error persists, please report an issue. ' + message
        super(ReactionError, self).__init__(msg)


class FileNotCreatedError(Exception):
    def __init__(self, message: str = ''):
        msg = 'File has not been created yet. %s' % message
        super(FileNotCreatedError, self).__init__(msg)


