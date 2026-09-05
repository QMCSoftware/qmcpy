
# def custom_formatwarning(msg, *args, **kwargs):
#     # ignore everything except the message
#     return '%s:%d\n\t%s: %s\n'%(args[1],args[2],args[0].__name__,str(msg))

# warnings.formatwarning = custom_formatwarning


class DimensionError(Exception):
    """Class for raising error about dimension
    """


class DistributionCompatibilityError(Exception):
    """Class for raising error about incompatible distribution
    """


class NotYetImplemented(Exception):
    """Class for raising error when a component has been implemented yet
    """


class MethodImplementationError(Exception):
    """Class for raising error when an abstract method has not been
    implemented in the child class.
    """

    def __init__(self, subclass, method_name):
        s_f = (
            "%s does not have an implementation of the  %s method. "
            + "See superclass for method description."
        )
        super(MethodImplementationError, self).__init__(
            s_f % (type(subclass).__name__, method_name)
        )


class ParameterError(Exception):
    """Class for raising error about input parameters
    """


class ParameterWarning(Warning):
    """Class for issuing warnings about unacceptable parameters
    """


class MaxSamplesWarning(Warning):
    """Class for issuing warning about using maximum number of data samples
    """


class MaxLevelsWarning(Warning):
    """Class for issuing warning about using maximum number of data samples
    """


class CubatureWarning(Warning):
    """Class for issuing warnings throughout cubature algorithms
    """
