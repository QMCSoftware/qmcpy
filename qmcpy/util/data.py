import gzip
import pickle

from ..util import _univ_repr


class Data(object):

    def __init__(self, parameters):
        self.parameters = parameters

    def save(self, path, compress=False, overwrite=False):
        """Save this Data object to disk using pickle.

        Warnings:
            ``pickle`` files are not secure against untrusted input. Only save
            and later load checkpoint files that you created yourself or that
            come from a trusted source.

        Args:
            path (str or pathlib.Path): File path to save to. If
                ``compress=True``, a ``.gz`` suffix is appended automatically
                when not already present.
            compress (bool): Gzip-compress the saved file. Defaults to False.
            overwrite (bool): If False (default), raise ``FileExistsError``
                when the file already exists. If True, overwrite any existing
                file.

        Returns:
            The final path the file was written to (may differ from *path* when
            ``compress=True`` appends ``.gz``).

        Raises:
            FileExistsError: If the target path already exists and
                ``overwrite=False``.
        """
        import os
        path = str(path)
        if compress and not path.endswith(".gz"):
            path = path + ".gz"
        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists; pass overwrite=True to replace it."
            )
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path):
        """Load a Data object from disk.

        Warnings:
            ``pickle`` deserialization can execute arbitrary code. Only load
            checkpoint files that you created yourself or that come from a
            trusted source.

        Args:
            path (str or pathlib.Path): Path to the saved file. Files ending in
                ``.gz`` are decompressed automatically.

        Returns:
            The loaded Data object.
        """
        path = str(path)
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, cls):
            raise TypeError(
                "checkpoint did not contain a %s instance; got %s."
                % (cls.__name__, type(loaded).__name__)
            )
        return loaded

    def __repr__(self):
        string = _univ_repr(self, "Data", self.parameters + ["time_integrate"])
        if hasattr(self, "stopping_crit") and self.stopping_crit:
            string += "\n" + str(self.stopping_crit)
        if hasattr(self, "integrand") and self.integrand:
            string += "\n" + str(self.integrand)
        if hasattr(self, "true_measure") and self.true_measure:
            string += "\n" + str(self.true_measure)
        if hasattr(self, "discrete_distrib") and self.discrete_distrib:
            string += "\n" + str(self.discrete_distrib)
        return string
