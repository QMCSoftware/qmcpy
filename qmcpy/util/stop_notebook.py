def stop_notebook(query="Type 'yes' to continue running notebook"):
    """Prompt at a notebook checkpoint and halt execution unless the user confirms.

    Placed between cells so that "Run All" pauses instead of running an
    expensive section unattended. Any answer other than ``yes`` (case
    insensitive) calls :func:`sys.exit`, which the notebook kernel reports as a
    stopped cell rather than a traceback.

    Args:
        query (str): Prompt shown to the user.

    Raises:
        SystemExit: If the answer is not ``yes``.
    """
    keep_running = input(query)
    if keep_running.casefold() != "yes":
        import sys
        import warnings

        warnings.filterwarnings("ignore")
        sys.exit("Pausing notebook execution")
