import threading
import datetime

global logfile_path

# really dirty hack to provide logging as functions instead of objects
def init_logger(logfilepath: str) -> None:
    logfile_path:str = logfilepath


def log_it(message: str) -> None:
    message = (
        "["
        + str(threading.get_ident())
        + "] ["
        + str(datetime.datetime.now())
        + "] "
        + str(message)
    )

    if logfile_path == "console": # type: ignore
        print(message)
    else:
        with open(logfile_path, "a") as f:  # type: ignore
            f.write(message + "\n")
