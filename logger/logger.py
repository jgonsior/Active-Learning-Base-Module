import datetime
import threading

# really dirty hack to provide logging as functions instead of objects
def init_logger(logfilepath: str) -> None:
    global logfile_path
    logfile_path = logfilepath


def log_it(message: str) -> None:
    message = (
        "["
        + str(threading.get_ident())
        + "] ["
        + str(datetime.datetime.now())
        + "] "
        + str(message)
    )

    if logfile_path == "console":
        print(message)
    else:
        with open(logfile_path, "a") as f:
            f.write(message + "\n")
