# imports from packages
import calendar
import time
import datetime

def get_readable_timestamp():
    ts = calendar.timegm(time.gmtime())
    readable = datetime.datetime.fromtimestamp(ts).isoformat()
    string_timestamp = str(readable)

    return string_timestamp
