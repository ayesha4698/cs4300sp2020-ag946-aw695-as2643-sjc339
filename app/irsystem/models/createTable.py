import json
import csv
import sys
import datetime
import psycopg2
from psycopg2 import connect, Error

update_podcasts_and_reviews = False

try:
    conn = connect(
        dbname="my_app_db",
        user="postgres",
        host="localhost",
        password="1234",
        # attempt to connect for 3 seconds then raise exception
        connect_timeout=3
    )

    cur = conn.cursor()
    print("\ncreated cursor object:", cur)

except (Exception, Error) as err:
    print("\npsycopg2 connect error:", err)
    conn = None
    cur = None