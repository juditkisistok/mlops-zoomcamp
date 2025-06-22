import datetime
import time
import random
import logging
import uuid
import pytz
import psycopg

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics (
    timestamp TIMESTAMP,
    value1 INTEGER,
    value2 VARCHAR,
    value3 FLOAT
);
"""

def prep_db():
    # access the database with psycopg connect
    with psycopg.connect("host=localhost port=5432 user=postgres password=postgres", autocommit=True) as conn:
        res = conn.execute("SELECT 1 from pg_database where datname = 'test'")
        if len(res.fetchall()) == 0:
            # if it returns nothing, it means that the database doesn't exist
            # then we create the database
            conn.execute("CREATE DATABASE test;")
    with psycopg.connect("host=localhost port=5432 user=postgres password=postgres dbname=test") as conn:
        conn.execute(create_table_statement)

def calculate_dummy_metrics(curr):
    value1 = rand.randint(0, 1000)
    value2 = str(uuid.uuid4())
    value3 = rand.random()

    curr.execute("INSERT INTO dummy_metrics (timestamp, value1, value2, value3) VALUES (%s, %s, %s, %s)", 
                 (datetime.datetime.now(pytz.timezone("Europe/Budapest")), 
                  value1, value2, value3))
    
def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)
    with psycopg.connect("host=localhost port=5432 user=postgres password=postgres dbname=test", autocommit=True) as conn:
        for i in range(100):
            with conn.cursor() as curr:
                calculate_dummy_metrics(curr)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send += datetime.timedelta(seconds=SEND_TIMEOUT)
            logging.info(f"Sent metrics at {last_send}")

if __name__ == "__main__":
    main()
