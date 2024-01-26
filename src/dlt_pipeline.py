import sys

import dlt
from pyspark.sql.functions import expr

from my_project import main


@dlt.view
def taxi_raw():
    return main.get_taxis()


@dlt.table
def filtered_taxis():
    return dlt.read("taxi_raw").filter(expr("fare_amount < 30"))
