# from pyspark.sql import SparkSession


# def get_df():
#     spark = GetSpark().init_spark()

#     data = [
#         {"Category": "A", "ID": 1, "Value": 121.44, "Truth": True},
#         {"Category": "B", "ID": 2, "Value": 300.01, "Truth": False},
#         {"Category": "C", "ID": 3, "Value": 10.99, "Truth": None},
#         {"Category": "E", "ID": 4, "Value": 33.87, "Truth": True},
#     ]
#     df = spark.createDataFrame(data)
#     df.show()

from my_package.get_spark import GetSpark


def get_taxis():
    spark = GetSpark().init_spark()
    # spark = SparkSession.builder.getOrCreate()
    return spark.read.table("samples.nyctaxi.trips")


def main():
    get_taxis().show(5)


if __name__ == "__main__":
    main()


# from my_package.my_module import *


# def main():
#     first = 200
#     second = 400

#     print(f"{first} + {second} = {add_two_numbers(first, second)}")
#     print(f"{second} - {first} = {subtract_two_numbers(second, first)}")
#     print(f"{first} * {second} = {multiply_two_numbers(first, second)}")
#     print(f"{second} / {first} = {divide_two_numbers(second, first)}")


# if __name__ == "__main__":
#     main()
