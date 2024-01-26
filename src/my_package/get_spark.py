"""
This module contains the GetSpark class which is responsible for initializing a Spark session.
It checks if the code is running in a Databricks environment and initializes the session accordingly.
If not running in Databricks, it uses the Databricks Connect SDK to establish a session.

Classes:
    GetSpark: A class used to manage the initialization of a Spark session.
"""

import os


class GetSpark:
    """
    This class is responsible for initializing a Spark session.
    It checks if the code is running in a Databricks environment and initializes the session accordingly.
    If not running in Databricks, it uses the Databricks Connect SDK to establish a session.

    Attributes:
        profile (str): The profile to use for the Databricks Connect SDK. Defaults to "default".
    """

    def __init__(self, profile: str = "DEFAULT"):
        """
        The constructor for GetSpark class.

        Parameters:
            profile (str): The profile to use for the Databricks Connect SDK. Defaults to "default".
        """
        self.profile = profile

    def is_running_in_databricks(self) -> bool:
        """
        Checks if the code is running in a Databricks environment.

        Returns:
            bool: True if running in Databricks, False otherwise.
        """
        return "DATABRICKS_RUNTIME_VERSION" in os.environ

    def init_spark(self, eager=True):
        """
        Initializes a Spark session. If the code is running in a Databricks environment, it initializes the session
        accordingly. If not running in Databricks, it uses the Databricks Connect SDK to establish a session.

        Returns:
            SparkSession: The initialized Spark session.
        """
        if self.is_running_in_databricks():
            from pyspark.sql import SparkSession

            return SparkSession.builder.getOrCreate()

        from databricks.connect import DatabricksSession
        from databricks.sdk.core import Config

        config = Config(profile=self.profile)
        spark = DatabricksSession.builder.sdkConfig(config).getOrCreate()
        if eager:
            spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
            spark.conf.set("spark.sql.repl.eagerEval.maxNumRows", 3)
            return spark
        return spark
