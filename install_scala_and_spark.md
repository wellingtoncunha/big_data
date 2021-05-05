
# Configuring Scala and Spatk on Hadoop cluster

## Scala Installation

First step in our configuration is installing scala, which is the default language used by Spark (and in Spark shell). But later you can use other languages, as Python

1. Installing Scala is pretty straighforward. We use apt-get to it:

    ```bash
    sudo apt-get install scala -y
    ```

## Spark Installation

Spark installation requires more steps. But it is not rocket science!

1. Download Spark files:

    ```bash
    wget https://apache.osuosl.org/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz -P ~/Downloads
    ```

2. Extract Spark files to the desired installation folder:

    ```bash
    sudo tar zxvf ~/Downloads/spark-2.4.7-bin-hadoop2.7.tgz -C /usr/local
    ```

3. Just to make things neat we are going to rename the installation folder and remove the installation files:

    ```bash
    # Remove installation file
    rm ~/Downloads/spark-2.4.7-bin-hadoop2.7.tgz 

    # Rename Hadoop directory
    sudo mv /usr/local/spark-2.4.7-bin-hadoop2.7 /usr/local/spark
    ```

4. We now need to add the environment variables to some files in order to have them set during boot. The reason for setting them on those differents files is that there are slightly differences on where those variables should be placed, depending on the configuration of the OS. Better be safe than sorry! Here are the variables that we are going to set:

    ```bash
    # SPARK configurations
    export SPARK_HOME=/usr/local/spark
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native
    export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH
    export PATH=$SPARK_HOME/bin:$SPARK_HOME/python:$PATH
    ```

    And here are the places (add the block above to the end of each file):
    * nano ~/.bashrc:

        ```bash
        nano ~/.bashrc
        ```

    * nano ~/.profile:

        ```bash
        nano ~/.profile
        ```

    * sudo nano /etc/profile:

        ```bash
        sudo nano /etc/profile
        ```

    * sudo nano /etc/bash.bashrc

        ```bash
        sudo nano /etc/bash.bashrc
        ```

    * And last but not least, let's set them for our current session:

        ```bash
        source ~/.bashrc
        ```

5. Configure 

    ```bash
    sudo nano $SPARK_HOME/conf/spark-defaults.conf
    ```

    * Then add the folllowing to the file:

        ```bash
        spark.master yarn
        spark.driver.memory 512m
        spark.yarn.am.memory 512m
        spark.executor.memory 512m
        ```

5. To start Spark shell we use:

    ```bash
    spark-shell
    ```

6. It takes a little while to start Spark, but once in, we can test using Scala (it is the default language for Spark shell):

    ```scala
    println("Spark shell is running")
    ```

7. An to exit Spark shell/Scala, we use:

    ```scala
    :q
    ```

I have used the [Installing Scala and Spark on Ubuntu](https://medium.com/@josemarcialportilla/installing-scala-and-spark-on-ubuntu-5665ee4b62b1) article by Jose Marcial Portilla to build this guide.




