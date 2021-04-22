## Scala Installation


* install scala

```bash
sudo apt-get install scala
```

* Starting Scala:

```bash
scala
```

* Testing Scala:

```scala
println(“Hello World”)
```

* Quitting Scala:

```scala
q:
```

## Spark Installation

* Download Spark:

```bash
wget https://apache.osuosl.org/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz -P ~/Downloads
```

* Untar Spark

```bash
sudo tar zxvf ~/Downloads/spark-2.4.7-bin-hadoop2.7.tgz -C /usr/local
```

* Rename folder

```bash
# Remove installation file
rm ~/Downloads/spark-2.4.7-bin-hadoop2.7.tgz 

# Rename Hadoop directory
sudo mv /usr/local/spark-2.4.7-bin-hadoop2.7 /usr/local/spark
```

*  add to ~/.bashrc:

    ```bash
    nano ~/.bashrc
    ```

    * Add the following

    ```bash
    # SPARK configurations
    export SPARK_HOME=/usr/local/spark
    export PATH=$PATH:$SPARK_HOME/bin
    ```
    
    * And last but not least, let's set them for our current session:

    ```bash
    source ~/.bashrc
    ```

* To start Spark

```bash
spark-shell
```

https://medium.com/@josemarcialportilla/installing-scala-and-spark-on-ubuntu-5665ee4b62b1