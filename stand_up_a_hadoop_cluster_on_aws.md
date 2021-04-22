# Configuring a Hadoop Cluster on AWS

## Generating PEM Key on AWS

The very first step (assuming you already have an AWS account) is to generate a PEM Key file (if you do not have one yet) in order to get access to the intances you are going to use, if you do not have yet, please, check [How-to generate a PEM Key on AWS](generate_aws_pem_key.md)

## Provisioning an EC2 image with Hadoop and JAVA

To make it easy for configuring our cluster, we are going to provision an EC2 instance, install JAVA, Hadoop and some other common components for any node and then generate an image that we can use in the future to provision machines like that.

1. First, we need to create an EC2 instance. If you are not familiar with that, please, check [How-to provision an EC2 instance](provision_ec2_server.md)

2. Once the server is available, we are going now to connect (SSH) to it:

    ```bash
    ssh -i ~/Downloads/hadoop.pem ubuntu@<Public IPv4 address>
    ```

    **Note**: I'm using a Mac, on Windows machines you will need an application like [Putty](www.putty.org). This is the only step that is different. The next ones, once you are connected into the EC2 Linux Instance (using Putty, or any other application), are exactly the same.

3. Then inside the EC2 Instance, we first update apt and install nano (yes, I like it way more than vim):

    ```bash
    sudo apt-get update && sudo apt-get dist-upgrade -y && sudo apt install nano -y
    ```

4. Now, we are going to install JAVA (version 8 is the one that is working with this version of Hadoop and as the old saying goes: "if it is working don't touch it!"):

    ```bash
    sudo apt-get install openjdk-8-jdk -y
    ```

5. We are going now to download Hadoop version 2.8 (again, it is working). This may take a while:

    ```bash
    wget https://archive.apache.org/dist/hadoop/core/hadoop-2.8.1/hadoop-2.8.1.tar.gz -P ~/Downloads
    ```

6. Extract Hadoop to the folder where you want to have its installation:

    ```bash
    sudo tar zxvf ~/Downloads/hadoop-2.8.1.tar.gz -C /usr/local
    ```

7. Just to make things neat, let's remove the tar file and rename the folder by removing the version of it:

    ```bash
    # Remove installation file
    rm ~/Downloads/hadoop-2.8.1.tar.gz 

    # Rename Hadoop directory
    sudo mv /usr/local/hadoop-2.8.1 /usr/local/hadoop
    ```

8. We now need to add the environment variables to some files in order to have them set during boot. The reason for setting them on those differents files is that there are slightly differences on where those variables should be placed, depending on the configuration of the OS. Better be safe than sorry! Here are the variables that we are going to set:

    ```bash
    # JAVA configurations
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    export PATH=$PATH:$JAVA_HOME/bin

    #Hadoop Related Options
    export HADOOP_HOME=/usr/local/hadoop
    export PATH=$PATH:$HADOOP_HOME/bin
    export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
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

9. For some unknown reason, one of the scripts required to start one of the Hadoop services is not reading the JAVA_HOME variable correcly. So, let's hard-code it on the script:

    ```bash
    sudo nano $HADOOP_CONF_DIR/hadoop-env.sh
    ```

    The piece that needs to be set is at the line 25, so, just replace the placeholder ${JAVA_HOME} by the JAVA path (/usr/lib/jvm/java-8-openjdk-amd64):
    * Before:
    !["Before"](https://user-images.githubusercontent.com/7594950/107885305-737c3800-6ec7-11eb-84a0-65f77dadae67.png)

    * After:
    !["After"](https://user-images.githubusercontent.com/7594950/107885324-8abb2580-6ec7-11eb-8c49-734aed0293f7.png)

10. So we are now good to generate an AIM image for this server and will be able to use it to stand up new Hadoop clusters. Please, check [How-to create an AWS EC2 instance AIM image](generate_EC2_image.md)

## Provisioning EC2 instances from image

In this step we are going to provisioning 4 instances from the image that we just created. Please, check step 5.a. from [How-to provision an EC2 instance](provision_ec2_server.md). Then, on ***Configure Instance*** page, type 4 (the number of nodes we are going to configure) on **Number of instances** box. You can following the next steps as they are described on the how-to above.

![image202](https://user-images.githubusercontent.com/7594950/107978761-55780b80-6f8b-11eb-8da6-355c67f781dd.png)

## Initial configuration for Name Node

We are now going to configure the initial settings Name Node. So, let's first pick one of the instances to be our name node. We are going to configure it and replicated the configuration for the data nodes:

1. Once the EC2 instances are available, pick one of them to be the Name Node. At this point, I am also changing its Name tag to make clear that this is my name node (take the chance to copy the Public IPv4 Address):
![image203](https://user-images.githubusercontent.com/7594950/107979306-59585d80-6f8c-11eb-9d4a-84453f2ffe31.png)

2. In order to get access to the data nodes from inside the Name Node, you need to use the same PEM Key. So, let's copy that into our name node:

    ```bash
    scp -i ~/Downloads/hadoop.pem ~/Downloads/hadoop.pem ubuntu@<Public IPv4 address>:~/.ssh/ 
    ```

3. Now we can connect to the name node to continue with the configuration:

    ```bash
    ssh -i ~/Downloads/hadoop.pem ubuntu@<Public IPv4 address>
    ```

4. First step we are going to take inside the Name Node is to create a SSH configuration file to make it easy to get access to the Data Nodes from inside Name Node:

    ```bash
    nano ~/.ssh/config
    ```

5. To configure the file, we need to get the Private IPv4 DNS (without the ".ec2.internal" part, like ip-172-31-21-59) from every node (name and data). We must use the Private IP DNS because this DNS is fixed for your instances, while a new Public IP address is assigned to the instance everytime that it is rebooted. It is a good chance to update the name of data nodes (I am using "Hadoop Data Node - 1", "Hadoop Data Node - 2", etc.). With the Private IPv4 DNS written down, add the following to the file just created on previous step:

    ```bash
    Host namenode #You can use any alias you want for the hosts
        HostName <Name Node Private IPv4 DNS>
        User ubuntu
        IdentityFile ~/.ssh/hadoop.pem  
    Host datanode1 
        HostName  <Data Node 1 Private IPv4 DNS>
        User ubuntu
        IdentityFile ~/.ssh/hadoop.pem
    Host datanode2
        HostName <Data Node 2 Private IPv4 DNS>
        User ubuntu
        IdentityFile ~/.ssh/hadoop.pem
    Host datanode3
        HostName <Data Node 3 Private IPv4 DNS>
        User ubuntu
        IdentityFile ~/.ssh/hadoop.pem
    ```

6. Now we need to generate a public SSH key to be used by the cluster communication:

    ```bash
    ssh-keygen -f ~/.ssh/id_rsa -t rsa -P ""
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    ```

7. I recommend you to test the connection to each datanode. Both for ensure that the configuration file was correctly set, as well to add the new IP to the list of the trus IP address on the master node:

    ```bash
    ssh datanode1 #Look how easy it is now to get access to the nodes =)
    ```

8. And then, copy the SSH key to every node:

    ```bash
    ssh datanode1 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub
    ssh datanode2 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub
    ssh datanode3 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub
    ```

## Common configuration for all nodes

We are going now to configure some files that are common to every node. We are going to do it on the name node and then copy to every datanode

1. Configure the **core-site.xml** file:

    ```bash
    cd $HADOOP_CONF_DIR
    sudo nano core-site.xml
    ```

    Replace the configuration section of template file with the following, adding the Private IPv4 DNS of Name Node to the appropriate placeholder:

    ```xml
    <configuration>
        <property>
            <name>fs.defaultFS</name>
            <value>hdfs://<Name Node Private IPv4 DNS>:9000</value>
        </property>
    </configuration>
    ```

2. Configure ***yarn-site.xml*** file:

    ```bash
    cd $HADOOP_CONF_DIR
    sudo nano yarn-site.xml
    ```

    Replace the configuration section of template file with the following, adding the Private IPv4 DNS of Name Node to the appropriate placeholder:

    ```xml
    <configuration>
        <property>
            <name>yarn.nodemanager.aux-services</name>
            <value>mapreduce_shuffle</value>
        </property>
        <property>
            <name>yarn.resourcemanager.hostname</name>
            <value><Name Node Private IPv4 DNS></value>
        </property>
    </configuration>
    ```

3. Configure mapred-site.xml file:

    ```bash
    cd $HADOOP_CONF_DIR
    sudo cp mapred-site.xml.template mapred-site.xml
    sudo nano mapred-site.xml
    ```

    Replace the configuration section of template file with the following, adding the Private IPv4 DNS of Name Node to the appropriate placeholder:

    ```xml
    <configuration>
        <property>
            <name>mapreduce.jobtracker.address</name>
            <value><Name Node Private IPv4 DNS>:54311</value>
        </property>
        <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
        </property>
    </configuration>
    ```

4. Configure hdfs-site.xml file:

    ```bash
    cd $HADOOP_CONF_DIR
    sudo nano hdfs-site.xml
    ```

    Replace the configuration section of template file with the following. You can change the number of replicas for the data (here we are sticking with 3, which is the default) and/or the HDFS location:

    ```xml
    <configuration>
        <property>
            <name>dfs.replication</name>
            <value>3</value>
        </property>
        <property>
            <name>dfs.namenode.name.dir</name>
            <value>file:///usr/local/hadoop/data/hdfs/datanode/namenode</value>
        </property>
        <property>
            <name>dfs.datanode.data.dir</name>
            <value>file:///usr/local/hadoop/data/hdfs/datanode</value>
        </property>
    </configuration>
    ```

5. Create directory for hdfs hosting (note that this should match with the configuration of the previous step):

    ```bash
    sudo mkdir -p $HADOOP_HOME/data/hdfs/datanode
    ```

6. Set ubuntu as the owner of Hadoop:

    ```bash
    sudo chown -R ubuntu $HADOOP_HOME
    ```

7. And now, instead of doing the same for every data node, you can just run the script below for each of the nodes and have everything set on them:

    ```bash
    clear
    echo "Copying core-site.xml" &&
    ssh datanode1 sudo rm -rf $HADOOP_CONF_DIR/core-site.xml &&
    scp $HADOOP_CONF_DIR/core-site.xml datanode1:~/ &&
    ssh datanode1 sudo mv ~/core-site.xml $HADOOP_CONF_DIR/ &&
    echo "Copying yarn-site.xml" &&
    ssh datanode1 sudo rm -rf $HADOOP_CONF_DIR/yarn-site.xml &&
    scp $HADOOP_CONF_DIR/yarn-site.xml datanode1:~/ &&
    ssh datanode1 sudo mv ~/yarn-site.xml $HADOOP_CONF_DIR/ &&
    echo "Copying mapred-site.xml" &&
    ssh datanode1 sudo rm -rf $HADOOP_CONF_DIR/mapred-site.xml &&
    scp $HADOOP_CONF_DIR/mapred-site.xml datanode1:~/ &&
    ssh datanode1 sudo mv ~/mapred-site.xml $HADOOP_CONF_DIR/ &&
    echo "Copying hdfs-site.xml" &&
    ssh datanode1 sudo rm -rf $HADOOP_CONF_DIR/hdfs-site.xml &&
    scp $HADOOP_CONF_DIR/hdfs-site.xml datanode1:~/ &&
    ssh datanode1 sudo mv ~/hdfs-site.xml $HADOOP_CONF_DIR/ &&
    echo "Creating HDFS folder" &&
    ssh datanode1 sudo mkdir -p $HADOOP_HOME/data/hdfs/datanode &&
    ssh datanode1 sudo mkdir -p $HADOOP_HOME/data/hdfs/namenode &&
    echo "Setting ubuntu as Hadoop owner" &&
    ssh datanode1 sudo chown -R ubuntu $HADOOP_HOME
    echo "Installing Python and mrjob pip package"
    ssh datanode1 sudo apt install python3
    ssh datanode1 sudo apt install python3-pip
    ssh datanode1 pip3 install mrjob
    ```

    **Note**: I'm taking the chance here to install python3, pip3 and mrjob because I'm using them and they need to be installed in each one of the data nodes

## Specific configurations for Name Node (and finally starting Hadoop)

The steps below are the final configuration for Name Node

1. Configure the Name Node host name:

    ```bash
    sudo nano $HADOOP_CONF_DIR/masters
    ```

    Add the host name to the file created above:

    ```<bash>
    <name node host name>
    ```

    **Note:** you can get the host name by executing the statement below:

    ```bash
    echo $(hostname)
    ```

2. Configure the Data Node host names:

    ```bash
    sudo nano $HADOOP_CONF_DIR/slaves
    ```

    Add the host name to the file created above (it is good to SSH each one of them to add them to the list of trust IP addresses):

    ```bash
    localhost # if you want to have data hosted on your Name Node too
    <data node 1 host name>
    <data node 2 host name>
    <data node 3 host name>
    <data node ... host name>
    ```

    **Notes:**

    * you can get the host name by executing the statement below:

        ```bash
        echo $(hostname)
        ```

    * For some unknown reason (another one!), the services were not starting if I add the localhost as one of the data nodes, so, I just stopped using it by removing any reference to localhost from slaves file

3. Format file system:

    ```bash
    hdfs namenode -format
    ```

4. Start HDFS:

    ```bash
    $HADOOP_HOME/sbin/start-dfs.sh
    ```

5. Check HDFS status:

    ```bash
    hdfs dfsadmin -report
    ```

6. Start YARN:

    ```bash
    $HADOOP_HOME/sbin/start-yarn.sh
    ```

7. Start History server:

    ```bash
    $HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
    ```

8. If you want to see what JAVA processes are running:

    ```bash
    jps
    ```

    !["jps"](https://user-images.githubusercontent.com/7594950/108001726-47da7a00-6fbb-11eb-8b3e-e4accd81f872.png)

9. You can check the status of the cluster using the Name Node Web UI:<br>\<Public IPv4 address>:50070 (port need to be available on security groups assigned to the server)

    !["image205"](https://user-images.githubusercontent.com/7594950/108001963-00082280-6fbc-11eb-8e95-4f52539a49ff.png)

This how-to guide is an extension of [Setup 4 Node Hadoop Cluster on AWS EC2 Instances](https://medium.com/@jeevananandanne/setup-4-node-hadoop-cluster-on-aws-ec2-instances-1c1eeb4453bd) article by Jeevan Anand.
