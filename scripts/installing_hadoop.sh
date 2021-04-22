scp -i ~/Downloads/hadoop.pem ~/Downloads/hadoop.pem ubuntu@107.21.64.46:~/.ssh/ 

ssh -i ~/Downloads/hadoop.pem ubuntu@107.21.64.46


ip-172-31-27-123 hadoop-master
ip-172-31-21-59 hadoop-slave-1
ip-172-31-19-189 hadoop-slave-2
ip-172-31-22-180 hadoop-slave-3



Host namenode
    HostName ip-172-31-27-123
    User ubuntu
    IdentityFile ~/.ssh/hadoop.pem    
Host datanode1
    HostName ip-172-31-21-59
    User ubuntu
    IdentityFile ~/.ssh/hadoop.pem
Host datanode2
    HostName ip-172-31-19-189
    User ubuntu
    IdentityFile ~/.ssh/hadoop.pem
Host datanode3
    HostName ip-172-31-22-180
    User ubuntu
    IdentityFile ~/.ssh/hadoop.pem

<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://ip-172-31-27-123:9000</value>
    </property>
</configuration>

<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.resourcemanager.hostname</name>
        <value>ip-172-31-27-123</value>
    </property>
</configuration>

<configuration>
    <property>
        <name>mapreduce.jobtracker.address</name>
        <value>ip-172-31-27-123:54311</value>
    </property>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>

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

clear
echo "Copying core-site.xml" &&
ssh datanode2 sudo rm -rf $HADOOP_CONF_DIR/core-site.xml &&
scp $HADOOP_CONF_DIR/core-site.xml datanode2:~/ &&
ssh datanode2 sudo mv ~/core-site.xml $HADOOP_CONF_DIR/ &&
echo "Copying yarn-site.xml" &&
ssh datanode2 sudo rm -rf $HADOOP_CONF_DIR/yarn-site.xml &&
scp $HADOOP_CONF_DIR/yarn-site.xml datanode2:~/ &&
ssh datanode2 sudo mv ~/yarn-site.xml $HADOOP_CONF_DIR/ &&
echo "Copying mapred-site.xml" &&
ssh datanode2 sudo rm -rf $HADOOP_CONF_DIR/mapred-site.xml &&
scp $HADOOP_CONF_DIR/mapred-site.xml datanode2:~/ &&
ssh datanode2 sudo mv ~/mapred-site.xml $HADOOP_CONF_DIR/ &&
echo "Copying hdfs-site.xml" &&
ssh datanode2 sudo rm -rf $HADOOP_CONF_DIR/hdfs-site.xml &&
scp $HADOOP_CONF_DIR/hdfs-site.xml datanode2:~/ &&
ssh datanode2 sudo mv ~/hdfs-site.xml $HADOOP_CONF_DIR/ &&
echo "Creating HDFS folder" &&
ssh datanode2 sudo mkdir -p $HADOOP_HOME/data/hdfs/datanode &&
ssh datanode2 sudo mkdir -p $HADOOP_HOME/data/hdfs/namenode &&
echo "Setting ubuntu as Hadoop owner" &&
ssh datanode2 sudo chown -R ubuntu $HADOOP_HOME
echo "Installing Python and mrjob pip package"
ssh datanode2 sudo apt install python3
ssh datanode2 sudo apt install python3-pip
ssh datanode2 pip3 install mrjob

clear
echo "Copying core-site.xml" &&
ssh datanode3 sudo rm -rf $HADOOP_CONF_DIR/core-site.xml &&
scp $HADOOP_CONF_DIR/core-site.xml datanode3:~/ &&
ssh datanode3 sudo mv ~/core-site.xml $HADOOP_CONF_DIR/ &&
echo "Copying yarn-site.xml" &&
ssh datanode3 sudo rm -rf $HADOOP_CONF_DIR/yarn-site.xml &&
scp $HADOOP_CONF_DIR/yarn-site.xml datanode3:~/ &&
ssh datanode3 sudo mv ~/yarn-site.xml $HADOOP_CONF_DIR/ &&
echo "Copying mapred-site.xml" &&
ssh datanode3 sudo rm -rf $HADOOP_CONF_DIR/mapred-site.xml &&
scp $HADOOP_CONF_DIR/mapred-site.xml datanode3:~/ &&
ssh datanode3 sudo mv ~/mapred-site.xml $HADOOP_CONF_DIR/ &&
echo "Copying hdfs-site.xml" &&
ssh datanode3 sudo rm -rf $HADOOP_CONF_DIR/hdfs-site.xml &&
scp $HADOOP_CONF_DIR/hdfs-site.xml datanode3:~/ &&
ssh datanode3 sudo mv ~/hdfs-site.xml $HADOOP_CONF_DIR/ &&
echo "Creating HDFS folder" &&
ssh datanode3 sudo mkdir -p $HADOOP_HOME/data/hdfs/datanode &&
ssh datanode3 sudo mkdir -p $HADOOP_HOME/data/hdfs/namenode &&
echo "Setting ubuntu as Hadoop owner" &&
ssh datanode3 sudo chown -R ubuntu $HADOOP_HOME
echo "Installing Python and mrjob pip package"
ssh datanode3 sudo apt install python3
ssh datanode3 sudo apt install python3-pip
ssh datanode3 pip3 install mrjob

<configuration>
<property>
<name>dfs.replication</name>
<value>1</value>
</property>
<property>
<name>dfs.namenode.name.dir</name>
<value>/home/hduser/hadoop_tmp/hdfs/namenode</value>
</property>
<property>
<name>dfs.datanode.data.dir</name>
<value>/home/hduser/hadoop_tmp/hdfs/datanode</value>
</property>
</configuration>

sudo nano /etc/hosts

sudo mkdir -p $HADOOP_HOME/data/hdfs/namenode

ip-172-31-27-123

ip-172-31-21-59
ip-172-31-19-189
ip-172-31-22-180

ssh ip-172-31-21-59
ssh ip-172-31-19-189
ssh ip-172-31-22-180

107.21.64.46:50070


$HADOOP_HOME/sbin/stop-all.sh

https://na01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fkprgr49g.r.us-east-1.awstrack.me%2FL0%2Fhttps%3A%252F%252Fcovidvaccine.nj.gov%252Ffollow-up-vaccine%252F%253Fid%3D454fc132-0b97-eb11-8ced-0003ff00ac2c%2F1%2F02000000n3jqugvb-01d50s2r-3aja-62kf-4fkl-1sk0fstasug0-000000%2Fn-HEMt36gz45R5zZk3U-Y-BwhnQ%3D210&data=04%7C01%7C%7Cc6e3c5ad669a41eead9308d9032c4c3b%7C84df9e7fe9f640afb435aaaaaaaaaaaa%7C1%7C0%7C637544311050767631%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C1000&sdata=d77Wh9rBXkLtkUwbJVMRqDAkjEPto0%2B%2BxZSWEhBxRmM%3D&reserved=0



# Update default security group (or create a new one) group to allow port 22

# Provision the server: on the very first time, the PEM key might need to be created. For the second time on the same PEM can be used

# Set up the security of PEM file
sudo chmod 600 "/Users/wcunha69/OneDrive/NJIT/02 CS644 Introduction to Big Data/hadoop/hadoop.pem" 

# Connect to EC2 server:
ssh -i ~/Downloads/hadoop.pem ubuntu@54.91.219.20

# Update apt
sudo apt-get update && sudo apt-get dist-upgrade -y

# Install Utilities
sudo apt install nano -y

# Install JAVA
sudo apt-get install openjdk-8-jdk -y

# Download Hadoop
wget https://archive.apache.org/dist/hadoop/core/hadoop-2.8.1/hadoop-2.8.1.tar.gz -P ~/Downloads

# extract to /usr_local
sudo tar zxvf ~/Downloads/hadoop-2.8.1.tar.gz -C /usr/local

# Remove installation file
rm ~/Downloads/hadoop-2.8.1.tar.gz 

# Rename Hadoop directory
sudo mv /usr/local/hadoop-2.8.1 /usr/local/hadoop

# Update .bashrc profile 
nano ~/.bashrc

# JAVA configurations
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

#Hadoop Related Options
export HADOOP_HOME=/usr/local/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop

source ~/.bashrc

# Update .profile
nano ~/.profile

# JAVA configurations
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

#Hadoop Related Options
export HADOOP_HOME=/usr/local/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop

source ~/.profile

# Update /etc/profile
sudo nano /etc/profile

# JAVA configurations
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

#Hadoop Related Options
export HADOOP_HOME=/usr/local/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop

# Apply the changes made to the shell configuration  
source /etc/profile

# Update /etc/bash.bashrc
sudo nano /etc/bash.bashrc

# JAVA configurations
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

#Hadoop Related Options
export HADOOP_HOME=/usr/local/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop


hard code JAVA_HOME (/usr/lib/jvm/java-8-openjdk-amd64) to hadoop-env.sh:
sudo nano hadoop-env.sh

# export HADOOP_HOME="/app/hadoop/hadoop-2.6.5"
# export PATH="$PATH:$HADOOP_HOME/bin"
# export PATH="$PATH:$HADOOP_HOME/sbin"
# export HADOOP_COMMON_HOME=$HADOOP_HOME
# export HADOOP_HDFS_HOME=$HADOOP_HOME
# export YARN_HOME=$HADOOP_HOME
# export HADOOP_YARN_HOME=$HADOOP_HOME
# export HADOOP_MAPRED_HOME=$HADOOP_HOME
# export HADOOP_CONF_DIR="$HADOOP_HOME/etc/hadoop"
# export HADOOP_COMMON_LIB_NATIVE_DIR="$HADOOP_HOME/lib/native"


# Create a image

https://medium.com/@jeevananandanne/setup-4-node-hadoop-cluster-on-aws-ec2-instances-1c1eeb4453bd


# Copy PEM key to master node
scp -i "/Users/wcunha69/OneDrive/NJIT/02 CS644 Introduction to Big Data/hadoop/hadoop.pem" "/Users/wcunha69/OneDrive/NJIT/02 CS644 Introduction to Big Data/hadoop/hadoop.pem" ubuntu@35.153.78.245:~/.ssh/ 
scp -i ~/Downloads/hadoop.pem ~/Downloads/hadoop.pem ubuntu@54.208.68.58:~/.ssh/ 

# Connect to EC2 server:
ssh -i "/Users/wcunha69/OneDrive/NJIT/02 CS644 Introduction to Big Data/hadoop/hadoop.pem" ubuntu@35.153.78.245
ssh -i ~/Downloads/hadoop.pem ubuntu@54.208.68.58

# Create SSH config file for easy access to nodes from master 
nano ~/.ssh/config

# Add the following to the file, configuring the name of the key and the internal ip addresses
Host datanode1
  HostName  172.31.62.146
  User ubuntu
  IdentityFile ~/.ssh/hadoop.pem
Host datanode2
  HostName 172.31.54.190
  User ubuntu
  IdentityFile ~/.ssh/hadoop.pem
Host datanode3
  HostName 172.31.50.95
  User ubuntu
  IdentityFile ~/.ssh/hadoop.pem

# Generate public key
ssh-keygen -f ~/.ssh/id_rsa -t rsa -P ""
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Copy autorized_keys to nodes
ssh datanode1 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub
ssh datanode2 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub
ssh datanode3 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub

### Repeat the steps below for each of the nodes (including name node). For IP address, use the internal, because the external IP
# What about changing it on namenode and just copy to datanodes?

# log into the datanode
ssh datanode1 # or datanode2 or whatever the name you gave on the config file

# Config core-site.xml
cd $HADOOP_CONF_DIR
sudo nano core-site.xml

# Add the following statement to the file (replacing the configuration keys)
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://172.31.60.212:9000</value>
  </property>
</configuration>

# Config yarn-site.xml
cd $HADOOP_CONF_DIR
sudo nano yarn-site.xml

# Add the following statement to the file (replacing the configuration keys)
<configuration>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>172.31.60.212</value>
  </property>
</configuration>

# Config mapred-site.xml
sudo cp mapred-site.xml.template mapred-site.xml
sudo nano mapred-site.xml

# Add the following statement
<configuration>
  <property>
    <name>mapreduce.jobtracker.address</name>
    <value>172.31.60.212:54311</value>
  </property>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
</configuration>

## End of node configuration

### Aditional configuration for the name node
# Add nodes to /etc/hosts - you can use echo $(hostname) on each node to get the ip
sudo nano /etc/hosts

# Add the following to the file
ip-172-31-57-112 namenode_hostname
ip-172-31-55-23 datanode1_hostname
ip-172-31-50-149 datanode2_hostname
ip-172-31-50-83 datanode3_hostname
127.0.0.1 localhost

# Reboot the machine to make effect thru VPCx console or
sudo reboot

# Edit hdfs-site.xml
sudo nano hdfs-site.xml

# Add the following to the file
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>3</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file:///usr/local/hadoop/data/hdfs/namenode</value>
  </property>
</configuration>

# Create a directory to store name node data
sudo mkdir -p $HADOOP_HOME/data/hdfs/namenode

# Set secondary name node, in this case, the same one - you can use echo $(hostname)
sudo nano $HADOOP_CONF_DIR/masters

# Add the following to the file
ip-172-31-60-212 (or you can use the alias on host: namenode_hostname)

# Set slaves 
sudo nano slaves 

# add the aliases or the $(hostname)
localhost
ip-172-31-60-68
ip-172-31-57-168
ip-172-31-61-253


# Finally change the owner of HADOOP_HOME
sudo chown -R ubuntu $HADOOP_HOME

## End of name node configuration


### Aditional configuration for the data nodes (Repeat the steps below for each of the data nodes )
# Config hdfs-site.xml
cd $HADOOP_CONF_DIR
sudo nano hdfs-site.xml

# Add the following content
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>3</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///usr/local/hadoop/data/hdfs/datanode</value>
  </property>
</configuration>

#create the directory for the data
sudo mkdir -p $HADOOP_HOME/data/hdfs/datanode

# Set ubuntu as owner of hadoop
sudo chown -R ubuntu $HADOOP_HOME

## End of data node configuration

# Formating the file system
hdfs namenode -format

# Start HDFS
$HADOOP_HOME/sbin/start-dfs.sh

#check status
hdfs dfsadmin -report

# Start YARN
$HADOOP_HOME/sbin/start-yarn.sh


54.208.68.58:50070
#######################################################

