# Installing MongoDB on Ubuntu

## Installing MongoDB and pre-requisites

After login into EC2, perform the following steps:

1. Update apt local package database:

    ```bash
    sudo apt-get update
    ```

2. Install Python (and some Python utilities) on EC2 machine:

    ```bash
    sudo apt install python3 -y
    sudo apt install python3-pip -y
    ```

3. Import the public key used by the package management system:

    ```bash
    cd ~
    mkdir Downloads
    cd Downloads
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
    ```

4. Create a list file for MongoDB:

    ```bash
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
    ```

5. Reload local package database (now with MongoDB):

    ```bash
    sudo apt-get update
    ```

6. Install the MongoDB:

    ```bash
    sudo apt-get install -y mongodb-org
    ```

7. Change configuration to accept external calls

    ```bash
    sudo nano /etc/mongod.conf
    ```

    replace the commented bindIp value 127.0.0.1 by 0.0.0.0

    ```bash
    # network interfaces
    net:
        port: 27017
    #  bindIp: 127.0.0.1  
        bindIp: 0.0.0.0
    ```

8. Start MongoDB service

    ```bash
    sudo systemctl start mongod
    ```

9. Verify if MongoDB service has started successfully.

    ```bash
    service mongod status
    ```

10. To enable MonboDB service to be started at boot:

    ```bash
    sudo systemctl enable mongod.service
    ```

11. Start MongoDB shell

    ```bash
    mongo
    ```

12. Test mongo:

    ```mongodb
    show dbs
    ```

13. Exit Mongo

    ```sql
    quit()
    ```

Source: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/