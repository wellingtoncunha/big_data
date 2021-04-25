## Interacting with MongoDB using Python

1. Create user

    ```bash
    mongo
    ```

    ```mongodb
    use admin
    db.createUser({ user: "mongoadmin" , pwd: "mongoadmin", roles: ["userAdminAnyDatabase", "dbAdminAnyDatabase", "readWriteAnyDatabase"]})
    ```

1. Install PyMongo Python pip package:

    ```bash
    pip3 install pymongo
    ```

2. Python examples

    * Start Python Shell

    ```bash
    python3
    ```

    * import pymongo package

    ```python
    import pymongo
    ```

    * Create a client object for MongoDB

    ```python
    myclient = pymongo.MongoClient("mongodb://<user>:<password>@localhost:27017/")
    ```

    * Create a database called "mydatabase"

    ```python
    mydb = myclient["mydatabase"]
    ```

    * Return a list of your system's databases<br>
    **Note**: *the new database will not appear because in MongoDB objects exist only if they have content (data) and after persisted*

    ```python
    print(myclient.list_database_names())
    ```

    * Check if "mydatabase" exists<br>
    **Note**: *as the new database is empty and was not yet persisted, the script won't get into "if" clause*

    ```python
    dblist = myclient.list_database_names()
    if "mydatabase" in dblist:
        print("The database exists.")
    ```

    * Create a collection called "customers"<br>
    **Note**: *as the new database is empty and was not yet persisted, objects don't exist yet into MongoDB and the collection is empty*

    ```python
    mycol = mydb["customers"]
    print(mydb.list_collection_names())
    ```

    * Return a list of all collections in your database<br>
    **Note**: *as the new database is empty and was not yet persisted, objects don't exist yet into MongoDB and the collection is empty*

    ```python
    collist = mydb.list_collection_names()
    ```

    * Check if the "customers" collection exists<br>
    **Note**: *as the new database is empty and was not yet persisted, objects don't exist yet into MongoDB and the collection is empty*

    ```python
    if "customers" in collist:
        print("The collection exists.")
    ```

    * Insert a record in the "customers" collection, and return the value of the _id field<br>
    **Note**: *now that the database has content (data), it was persisted, along with the collection, and will be available on the statements above*

    ```python
    mydict = { "name": "John", "address": "Highway 37" }
    x = mycol.insert_one(mydict)
    print(x.inserted_id)
    ```

    * Insert Multiple Documents

    ```python
    mylist = [
    { "name": "Amy", "address": "Apple st 652"},
    { "name": "Hannah", "address": "Mountain 21"},
    { "name": "Michael", "address": "Valley 345"},
    { "name": "Sandy", "address": "Ocean blvd 2"},
    { "name": "Betty", "address": "Green Grass 1"},
    { "name": "Richard", "address": "Sky st 331"},
    { "name": "Susan", "address": "One way 98"},
    { "name": "Vicky", "address": "Yellow Garden 2"},
    { "name": "Ben", "address": "Park Lane 38"},
    { "name": "William", "address": "Central st 954"},
    { "name": "Chuck", "address": "Main Road 989"},
    { "name": "Viola", "address": "Sideway 1633"}
    ]

    x = mycol.insert_many(mylist)
    ```

    * Print the list of the _id values of the inserted documents

    ```python
    print(x.inserted_ids)
    ```

    * Insert Multiple Documents, with Specified IDs. If you do not want MongoDB to assign unique ids for you document, you can specify the _id field when you insert the document(s). Remember that the values has to be unique. Two documents cannot have the same _id.

    ```python
    mylist = [
    { "_id": 1, "name": "John", "address": "Highway 37"},
    { "_id": 2, "name": "Peter", "address": "Lowstreet 27"},
    { "_id": 3, "name": "Amy", "address": "Apple st 652"},
    { "_id": 4, "name": "Hannah", "address": "Mountain 21"},
    { "_id": 5, "name": "Michael", "address": "Valley 345"},
    { "_id": 6, "name": "Sandy", "address": "Ocean blvd 2"},
    { "_id": 7, "name": "Betty", "address": "Green Grass 1"},
    { "_id": 8, "name": "Richard", "address": "Sky st 331"},
    { "_id": 9, "name": "Susan", "address": "One way 98"},
    { "_id": 10, "name": "Vicky", "address": "Yellow Garden 2"},
    { "_id": 11, "name": "Ben", "address": "Park Lane 38"},
    { "_id": 12, "name": "William", "address": "Central st 954"},
    { "_id": 13, "name": "Chuck", "address": "Main Road 989"},
    { "_id": 14, "name": "Viola", "address": "Sideway 1633"}
    ]
    x = mycol.insert_many(mylist)
    ```

    * Print list of the _id values of the inserted documents

    ```python
    print(x.inserted_ids)
    ```

    * Find the first document in the customers collection

    ```python
    x = mycol.find_one()
    print(x)
    ```

    * Return all documents in the "customers" collection, and print each document

    ```python
    for x in mycol.find():
        print(x)
    ```

    * Return only the names and addresses, not the _ids<br>
    **Note**: *note that 0 and 1 are flags for False and True*

    ```python
    for x in mycol.find({},{ "_id": 0, "name": 1, "address": 1 }):
        print(x)
    ```

    * This example will exclude "address" from the result

    ```python
    for x in mycol.find({},{ "address": 0 }):
        print(x)
    ```

    * Find document(s) with the address "Park Lane 38"

    ```python
    myquery = { "address": "Park Lane 38" }
    mydoc = mycol.find(myquery)
    for x in mydoc:
        print(x)
    ```

    * Find documents where the address starts with the letter "S" or higher

    ```python
    myquery = { "address": { "$gt": "S" } }
    mydoc = mycol.find(myquery)
    for x in mydoc:
        print(x)
    ```

    * Find documents where the address starts with the letter "S"

    ```python
    myquery = { "address": { "$regex": "^S" } }
    mydoc = mycol.find(myquery)
    for x in mydoc:
        print(x)
    ```

    * Sort the result alphabetically by name

    ```python
    mydoc = mycol.find().sort("name")
    for x in mydoc:
        print(x)
    ```

    * Use the value -1 as the second parameter to sort descending

    ```python
    mydoc = mycol.find().sort("name", -1)
    for x in mydoc:
        print(x)
    ```

    * Delete the document with the address "Mountain 21"

    ```python
    myquery = { "address": "Mountain 21" }
    mycol.delete_one(myquery)
    ```

    * Delete all documents were the address starts with the letter S

    ```python
    myquery = { "address": {"$regex": "^S"} }
    x = mycol.delete_many(myquery)
    print(x.deleted_count, " documents deleted.")
    ```

    * Change the address from "Valley 345" to "Canyon 123"

    ```python
    myquery = { "address": "Valley 345" }
    newvalues = { "$set": { "address": "Canyon 123" } }
    mycol.update_one(myquery, newvalues)
    #print "customers" after the update:
    for x in mycol.find():
        print(x)
    ```

    * Update all documents where the address starts with the letter "S"

    ```python
    myquery = { "address": { "$regex": "^S" } }
    newvalues = { "$set": { "name": "Minnie" } }
    x = mycol.update_many(myquery, newvalues)
    print(x.modified_count, "documents updated.")
    ```

    * Limit the result to only return 5 documents

    ```python
    myresult = mycol.find().limit(5)
    #print the result:
    for x in myresult:
        print(x)
    ```

    * Delete all documents in the "customers" collection

    ```python
    x = mycol.delete_many({})
    print(x.deleted_count, " documents deleted.")
    ```

    * Delete the "customers" collection<br>
    **Note**: *as the database will be empty it will also no longer exist*

    ```python
    mycol.drop()
    ```