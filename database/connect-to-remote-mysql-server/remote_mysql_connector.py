import mysql.connector as mysql

# enter your server IP address/domain name
HOST = "x.x.x.x" # or "domain.com"
# database name, if you want just to connect to MySQL server, leave it empty
DATABASE = "database"
# this is the user you create
USER = "python-user"
# user password
PASSWORD = "Password1$"
# connect to MySQL server
db_connection = mysql.connect(host=HOST, database=DATABASE, user=USER, password=PASSWORD)
print("Connected to:", db_connection.get_server_info())

# enter your code here!