/* '%' means from any where in the world*/
CREATE USER 'python-user'@'%' IDENTIFIED BY 'Password1$';
/* grant all privileges to the user on all databases & tables available in the server */
GRANT ALL ON *.* TO 'python-user'@'%';
FLUSH PRIVILEGES;