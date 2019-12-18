import hashlib
import sys

def read_file(file):
    """Reads en entire file and returns file bytes."""
    BUFFER_SIZE = 16384 # 16 kilo bytes
    b = b""
    with open(file, "rb") as f:
        while True:
            # read 16K bytes from the file
            bytes_read = f.read(BUFFER_SIZE)
            if bytes_read:
                # if there is bytes, append them
                b += bytes_read
            else:
                # if not, nothing to do here, break out of the loop
                break
    return b

if __name__ == "__main__":
    # read some file
    file_content = read_file(sys.argv[1])
    # some chksums:
    # hash with MD5 (not recommended)
    print("MD5:", hashlib.md5(file_content).hexdigest())

    # hash with SHA-2 (SHA-256 & SHA-512)
    print("SHA-256:", hashlib.sha256(file_content).hexdigest())

    print("SHA-512:", hashlib.sha512(file_content).hexdigest())

    # hash with SHA-3
    print("SHA-3-256:", hashlib.sha3_256(file_content).hexdigest())

    print("SHA-3-512:", hashlib.sha3_512(file_content).hexdigest())

    # hash with BLAKE2
    # 256-bit BLAKE2 (or BLAKE2s)
    print("BLAKE2c:", hashlib.blake2s(file_content).hexdigest())
    # 512-bit BLAKE2 (or BLAKE2b)
    print("BLAKE2b:", hashlib.blake2b(file_content).hexdigest())
