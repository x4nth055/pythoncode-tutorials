import qrcode
import sys

data = sys.argv[1]
filename = sys.argv[2]

# generate qr code
img = qrcode.make(data)
# save img to a file
img.save(filename)