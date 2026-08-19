#qr generation using python


import qrcode

data=input("enter the data to generate qr code:")

qr=qrcode.make(data)
qr.save("qr.png")
qr.show()

print("qr code generated successfully")