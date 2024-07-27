from PIL import Image ,ImageFilter

image=Image.open("image.png")
image.show()
image.rotate(180).show()
image.transpose(Image.FLIP_LEFT_RIGHT).show()

rec=608, 505 ,721 ,616
image.crop(rec).show()

image.filter(ImageFilter.EMBOSS).show()

size=500,500
image.thumbnail(size)
image.show()





