from PIL import Image


image = Image.open('image.png')

x1=[608,836,1073,985,994,1042]
x2=[721,916,1180,1042,1042,1088]
y1=[505,477,600,417,346,282]
y2=[616,557,726,468,383,314]
for i in range(6):
    for x in range(x1[i],x2[i]):
        for y in range(y1[i],y2[i]):
            if((x-x1[i]<5 or x2[i]-x<5)or(y-y1[i]<5 or y2[i]-y<5)):
                image.putpixel((x,y),(255,0,0))
        
                
image.show()
