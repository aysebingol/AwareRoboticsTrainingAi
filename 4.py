import cv2
import numpy as np

input_image_name = 'ai.png'

output_image_name = 'sari_alan.png'

img = cv2.imread(input_image_name)

if img is None:
    print(f"HATA: '{input_image_name}' dosyası bulunamadı")
else:
     resized_img = cv2.resize(img, (400, 300))

     x,y,w,h = 50,50,200,150

     yellow = (0, 255, 255)

    
    #çerçece çizme
     cv2.rectangle(resized_img, (x,y), (x+w, y+h), yellow, 2)

    #alanı doldur
     cv2.rectangle(resized_img, (x,y), (x+w, y+h), yellow, -1)


#kaydetme
cv2.imwrite(output_image_name, resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows