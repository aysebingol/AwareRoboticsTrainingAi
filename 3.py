import cv2


input_image_name = 'ai.png'

output_image_name = 'kirpilmis_foto.png'

img = cv2.imread(input_image_name)

if img is None:
    print(f"HATA: '{input_image_name}' dosyası bulunamadı")
else:
    #fotoyu boyutlandırma
    resized_img = cv2.resize(img, (400, 300))

    #belirli alanı kırpmak
    x,y,w,h = 50,50,200,150
    cropped_img = resized_img[y:y+h, x:x+w]

    #kaydet yeni görseli
    cv2.imwrite(output_image_name, cropped_img)
    print(f"'{output_image_name}' basariyla kaydedildi")

    #kırpılmış alanı gösterme
    cv2.imshow("yeni alan", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()