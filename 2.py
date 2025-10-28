import cv2

input_image_name = 'ai.png'

output_image_name = 'gri_foto.png'

img = cv2.imread(input_image_name)

if img is None:
    print(f"HATA: '{input_image_name}' dosyası bulunamadı ve okunaamdı")
else:
    #gri tonlamaya dönüştürür
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    
    #gri tonlamalı fotoğrafı kaydet
    kayit_basarisi = cv2.imwrite(output_image_name, gray_img)

    if kayit_basarisi:
        print(f"'{output_image_name}' başarıyla kaydedildi")
    else:
        print(f"HATA: Fotoğraf '{output_image_name}' olarak kaydedilemedi")    