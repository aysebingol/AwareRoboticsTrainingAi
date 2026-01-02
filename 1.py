import cv2

#okunacak fotoğraf
input_image_name = 'ai.png'
#kaydedilecek fotoğrafın yeni adı
output_image_name = 'kaydedilmis_foto.png'

#fotoğrafı okur
img = cv2.imread(input_image_name)


if img is None:
    print(f"HATA: '{input_image_name}' dosyası bulunamadı ve okunamadı.")
else:
    //bu görüntüyü(img) bu isimle diske kaydet
    kayit_basarisi = cv2.imwrite(output_image_name,img)



    if kayit_basarisi:
        print(f"'{output_image_name}' başarıyla okundu.")
        print(f"Fotoğraf, '{output_image_name}' olarak aynı dizine kaydedildi.")

    else:
        print(f"Hata: Fotoğraf '{output_image_name}' olarak kaydedilemedi.")   


 
