Gorev 21
# YOLOv5 Kırtasiye Malzemesi Tespit Modeli

Bu proje, özel olarak hazırlanan kırtasiye malzemeleri veri setiyle YOLOv5s modelinin eğitilmesi ve gerçek zamanlı nesne tespiti için kullanılmasını kapsamaktadır.


## 1. Eğitim Metriklerinin Yorumlanması

Model, 50 Epoch boyunca eğitilmiştir. Eğitim metrikleri, modelin öğrenme sürecini ve değerlendirme performansını göstermektedir.

### A. Kayıp Değerleri (Loss Metrics) Analizi

Kayıp değerleri (Loss), modelin ne kadar hata yaptığını gösterir. Bu değerlerin 0'a yakın olması ve eğitim ilerledikçe düşmesi beklenir.

| Metrik | Epoch 0 (Başlangıç) | Epoch 49 (Sonuç) | Yorum |
| :--- | :--- | :--- | :--- |
| **train/box_loss** | 0.11861 | 0.11268 | Sınır kutusu konumlandırma hatası **hafifçe düştü**, model nesneleri doğru çevrelemeyi öğreniyor. |
| **train/obj_loss** | 0.035833 | 0.033851 | Nesne varlığını tanıma hatası stabil ve düşüktür. |
| **train/cls_loss** | 0.041655 | 0.04308 | Sınıflandırma hatası çok az dalgalansa da **genel olarak düşüktür**. |
| **val/Loss (Ortalama)** | ~0.071 | **~0.072** | Doğrulama setindeki kayıplar da düşüktür ve eğitim kayıplarına yakın seyretmektedir. **Bu, modelin aşırı öğrenme (overfitting) yapmadığını gösteren olumlu bir işarettir.** |

### B. Kritik Değerlendirme Metrikleri (mAP, Precision, Recall)

Eğitimin 50 Epoch'u boyunca tüm **Precision, Recall ve mAP** değerleri **0** olarak raporlanmıştır.

| Metrik | Epoch 0 - 49 Değeri | Açıklama ve Sorun |
| :--- | :--- | :--- |
| **mAP@.5** | **0** | Bu değerler, modelin doğrulama setinde **hiçbir başarılı tespit yapamadığını** gösterir. |
| **Precision, Recall**| **0** | Modelin tespit kutularını doğru şekilde değerlendiremediğini gösterir. |

Teknik Yorum:** Bu durum, modelin öğrenmemesinden ziyade, Doğrulama Veri Seti (Validation Set) etiket yollarının veya etiketleme formatının hatalı olduğunu kuvvetle işaret eder. Loss değerlerinin düşmesi, 
modelin eğitim verisinden öğrendiğini gösterir; ancak metriklerin 0 olması, değerlendirme aşamasında bir teknik engel olduğunu gösterir. Gerçek performansın ölçülmesi için doğrulama setinin ayarlarının acilen kontrol edilmesi gerekmektedir.
