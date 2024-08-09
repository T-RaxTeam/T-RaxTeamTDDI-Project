<h1 align = 'Center'>T-RaxTeam Teknofest 2024 TDDİ Repository</h1>

**Uyarı:**Dosyada İki adet FastApi projesi bulunmaktadır.(İkiside %100 Bize Aittir.)

* Bu Depo hem FastApi hemde Takım İçi Tanıtım Depomuzdur. 

* Veri kazıma işleminde kullandığımız kodumuz ise : https://github.com/T-RaxTeam/T-RaxTeamWebScrapping

------------------------------------------------------------------------------------------------------------------

## PROJEMİZİN AMACI  
X isimli sitede yer alan müşteri yorumlarını analiz ederek, ürünler hakkında detaylı duygu analizi yapmayı amaçlıyoruz. Bu proje, müşteri geri bildirimlerini otomatik olarak sınıflandırarak, işletmelerin ürün ve hizmetlerini iyileştirmelerine yardımcı olmayı hedeflemektedir.

## PROJENİN İŞ AKIŞI 
-Veri Toplama : Müşteri yorumlarının toplanması.  
-Veri Ön İşleme : Yorumların temizlenmesi ve etiketlenmesi.  
-Model Eğitimi : Verilerin kullanılarak modelin eğitilmesi.  
-Model Değerlendirmesi : Modelin performansının ölçülmesi ve iyileştirilmesi.  
-Sonuçların Analizi : Modelin yorumları doğru bir şekilde sınıflandırıp sınıflandıramadığının incelenmesi.  
-Uygulama ve Entegrasyon : Modelin canlı sistemde kullanılması ve test edilmesi.  

## VERİ SETİ
Veri seti, Hepsiburada'dan alınan müşteri yorumlarından oluşmaktadır. Veri seti, temizlenmiş ve önceden etiketlenmiş yorumlardan oluşmaktadır. Veriler, Excel formatında depolanarak hızlı erişim ve işlemeye uygun hale getirilmiştir. Ayrıca
T-Rax Team Hepsiburada Yorum Çekici Sistemi İle de Veri setini geliştirmiş bulunmaktayız.

## YÖNTEM VE TEKNİKLER
Proje, BERT (Bidirectional Encoder Representations from Transformers) modeli kullanılarak gerçekleştirilmiştir. Doğal dil işleme teknikleri arasında tokenizasyon, lemmatizasyon ve sentiment analizi yer almaktadır.

## MODEL EĞİTİMİ VE DEĞERLENDİRME
Model, yorumları beş duygu kategorisine ayıracak şekilde eğitilmiştir. Eğitim sırasında GPU kullanılarak hızlandırma sağlanmıştır. Modelin performansı, doğruluk, precision, recall ve F1-score metrikleri ile değerlendirilmiştir

## SONUÇLAR
Proje kapsamında elde edilen bulgular, modelin yüksek doğruluk oranı ile müşteri yorumlarını doğru bir şekilde sınıflandırabildiğini göstermiştir. İşletmeler, bu analizler sayesinde müşteri memnuniyetsizliklerini hızlı bir şekilde tespit edebilir ve gerekli önlemleri alabilir.

## PROJE YOL HARİTASI
Gelecekte, modelin daha büyük veri setleri üzerinde eğitilmesi ve farklı büyük şirket platformlarına uygulanması planlanmaktadır. Ayrıca, modelin duygu analizine ek olarak, öneri sistemleri ve müşteri profilleme gibi ek özellikler eklenmesi düşünülmektedir. Modelimizi geliştirmek istemekteyiz.

----------------------------------------------------------------------------------------------------------
## TAKIM ÜYELERİ VE GÖREV DAĞILIMLARI

**Hüseyin Can Biçer** : Veri Seti ve Veri Tabanı Hazırlama - **Kaptan**  

**Arif Dalyan** : Yazılım ve Geliştirme - **Kaptan Yardımcısı**  

**Hüseyin Özdemir** : Makine Öğrenimi - **Üye**

**Remzi Efe Karakuş** : Yapay Zeka Geliştirme - **Üye**

**Bülent Karaatlı** - **Danışman**

----------------------------------------------------------------------------------------------------------

**Yarışmaya Katılacak Arkadaşlara Başarılar Dileriz.**

















