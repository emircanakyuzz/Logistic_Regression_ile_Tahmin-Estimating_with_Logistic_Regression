Merhaba. Bu çalışmada şirket harcamalarının sınıflandırıldığı bir dataset üzerinde preprocessing işlemlerini gerçekleştirdikten sonra ML algoritmalarından biri olan Logistic Regression kullanarak model eğitiyoruz.
Eğittiğimiz model ile şirketlerin yapacağı harcamaların hangi sınıfa ait olduğunu tahmin ediyoruz. Datasetimizdeki sütunlara göz atalım:
- **company_code:** Şirketlere atanan bir ID adresi olarak düşünebiliriz.
- **document_number:** Şirketlerin gerçekleştirdiği işlem sonrasında oluşturulan belgenin numarası.
- **description:** Gerçekleştirilen işlem ile ilgili kısa bir açıklama.
- **payment_type:** Bu sütun ödeme türünü ifade ediyor. S: Slip benzeri ya da kredi kartı ile ilgili bir kısaltma olabilir. H: Nakitle ilgili bir kısaltma olabilir.
- **amount:** Gerçekleştirilen işlem miktarı.
- **currency_code:** İşlemin hangi para biriminde gerçekleştirildiğini belirtir. 
- **transaction_type:** Bankacılık sektöründeki mesaj formatlarını içeriyor. Örnek: NTAX, vergi ödemelerini ifade etmektedir. 
- **seller_number:** Çıktı sınıfı
- **customer_number:** Çıktı sınıfı
- **main_account:** Çıktı sınıfı
## Adımlar:
1. Output sütunu oluşturma
2. Dataset görselleştirme
3. Eksik veri analizi
4. Kategorik verilerin sayısal verilere çevrimi
5. Aykırı değer analizi ve işlenmesi
6. Korelasyon analizi
7. Verilerin ölçeklenmesi
8. Model geliştirilmesi
9. Performans sonuçları
