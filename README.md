# YouTube Yorum Duygu Analizi Proje Raporu

**Geliştirici:** uabali  
**Tarih:** Aralık 2025

---

Özet

Bu proje, YouTube platformundaki kullanıcı yorumlarını analiz ederek duygu durumlarını (Positive, Negative, Neutral) sınıflandıran derin öğrenme tabanlı bir sistem geliştirmeyi amaçlamaktadır. Proje kapsamında LSTM, Bi-LSTM ve Attention mekanizmalı Bi-LSTM olmak üzere üç farklı mimari sıfırdan eğitilmiş ve performansları karşılaştırılmıştır. Deneysel sonuçlar, **Bi-LSTM** mimarisinin **%66.97 doğruluk** oranı ile en başarılı model olduğunu göstermiştir.

---

## 1. Proje Konusu

### 1.1 Seçilme Gerekçesi
Dijital çağda kullanıcıların sosyal medya platformlarında bıraktığı izler, paha biçilemez bir veri kaynağı oluşturmaktadır. Özellikle YouTube, aylık 2.5 milyar aktif kullanıcısı ile dünyanın en büyük video barındırma platformudur. Bu platformdaki yorumlar, sadece video içeriği hakkında değil, toplumsal eğilimler, ürün algıları ve kültürel tepkiler hakkında derinlemesine bilgi sağlamaktadır. Bu projenin seçilme nedeni, yapılandırılmamış (unstructured) metin verisinden anlamlı ve yapılandırılmış bilgi çıkarımı yaparak NLP (Doğal Dil İşleme) yetkinliklerini derin öğrenme teknikleri ile birleştirmektir.

### 1.2 İlgili Alanda Yapılan Uygulamalar
Duygu analizi alanında yapılan akademik çalışmalar incelendiğinde üç ana yaklaşım göze çarpmaktadır:

1.  **Sözlük Tabanlı (Lexicon-Based) Yaklaşımlar:** Kelimelerin önceden tanımlı duygu skorlarına göre analiz edilmesi (örn. VADER, TextBlob). Basit ancak bağlamı kaçıran yöntemlerdir.
2.  **Geleneksel Makine Öğrenmesi:** Naive Bayes, SVM gibi algoritmalar. *O'Keefe et al. (2018)*, TF-IDF öznitelikleri ile %78 civarında başarı raporlamıştır.
3.  **Derin Öğrenme Yaklaşımları:**
    *   **RNN/LSTM:** *Hochreiter & Schmidhuber (1997)*, uzun vadeli bağımlılıkları çözmek için LSTM'i önermiştir.
    *   **Attention Mechanisms:** *Bahdanau et al. (2014)*, modelin önemli kelimelere odaklanmasını sağlayan dikkat mekanizmasını geliştirmiştir.
    *   **Transformers:** *BERT (Devlin et al., 2019)* ve *RoBERTa*, transfer learning ile %90+ başarı oranlarına ulaşmıştır.

### 1.3 İlgili Alanın Önemi
Bu alanın önemi şu üç ana başlıkta özetlenebilir:
*   **Ticari İstihbarat:** Markaların müşteri memnuniyetini ölçmesi.
*   **Sosyal İzleme:** Nefret söylemi ve siber zorbalığın otomatik tespiti.
*   **İçerik Stratejisi:** YouTuber'ların izleyici beklentilerini veri odaklı anlaması.

---

## 2. Veri Setinin Belirlenmesi

### 2.1 Veri Kaynağı ve İstatistikleri
Projede, Kaggle üzerinden temin edilen ve YouTube API kullanılarak toplanmış geniş kapsamlı bir veri seti kullanılmıştır.

*   **Toplam Veri:** 1,032,225 satır
*   **Kullanılan Örneklem:** Eğitim süresini optimize etmek amacıyla 100,000 rastgele seçilmiş veri kullanılmıştır.
*   **Sınıf Dağılımı (Dengeli):**
    *   Negatif: ~%33.5
    *   Pozitif: ~%33.3
    *   Nötr: ~%33.2

### 2.2 Veri Ön İşleme
Ham veri, modele verilmeden önce aşağıdaki işlemlerden geçirilmiştir:
1.  **Temizlik:** URL'ler, HTML etiketleri, @mention'lar ve #hashtag'ler RegEx ile temizlendi.
2.  **Normalizasyon:** Tüm metin küçük harfe çevrildi.
3.  **Filtreleme:** ASCII dışı karakterler (emojiler hariç) ve noktalama işaretleri kaldırıldı.
4.  **Tokenizasyon:** Özgün bir kelime dağarcığı (Vocabulary) oluşturuldu (Boyut: 27,131 kelime).
5.  **Padding:** Tüm cümleler sabit 128 token uzunluğuna getirildi.

---

## 3. Yöntem ve Seçim Gerekçesi

### 3.1 Yöntem Seçimi ve Karşılaştırmalı Analiz
Bu projede **derin öğrenme (Deep Learning)** yaklaşımı benimsenmiştir. Bunun nedeni, geleneksel yöntemlerin aksine derin öğrenmenin özellik çıkarımı (feature extraction) işlemini otomatik yapması ve anlamsal bağlamı daha iyi yakalamasıdır.

Hazır (pre-trained) modellerin (BERT vb.) kullanılmama nedeni, **öğrenme hedefleri doğrultusunda** LSTM ve Attention mekanizmalarının çalışma mantığının kod düzeyinde kavranması ve sıfırdan bir mimari kurma deneyimidir.

### 3.2 Uygulanan Mimariler

1.  **LSTM (Long Short-Term Memory):**
    *   *Gerekçe:* Standart RNN'lerdeki "vanishing gradient" problemini çözmesi ve cümle içindeki uzun mesafeli kelime ilişkilerini hafızasında tutabilmesi.
    *   *Mimari:* Embedding -> LSTM -> Dropout -> FC -> Softmax

2.  **Bi-LSTM (Bidirectional LSTM):**
    *   *Gerekçe:* Metni hem baştan sona hem de sondan başa okuyarak, bir kelimenin sadece geçmişten değil, gelecekten de bağlam almasını sağlar.
    *   *Mimari:* Embedding -> BiLSTM -> Dropout -> FC -> Softmax

3.  **Bi-LSTM + Attention:**
    *   *Gerekçe:* Tüm cümlenin tek bir vektöre sıkıştırılması yerine, duygu durumunu belirleyen kritik kelimelere (örn. "harika", "berbat") daha fazla ağırlık verilmesini sağlar.

---

## 4. Model Eğitimi & Değerlendirilmesi

### 4.1 Eğitim Konfigürasyonu
Eğitim, **NVIDIA GeForce RTX 5070 Ti** GPU üzerinde gerçekleştirilmiştir.

*   **Loss Function:** CrossEntropyLoss
*   **Optimizer:** Adam (`lr=0.001`)
*   **Batch Size:** 64
*   **Epochs:** 15 (Early Stopping: 5)
*   **Embedding Dimension:** 128

### 4.2 Deneysel Sonuçlar
Modellerin test seti üzerindeki performansları aşağıda özetlenmiştir:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM** | 0.6632 | 0.6666 | 0.6632 | 0.6640 |
| **Bi-LSTM** | **0.6697** | **0.6709** | **0.6697** | **0.6700** |
| **Bi-LSTM + Attention** | 0.6682 | 0.6700 | 0.6682 | 0.6672 |

### 4.3 Sonuçların Tartışılması
1.  **Random Baseline Karşılaştırması:** 3 sınıflı bir problemde rastgele başarı %33.3'tür. Eğitilen modellerimiz **%67** bandına ulaşarak **rastgele tahminden 2 kat daha başarılı** olmuştur.
2.  **Bi-LSTM Üstünlüğü:** Bi-LSTM modeli, en basit LSTM'e göre daha iyi performans göstermiştir. Bu durum, yorumun sonundaki bir ifadenin (örn: "...tavsiye etmem") cümlenin başındaki anlamı değiştirebildiğini ve çift yönlü okumanın avantajını kanıtlar.
3.  **Attention Etkisi:** Attention mekanizması bu veri setinde belirgin bir fark yaratmamıştır. Bunun nedeni, YouTube yorumlarının genellikle kısa olması ve karmaşık dikkat mekanizmasına ihtiyaç duymadan da bağlamın yakalanabilmesi olabilir.

---

## 5. Proje Dokümantasyonu

Proje çıktıları `results/` klasöründe yer almaktadır:
*   **Confusion Matrices:** Her modelin sınıf bazlı hata matrisleri.
*   **Loss/Accuracy Grafikleri:** Eğitim sürecindeki öğrenme eğrileri.
*   **Model Comparison:** Modellerin yan yana performans kıyaslaması.

---
**Telif Hakkı:** Bu proje akademik eğitim amaçlı hazırlanmış olup, kullanılan veri seti ve kodlar açık kaynak prensiplerine uygundur.
