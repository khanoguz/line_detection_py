import cv2
import numpy as np
import matplotlib.pyplot as plt

"""renkli bir resim 3 farklı yoğunlukta renk içerir
grayscale resim sadece 1 yoğunlukta renk içerir ve bu görüntünün işlenmesinde daha hızlıdır """

def canny(image):
    gri = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #resmi griye çevirir
    blur = cv2.GaussianBlur(gri,(5,5), 0) #resim üzerindeki gürültüyü kaldırır. ikinci parametresi 5x5 matristir ve çoğu durum için iyi sonuc verir. son parametre sapma(deviyasyon) parametresidir.
    """canny bizim görüntüler üzerindekki çizgileri yakalamamızı saglar. alt ve üst eşik degerleri ile resim üzerindeki farklı edge degerlerine sahip
    çizgileri yakayalabiliriz. parlaklıktaki değişime göre istenilen çizgiler yakalanabilir"""
    cannny = cv2.Canny(blur, 50, 150) #resimdeki tüm yönler üzerinde hesaplamalar(türev) yapar ve beyaz seri üzerindeki en güçlü gradyanı işaretler. yapılması önerilir, yapılmasa da olur.
    return cannny

def istenilen_bolge(image):
    yukseklik = image.shape[0] #listenin kac satır ve kac sutundan olustugunu gösteren bir demet veri tipi döndürür.
    ucgen = np.array([[(300, yukseklik), (1000, yukseklik), (550,250)]]) #boyadıgımız alani birden fazla poligon ile sinirlandırıriz. bu yüzden bu sekilde matris olusturduk
    maske = np.zeros_like(image) #siyah bir maske olusturur.
    cv2.fillPoly(maske, ucgen, 255) #olusturdugumuz maskenin üzerine cizdigimiz ucgen matrisini yerlestiriyoruz. tamamiyle beyaz renk.
    maske_resim = cv2.bitwise_and(image, maske)
    """ sayilari bir bütün olarak değil bit bit işleme sokan operatordür. float ya da double tipinde veriler bu operatorlere kullanılmaz
        genellikle sayilarin bit durumlarını öğrenmek veya diğer bitlere dokunmadan çeşitli bitlerin değerlerini değiştirmek için kullanılır.
        bitwise_and ise karşılıklı bitleri and işlemine tabi tutar.
    """
    return maske_resim

def serit_yakala(image,serit):
    line_image = np.zeros_like(image)
    if serit is not None:
        for line in serit:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2,y2),(255, 0, 0), 10)
    return line_image

def serit_tamamla(image, lines_param):
    slope, intercept = lines_param
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) #sagdaki uzun kenarin 3/5i kadar serit buluyoruz
    x1 = int((y1-intercept)/slope)  # y=mx+b
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])
def ort_egim_poz(image,lines):
    sag_fit = []
    sol_fit = []
    for line in lines:
        x1,y1, x2, y2 = line.reshape(4)
        parametre = np.polyfit((x1,x2), (y1,y2), 1)
        egim = parametre[0]
        pozisyon = parametre[1]
        if egim < 0:
            sol_fit.append((egim, pozisyon))
        else:
            sag_fit.append((egim, pozisyon))

    sag_ortalama = np.average(sag_fit, axis = 0)
    sol_ortalama = np.average(sol_fit, axis = 0)

    sag_serit = serit_tamamla(image, sag_ortalama)
    sol_serit = serit_tamamla(image, sol_ortalama)

    return np.array([sol_serit, sag_serit])
"""
resim = cv2.imread("b.jpg")
serit_resim = np.copy(resim)
cany = canny(serit_resim)
kirpilmis_resim = istenilen_bolge(cany)
aci = np.pi/180
seritler =  cv2.HoughLinesP(kirpilmis_resim, 2, aci, 100, np.array([]), minLineLength=40, maxLineGap=5)
 #ikinci input resulation(cozunurluk değeri. daha kucuk parcalara boldukce çözünürlük artar). en iyi acı değeri yakalanir. biz burda 1 aldik

Hough transform: matematiksel olarka açıklanabilen her şeklin görüntüde boşluklar olsa bile tamamlayabilen bir tekniktir.
Bir çizginin formülü koornidat sisteminde y=mx+c'dir. polar koornidat sisteminde ise rho=x*cos(theta)+y*sin(theta)'dır.
rho ---> çizginin orijine olan dik uzaklığı
theta ---> rho ile x ekseni arasindaki açı.
kademe kademe şöyle işler
1)kaynak görüntü üzerinde kenarlar belli edilir.
2)bir eşikleme yöntemi kullanılarak resim ikili(siyah-beyaz) hale getirilir.
3)her kenar pikseli için noktanın üzerinde olabileceği olası geometrik şekillerin polar kordinattaki değerleri
    kullanılan bir akümülatör matrisi üzerinde birer birer artırılarak her kenarin pikseli olası şekilleri oylaması
    sağlanmış olur.
4)akümülatör değeri en yüksek olan şekiller en çok oy alan şekiller olduklarından görüntü üzerinde bulunma veya
    belirgin olma olasıklıkları en yüksek olmaktadır.

Not:bir çizgi üzerinde ne kadar çok nokta varsa oy sayısı o kadar artar.
"""
"""
ortalanmıs_serit = ort_egim_poz(serit_resim, seritler)
mavi_serit = serit_yakala(serit_resim, ortalanmıs_serit)
#son deger eşik değeri. en az 100 kere oylanması gerekiyor bir kenar yakalamak için.

birlesmis_resim = cv2.addWeighted(serit_resim, 1, mavi_serit, 1, 1)
"""
"""
Python: cv2.addWeighted(src1, alpha, src2, beta, gamma)
src1 --> ilk input dizimiz
alpha --> ilk dizinin agırlık elemani
src2 --> ilk dizimizle ayni boyutta ikinci dizimiz
beta --> ikinci dizinin ağırlık elemani
gamma --> her toplamımıza eklenen sayısal değer
""" """
cv2.imshow('sonuc',birlesmis_resim)
cv2.waitKey(0)
""""""
##################################################
video icin kodlar"""
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    cany = canny(frame)
    kirpilmis_resim = istenilen_bolge(cany)
    aci = np.pi/180
    seritler =  cv2.HoughLinesP(kirpilmis_resim, 2, aci, 100, np.array([]), minLineLength=40, maxLineGap=5)
    ortalanmıs_serit = ort_egim_poz(frame, seritler)
    mavi_serit = serit_yakala(frame, ortalanmıs_serit)
    birlesmis_resim = cv2.addWeighted(frame, 1, mavi_serit, 1, 1)
    cv2.imshow('sonuc',birlesmis_resim)
    cv2.waitKey(1)
