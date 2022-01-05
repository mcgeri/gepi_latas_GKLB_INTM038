from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import math
import numpy as np #tomb
import argparse #parancssor
import glob #pattern
import cv2 #OpenCV képkezeléshez

# argumentum konstruktor
#https://docs.python.org/3/library/argparse.html
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

#imread kep beolvasas filebol
#https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
image = cv2.imread(args["image"])

#keparany tartassal meretezes fix meretre
#https://note.nkmk.me/en/python-opencv-pillow-image-size/
#https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
d = 1024 / image.shape[1] #d = 1024/ kep meretenek a magassagaval.
dim = (1024, int(image.shape[0] * d)) # dim = 1024,szelesseg szor a d
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA) #kep, kep merete es viszintes tengely lepteke a dim, újramintavételezés pixel terület relációval. Előnyös módszer lehet a képtizedelésre, mivel moire-mentes eredményeket ad
# print (d)
# print (dim)

#eredeti megjelniteshez lemasolni
output = image.copy()

#szurke aranyalatosra konvertalas
#https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/
#https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
#gepi latas problemainak megoldasa. Sz
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Szurke kep", gray)

#kontraszt javitas a fenyviszonyok kozotti kulonbsegekkel
#clachbe belerak a kontraszt kepet a hisztorgram alkalmazasahoz ?????
#https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html#gad689d2607b7b3889453804f414ab1018
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #kontraszt koszoberteke, racs merete
gray = clahe.apply(gray)
# cv2.imshow("Szurke kep javitva", gray)

#function letrehozas. Bemeno parameter az img
def calcHistogram(img):
    #maszk letrehozas
    m = np.zeros(img.shape[:2], dtype="uint8") #img dimenziojanak elso ket eleme, tipus unassigned int 8 bit. 255on feluli ertekek levagasa
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2)) #masodik es elso ertek osztva kettovel
    #https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
    cv2.circle(m, (w, h), 60, 255, -1) #koroket rajzol a kepre. m a kep, a kor kozeppontja, 60 radius, szin, korvonal vastagsaga. -1 a kor alakja 

    #calcHist-nek atad kep, szin csatornak, maszk, felbontas, tartomany teljes mivel 0,256
    #tomkeszletnek a hisztogramjat szmaolja ki
    #https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    #normalizalt lapitott histogram
    #https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd
    return cv2.normalize(h, h).flatten()


def calcHistFromFile(file):
    img = cv2.imread(file)
    return calcHistogram(img)


#enum, tuple jelenti, hogy tobb ertek egy valtozoba
class Enum(tuple): __getattr__ = tuple.index

#rez, sragarez
#material feltoltese
Material = Enum(('Copper', 'Brass', 'Euro1', 'Euro2'))

#pelda kepek helye. glob minden fajlt megtalal az adott mapban a * patternel. 
#https://docs.python.org/3/library/glob.html
sample_images_copper = glob.glob("sample_images/copper/*")
sample_images_brass = glob.glob("sample_images/brass/*")
sample_images_euro1 = glob.glob("sample_images/euro1/*")
sample_images_euro2 = glob.glob("sample_images/euro2/*")

# define training data and labels
X = []
y = []

#adatok es feliratok tarolasa for cuklissal
#sample_images elemein vegig megy es atadja a calchist-nek majd az eredmenyet xhez fuzi. y-ba pedig belerakja a marail enumot (sorszamat)
for i in sample_images_copper:
    X.append(calcHistFromFile(i))
    y.append(Material.Copper)
for i in sample_images_brass:
    X.append(calcHistFromFile(i))
    y.append(Material.Brass)
for i in sample_images_euro1:
    X.append(calcHistFromFile(i))
    y.append(Material.Euro1)
for i in sample_images_euro2:
    X.append(calcHistFromFile(i))
    y.append(Material.Euro2)

# print(sample_images_copper)
# print(X)
# print(y)

#osztalyozo eredmenye
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
# lbfgs Newton modszerbol szarmazo optimalizalo.
# nyilt forraskodu gepi tanulas modulok adat elemzeshez
clf = MLPClassifier(solver="lbfgs")

#trening es teszt adatok felosztasa
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#tombok felosztasa veletlenszeru test es trening adatokra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#trening es eredmenyek osztalyozasa
#.fit-el illeszti az adat matrix x-et az y celokra
#.score az atlagos pontossag a tesztadatokon
clf.fit(X_train, y_train)
score = int(clf.score(X_test, y_test) * 100)
print("A vizsgalat pontossaga: ", score)

#kep elhomalyositasa gauss szuro hasznaltaval
#bemeno a kep, 7x7-es kernel, az automatikus erzekelshez 0. Ilyenkor a kerenlbol szamolodik a sigma x es y
#https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

#kor vektorba taroljuk, amit eremnek gondol 3 paramaterrel. x,y,r. Kozeppont es sugar. 
#blurred a bementi szurke kep, metodus hough gradeint az elek gradiens infoit hasznalja, dp inverz ratio 1-nel egyforma a bementivel,de 2-nel fele akkora mar
#min_dist tavolsag a kozeppontok kozott, param_1,param_2 canny eldetektalonak atadott ertekek, min_radius, max_radius sugar meretek.
#https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                           param1=200, param2=100, minRadius=50, maxRadius=120)

#anyag felismerese, tipus joslas majd visszadas
def predictMaterial(roi):
    hist = calcHistogram(roi)
    s = clf.predict([hist]) #elorejelzes a tobb retego perceptron osztalyozo segitsegevel
    return Material[int(s)]

diameter = []
materials = []
coordinates = []

count = 0
if circles is not None:
    #sugar hozzafuzes
    for (x, y, r) in circles[0, :]:
        diameter.append(r)

    #intre alakit a kordinatakat
    circles = np.round(circles[0, :]).astype("int")

    
    for (x, y, d) in circles:
        count += 1
        #listahoz adas
        coordinates.append((x, y))

        #nem szukseges terulet kivon
        roi = image[y - d:y + d, x - d:x + d]

        #anyagok
        material = predictMaterial(roi)
        materials.append(material)

        #felismert ermek fileba irasa
        if False:
            m = np.zeros(roi.shape[:2], dtype="uint8")
            w = int(roi.shape[1] / 2)
            h = int(roi.shape[0] / 2)
            cv2.circle(m, (w, h), d, (255), -1)
            maskedCoin = cv2.bitwise_and(roi, roi, mask=m)
            cv2.imwrite("extracted/01coin{}.png".format(count), maskedCoin)

        #berajzol az eredmeny es kiir a kepre
        cv2.circle(output, (x, y), d, (0, 255, 0), 2)
        cv2.putText(output, material,
                    (x - 40, y), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

#legnagyobb erme
biggest = max(diameter)
i = diameter.index(biggest)

#atemeretez a legnagyobb aranyaban
#meretek beallitasa, ha nincs akkor unable
if materials[i] == "Euro2":
    diameter = [x / biggest * 25.75 for x in diameter]
    scaledTo = "2 Euro-hoz meretezve"
elif materials[i] == "Brass":
    diameter = [x / biggest * 24.25 for x in diameter]
    scaledTo = "50 Cent-hez meretezve"
elif materials[i] == "Euro1":
    diameter = [x / biggest * 23.25 for x in diameter]
    scaledTo = "1 Euro-hoz meretezve"
elif materials[i] == "Copper":
    diameter = [x / biggest * 21.25 for x in diameter]
    scaledTo = "5 Cent-hez meretezve"
else:
    scaledTo = "Nem meretezheto.."

i = 0
total = 0
while i < len(diameter):
    d = diameter[i]
    m = materials[i]
    (x, y) = coordinates[i]
    t = "Ismeretlen"

    #osszehasonlitas a hibahatar novelesevel
    if math.isclose(d, 25.75, abs_tol=1.25) and m == "Euro2":
        t = "2 Euro"
        total += 200
    elif math.isclose(d, 23.25, abs_tol=2.5) and m == "Euro1":
        t = "1 Euro"
        total += 100
    elif math.isclose(d, 19.75, abs_tol=1.25) and m == "Brass":
        t = "10 Cent"
        total += 10
    elif math.isclose(d, 22.25, abs_tol=1.0) and m == "Brass":
        t = "20 Cent"
        total += 20
    elif math.isclose(d, 24.25, abs_tol=2.5) and m == "Brass":
        t = "50 Cent"
        total += 50
    elif math.isclose(d, 16.25, abs_tol=1.25) and m == "Copper":
        t = "1 Cent"
        total += 1
    elif math.isclose(d, 18.75, abs_tol=1.25) and m == "Copper":
        t = "2 Cent"
        total += 2
    elif math.isclose(d, 21.25, abs_tol=2.5) and m == "Copper":
        t = "5 Cent"
        total += 5

    #eredmeny kiirasa
    cv2.putText(output, t,
                (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN,
                1.5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    i += 1

d = 768 / output.shape[1]
dim = (768, int(output.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

#felirat a kepre irasa
cv2.putText(output, scaledTo,
            (5, output.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(output, "Erme felismerve: {} db, EUR {:2}".format(count, total / 100),
            (5, output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(output, "A vizsgalat pontossaga: {}%".format(score),
            (5, output.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)

#megjelenit es billentyure var kilepeshez
cv2.imshow("Output", np.hstack([image, output]))
cv2.waitKey(0)
cv2.destroyAllWindows()

