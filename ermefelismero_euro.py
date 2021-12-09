from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import math
import numpy as np #tomb
import argparse #parancssor
import glob #pattern
import cv2 #OpenCV képkezeléshez

# argumentum konstruktor
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

#keparany tartassal meretezes fix meretre
d = 1024 / image.shape[1]
dim = (1024, int(image.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

#eredeti megjelniteshez lemasolni
output = image.copy()

#szurke aranyalatosra konvertalas
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#kontraszt javitas a fenyviszonyok kozotti kulonbsegekkel
#clachbe belerak a kontraszt kepet a hisztorgram alkalmazasahoz
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)


def calcHistogram(img):
    #maszk letrehozas
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 60, 255, -1)

    #calcHist-nek atad kep, szin csatornak, maszk, felbontas, tartomany
    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    #normalizalt lapitott histogram
    return cv2.normalize(h, h).flatten()


def calcHistFromFile(file):
    img = cv2.imread(file)
    return calcHistogram(img)


#enum
class Enum(tuple): __getattr__ = tuple.index

#rez, sragarez
Material = Enum(('Copper', 'Brass', 'Euro1', 'Euro2'))

#pelda kepek helye
sample_images_copper = glob.glob("sample_images/copper/*")
sample_images_brass = glob.glob("sample_images/brass/*")
sample_images_euro1 = glob.glob("sample_images/euro1/*")
sample_images_euro2 = glob.glob("sample_images/euro2/*")

# define training data and labels
X = []
y = []

#adatok es feliratok tarolasa for cuklissal
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

#osztalyozo eredmenye
clf = MLPClassifier(solver="lbfgs")

#trening es teszt adatok felosztasa
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2)

#trening es eredmenyek osztalyozasa
clf.fit(X_train, y_train)
score = int(clf.score(X_test, y_test) * 100)
print("A vizsgalat pontossaga: ", score)

#blurrel Gaussia elmosas szurke aranyalatos
#7x7-es kernel, az automatikus erzekelshez 0
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

#kor vektorba taroljuk, amit eremnek gondol 3 paramaterrel. x,y,r. Kozeppont es sugar. 
#blurred a bementi szurke kep, metodus, dp inverz ratio, min_dist tavolsag a kozeppontok kozott, param_1,param_2 canny erzekelo ertekei, min_radius, max_radius sugar meretek.
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                           param1=200, param2=100, minRadius=50, maxRadius=120)

#anyag felismerese, tipus joslas majd visszadas
def predictMaterial(roi):
    hist = calcHistogram(roi)
    s = clf.predict([hist])
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
    scaledTo = "Scaled to 2 Euro"
elif materials[i] == "Brass":
    diameter = [x / biggest * 24.25 for x in diameter]
    scaledTo = "Scaled to 50 Cent"
elif materials[i] == "Euro1":
    diameter = [x / biggest * 23.25 for x in diameter]
    scaledTo = "Scaled to 1 Euro"
elif materials[i] == "Copper":
    diameter = [x / biggest * 21.25 for x in diameter]
    scaledTo = "Scaled to 5 Cent"
else:
    scaledTo = "unable to scale.."

i = 0
total = 0
while i < len(diameter):
    d = diameter[i]
    m = materials[i]
    (x, y) = coordinates[i]
    t = "Unknown"

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
cv2.putText(output, "Coins detected: {}, EUR {:2}".format(count, total / 100),
            (5, output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(output, "Classifier mean accuracy: {}%".format(score),
            (5, output.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)

#megjelenit es billentyure var kilepeshez
cv2.imshow("Output", np.hstack([image, output]))
cv2.waitKey(0)

