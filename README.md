# CREDIT_CARD_FRAUD_DETECTION_DU
 
Duboko učenje i neuronske mreže

## 1. Opis problema

Link za dataset: [Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Cilj projekta je da pomoću neuronske mreže prepoznamo da li je transakcija kreditnom karticom regularna ili predstavlja prevaru.

U pitanju je binarna klasifikacija:

- `0` – regularna transakcija
- `1` – prevara

Najveći problem je neuravnoteženost klasa. 

## 2. Podaci (izvor, struktura, analiza i preprocesiranje)

### Izvor podataka

Za realizaciju projekta korišćen je javno dostupan skup podataka **Credit Card Fraud Detection**, preuzet sa platforme **Kaggle**. Podaci su sačuvani u fajlu `creditcard.csv`, koji se učitava direktno u Google Colab okruženju.

Dataset je namenjen problemu binarne klasifikacije, odnosno prepoznavanju da li je određena transakcija regularna ili predstavlja prevaru.

### Struktura podataka

Dataset sadrži **284 807 transakcija i 31 kolonu**. Sve kolone su numeričkog tipa, pri čemu je 30 kolona tipa `float64`, a kolona `Class` tipa `int64`.

Strukturu čine:

* `Time` – vreme proteklo od prve transakcije,
* `V1`–`V28` – PCA-transformisane karakteristike transakcija,
* `Amount` – iznos transakcije,
* `Class` – ciljna promenljiva.

Kolona `Class` predstavlja oznaku klase:

* `0` – regularna transakcija,
* `1` – prevara.

### Analiza podataka

Prilikom početne analize proverene su dimenzije dataseta, tipovi podataka, nedostajuće vrednosti i raspodela transakcija po klasama. U datasetu nema nedostajućih vrednosti. Proverom je utvrđeno da sve kolone imaju po **284 807** validnih vrednosti.

Posebno je analizirana raspodela ciljne promenljive `Class`. Rezultati pokazuju izraženu neuravnoteženost klasa:

| Klasa         | Broj transakcija |
| ------------- | ---------------: |
| 0 – regularna |          284 315 |
| 1 – prevara   |              492 |

Od ukupnog broja transakcija, samo 492 pripadaju klasi prevara, što čini približno **0,17%** svih transakcija. 

Pored raspodele klasa, analizirana je i raspodela **iznosa transakcija (`Amount`)**. Za pregledniju vizuelizaciju prikazan je najveći deo podataka do 99. percentila, jer mali broj transakcija sa veoma velikim iznosima može da oteža prikaz kompletne raspodele. Takođe je analizirana raspodela vremena transakcija (`Time`). Ova analiza omogućava bolje razumevanje podataka pre njihovog prosleđivanja neuronskim mrežama.

### Preprocesiranje podataka

Pre treniranja modela ciljna promenljiva `Class` izdvaja se od ulaznih promenljivih. Na ovaj način `X` sadrži 30 ulaznih karakteristika, dok `y` predstavlja ciljnu promenljivu koju model treba da predvidi.

Podaci se zatim dele na trening i test skup. Za testiranje se izdvaja **20% podataka**, dok se preostalih **80%** koristi za trening. Prilikom podele koristi se `stratify=y`, čime se čuva približan odnos klasa u trening i test skupu. Nakon podele vrši se standardizacija promenljivih `Time` i `Amount` pomoću klase `StandardScaler`. PCA-transformisane promenljive `V1`–`V28` se dodatno ne skaliraju u ovom koraku.

### Rešavanje neuravnoteženosti klasa

Zbog veoma malog broja prevara, u projektu je korišćen pristup zasnovan na **class weights**. Za Model 2 i Model 3 težine klasa računaju se pomoću funkcije `compute_class_weight` na trening skupu.

Dobijene težine su približno:

* klasa `0`: **0,50**
* klasa `1`: **289,14**

Na taj način greška koju model napravi na fraud transakciji dobija znatno veću težinu tokom treniranja. Model 1 namerno ne koristi balansiranje i služi kao osnovni model za poređenje sa modelima kod kojih je uveden `class_weight`.

## 3. Arhitektura modela

Sva tri modela imaju istu arhitekturu:

**30 → 32 → 16 → 1**

- ulazni sloj: 30 neurona
- prvi skriveni sloj: 32 neurona, ReLU
- drugi skriveni sloj: 16 neurona, ReLU
- izlazni sloj: 1 neuron, Sigmoid

Arhitektura modela je definisana pomoću Sequential modela iz biblioteke TensorFlow/Keras. 

## 4. Trening

Treniranje modela izvršeno je pomoću `Adam` optimizatora sa learning rate vrednošću **0.001** i `binary_crossentropy` funkcijom gubitka.

Za svaki model postavljen je maksimalan broj od **50 epoha**, uz `batch_size=2048`. Za validaciju se koristi **20% trening skupa**.

Tokom treniranja koristi se **Early Stopping** koji prati `validation loss`. Ako se rezultat ne poboljšava tokom 5 uzastopnih epoha, trening se zaustavlja i vraćaju se težine modela koje su dale najbolji rezultat.

Kod Modela 2 i Modela 3 koristi se `class_weight`. Pošto je broj fraud transakcija mnogo manji od broja regularnih transakcija, ovim pristupom se greškama na klasi prevara daje veća težina.

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Analiza je sprovedena poređenjem tri različite konfiguracije neuronske mreže za detekciju prevara.

- **Model 1** predstavlja osnovni model bez dodatnog tretiranja neuravnoteženosti klasa.
- **Model 2** uvodi `class_weight` kako bi se veća pažnja posvetila ređoj klasi, odnosno prevarama.
- **Model 3**, pored `class_weight`, koristi `Dropout` od `0.30` i L2 regularizaciju od `0.0001`.

Za trening modela korišćen je **Adam optimizator** sa `learning_rate=0.001`, `batch_size=2048` i **Early Stopping**.

Promena hiperparametara pokazala je da korišćenje `class_weight` povećava sposobnost modela da prepozna stvarne prevare. Istovremeno, povećanje **recall-a** dovodi do smanjenja **precision-a**.

## 6. Rezultati evaluacije

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Model 1 | 99.94% | 83.52% | 77.55% | 0.804 | 0.982 | 0.824 |
| Model 2 | 99.08% | 14.67% | 89.80% | 0.252 | 0.960 | 0.700 |
| Model 3 | 97.97% | 7.25% | 91.84% | 0.134 | 0.977 | 0.714 |

Rezultati pokazuju da **Model 1** ostvaruje najbolji balans između precision i recall metrike.

**Model 3** postiže najveći recall od **91.84%**, što znači da uspešno prepoznaje najveći procenat stvarnih prevara. Međutim, njegova precision vrednost je znatno niža, što ukazuje na veći broj lažnih uzbuna.

Zbog velike neuravnoteženosti klasa, sama **accuracy** metrika nije dovoljna za procenu kvaliteta modela. Zato su posebno posmatrane **precision, recall, F1 i PR-AUC** metrike.

## 7. Diskusija
Model 3 može biti koristan banci jer ostvaruje najveći Recall od 90,8%, što znači da uspeva da prepozna najveći procenat stvarnih prevara među testiranim modelima. Iako je Precision nizak (7%), njegov cilj je da što manje prevara prođe neprimećeno. Transakcije koje model označi kao sumnjive mogu se zatim poslati na dodatnu proveru, SMS verifikaciju ili privremenu blokadu, čime se smanjuje rizik od propuštanja prevarnih transakcija.


Kod Modela 3 dodatno se koriste:

* **Dropout = 0.30**, kojim se tokom treninga nasumično isključuje 30% neurona;
* **L2 regularizacija = 0.0001**, koja ograničava prevelike vrednosti težina i pomaže u smanjenju overfitting-a.

## 8. Zaključak

Kroz proces eksperimentisanja i podešavanja hiperparametara uspešno je razvijena neuronska mreža za detekciju prevara u uslovima neuravnoteženosti klasa. Kombinacijom `Class Weights` balansiranja i metoda regularizacije, kao što su `Dropout` i L2 regularizacija, Model 3 je ostvario najveći **Recall od 91.84%** na testnim podacima. Istovremeno, Model 1 je ostvario najbolji balans između **Precision** i **Recall** metrike. Rezultati pokazuju da izbor hiperparametara direktno utiče na ponašanje modela i da izbor konačne konfiguracije zavisi od prioriteta sistema — veće otkrivanje prevara ili smanjenje broja lažnih uzbuna.

Na ovaj način se porede osnovni model, model sa rešavanjem neuravnoteženosti klasa i model koji pored toga koristi i regularizaciju.
