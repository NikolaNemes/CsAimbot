# CsAimbot

Aplikacija prepoznaje protivničke i timske igrače, naznačava njihove pozicije na ekranu i pomaže igraču pri ciljanju u slučaju protivnika.

# Demonstracija

![](Demonstration.gif)

# Uputsvo za podešavanje

Da bi aplikacija uspešno radila na računaru, potrebno je obezbediti dobro python okruženje, i napraviti neke izmene Counter Strike 1.6 igrici.

## Podešavanje pythona

Projekat je izgrađen na osnovu ovog projekta, [link](https://github.com/qqwweee/keras-yolo3). Tako da ista podešavanja koji važe za taj projekat, važe i za ovaj.

Jedina razlika u odnosu na podešavanje projekta je da je ovde korišćen python 3.6.8 umesto navedenog 3.5.2. Verzije kerasa i tensorflow-a su iste.

Da bi se lakše namestila aplikacija na novom računaru, u rootu repozitorijuma se nalazi requirments.txt koji sadrži sve potrebne zavnisnosti za aplikaciju.

Podešavanje python okruženja se onda svodi na pravljenje python virtuelnog okruženja 3.6.8 i instalaciju zavinosti pomoću requirements.txt-a.

S obzirom da je YOLO algoritam zahtevan, a aplikacija treba da radi u realnom vremenu, preporučuje se instalacija NVIDIA CUDA softwarе-a radi boljih rezultata.


## Podešavanje igrice

Model je istreniran da prepoznaje samo dva modela igrača u igri, tako da je potrebno zameniti sve ostale modele sa samo ova dva.

Za terorističku stranu korišćen je **leet** model, a za kanter-terorističku stranu korišćen je **sas** model.

Da bi se izvršila zamena potrebno je otići do direktorijuma instalacije igrice, pa zatim iznavigirati kroz cstrike/models/player.

U leet folderu se nalazi leet.mdl, potrebno je prekopirati taj fajl u sledeće direktorijume: **arctic**, **terror** i **guerilla**.

Nakon kopiranja leet.mdl-a, potrebno je izbrisati stari mdl fajl, i preimenovati leet.mdl u ime starog fajla (na primer u arctic folderu brišemo arctic.mdl, prebacujemo leet.mdl, pa zatim preimenujemo leet.mdl u arctic.mdl)

Za sas model se radi analogno, njegov model se prekopira u sledeće direktorijume: **gign**, **gsg9** i **urban**.

## Korišcenje

Aplikacija se pokreće sa pozivom yolo_video.py fajla bez argumenata.

Kada se pokrene Counter Strike, potrebno je postaviti igru u windowed mode, i postaviti rezoluciju na 800x600. Nakon ovoga, sam prozor igre treba postaviti u gornji levi ugao ekrana da bi aplikacija mogla da vidi igru.

Lako se može videti dal su uspešno poravnate po tome što bi se prikaz aplikacije i same igre trebao poklapati.

Bot po defaultu ne puca, nego samo detektuje igrače, što se može prepoznati po "SHOOTING OFF" tekstu u gornjem levom uglu aplikacije.

Prebacivanje između defaultnog statusa i pucanja se vrši pomoću home dugmeta.

## Kalibracija miša

Pošto se mouse sensitivity razlikuje između računara, velika je verovatnoća da ciljanje u novom okruženju nije dobro.

Da bi se ovo ispravilo napisane su skripte koje kalibrišu ciljanje bota.

Da bi se skripte primenile, potrebno je ući u mapu bez protivnika, i postaviti crveni poster.

Poster se nalazi u rootu repozitorijuma, pod fajlom tempdecal.wad. Ovaj fajl je potrebno zameniti u cstrike direktorijumu igre.

Da bi skripte za kalibraciju radile, potrebno je postaviti poster na dobro osvetljenom zidu kao na slici ispod.

![Spray example](https://raw.githubusercontent.com/NikolaNemes/CsAimbot/master/sprayExample.png)

Potrebno je udaljiti se od postera, kao na slici, i pokrenuti *calibrator.py* i  *calibratory.py* skripte iz calibrator direktorijuma.

Nakon ovoga potrebno je primeniti *corrector.py* skriptu nad rezultujućim json fajlovima.

Da bi bot počeo da koristi nova mapiranja potrebno je prekopirati nove json fajlove preko starih.

Fajlovi koje bot koristi se nalaze u AutomaticCsBot/MouseDicts direktorijumu.
