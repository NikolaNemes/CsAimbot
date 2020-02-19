# CsAimbot

# Uputsvo za podešavanje

Da bi aplikacija uspešno radila na računaru, potrebno je obezbediti dobro python okruženje, i napraviti neke izmene Counter Strike 1.6 igrici.

## Podesavanje pythona

Projekat je izgrađen na osnovu ovog projekta, [link](https://github.com/qqwweee/keras-yolo3). Tako da isti podešavanja koji važe za taj projekat, važe i za ovaj.

Jedina razlika u odnosu na podešavanje projekta je da smo mi koristili verziju pythona 3.6.8 umesto navedene 3.5.2, verzije kerasa i tensorflow-a su iste.

Da bi se lakše namestila aplikacija na novom računaru, u rootu našeg repozitorijuma se nalazi requirments.txt koji sadrži sve potrebne zavnisnosti za aplikaciju.

Podešavanje python okruženja se onda svodi na pravljenje python virtuelnog okruženja 3.6.8 i instaliranje pomoću requirements.txt-a.

S obzirom da je YOLO algoritam zahtevan, a aplikacija treba da radi u realnom vremenu, preporučuje se instalacija NVIDIA CUDA softwarе-a radi boljih rezultata.


## Podesavanje igrice

Model je istreniran da prepoznaje samo dva modela igrača u igri, tako da je potrebno zameniti sve ostale modele sa samo ova dva.

Za terorističku stranu korišćen je **leet** model, a za kanter-terorističku stranu korišćen je **sas** model.

Da bi se izvršila zamena potrebno je otići do direktorijuma instalacije igrice, pa zatim iznavigirati kroz cstrike/models/player.

U leet folderu se nalazi leet.mdl, potrebno je prekopirati taj fajl u sledeće direktorijume: **arctic**, **terror** i **guerilla**.

Nakon kopiranja leet.mdl-a, potrebno je izbrisati stari mdl fajl, i preimenovati leet.mdl u ime starog fajla (na primer u arctic folderu brišemo arctic.mdl, prebacujemo leet.mdl, pa zatim preimenujemo leet.mdl u arctic.mdl)

Za sas model se radi analogno, njegov model se prekopira u sledeće direktorijume: **gign**, **gsg9** i **urban**.

## Koriscenje

Aplikacija se pokreće sa pozivom yolo_video.py fajla bez argumenata.

Kada se pokrene Counter Strike, potrebno je postaviti igru u windowed mode, i postaviti rezoluciju na 800x600. Nakon ovoga, sam prozor igre treba postaviti u gornji levi ugao ekrana da bi se poravnala sa našom aplikacijom.

Lako se može videti dal su uspešno poravnate po tome što bi se prikaz naše aplikacije i same igre trebao poravnati.

Bot po defaultu ne puca, nego samo detektuje objekte, što se može prepoznati po "SHOOTING OFF" tekstu u gornjem levom uglu aplikacije.

Prebacivanje izmežu ova dva statusa se vrši pomoću home dugmeta.
