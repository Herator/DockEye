# Presentasjonsguide — oppdatert branch

Presentasjonen er laget på nytt fra `presentation.json`. Redigerbar PowerPoint
og PDF ligger i `presentation/`. PowerPoint har norske foredragsnotater.
Bygg på nytt med `python tools/build_presentation.py`.

## Fortellingen

«Vi undersøker aktivitet ved én brygge. Vi startet med bevegelsesdeteksjon og
har nå koblet sammen modell, tracking og hendelseslogging. Systemet kan kjøres
med en generell person/båt-modell. En egen fiskemodell krever fortsatt merkede
opptak og evaluering. Som et alternativ prøver vi person pluss stanggeometri,
men dette gir mulige fiskehendelser som må vurderes.»

Vis forskjellen mellom implementert programflyt, hvilke modellvekter som faktisk
finnes, og hva dere har målt. Ikke presenter rod-kandidater eller syntetiske
tester som dokumentert fiskegjenkjenning.

## Architecture — ca. 60 sekunder

«Videoen behandles ett bilde om gangen. Først hopper vi over svarte startbilder
og laster eller definerer sonene. Deretter finner bakgrunnsmodellen endringer
på brygga og i vannet, med ulike innstillinger for å håndtere bølger.

Bevegelsen foreslår områder som en YOLO-modell undersøker. Modellen klassifiserer
objekter, mens trackeren kobler observasjoner mellom bilder og gir ID-er. Aktive
spor undersøkes videre når de står stille; ellers kunne vi mistet en fisker.

Hendelseslogikken bruker tid fra videoen. Ved gjentatt fiskeevidens lenge nok
opprettes én loggrad og ett bilde. Båter telles når et spor krysser en valgt linje.
Operatøren kan se video og hendelser og justere bevegelsesparametere.

Den generelle modellen har ingen fiskeklasse. Full fiskeklassifisering krever
egne trente vekter. Stangmetoden er et separat eksperiment som logger mulig
fisking, ikke en sikker konklusjon.»

Pek på flyten mens du snakker. Forklar ansvar og dataflyt, ikke hver parameter.
Ved spørsmål: trackeren matcher bokser én-til-én med en kort frist ved manglende
deteksjon. Den er ikke en Kalman-tracker og garanterer ikke ID etter lang skjuling.

## Project scope — ca. 40 sekunder

«Prosjektet er avgrenset til opptak fra ett fast kamera ved vår egen brygge.
Vi behandler video i etterkant. Løsningen omfatter soner, objektklassifisering,
tracking, båttelling og hendelseslogging, med bilde først ved en bekreftet
hendelse eller tidsbekreftet kandidat fra den eksperimentelle metoden.

Vi skal evaluere flere lys- og værforhold; vi hevder ikke at disse er løst ennå.
Vi identifiserer ikke personer og avgjør ikke om de har tillatelse til å være der.
Direkte varsling, nattkamera og flere kameravinkler er utenfor scope.

Valgfri kalibrering gir bakkeposisjoner på brygga i meter fra kjente mål.
Dette er ikke en generell metode for å måle menneskehøyde eller båtlengde.»

## Use cases — ca. 60 sekunder

«Operatøren starter med å markere sonene, som lagres og brukes på nytt.
Deretter velger operatøren et opptak og en modell, kjører analysen og ser bokser
og ID-er. Parametere kan justeres under avspilling, og resultatene lagres i en
egen mappe for hver kjøring.

Ved mulig fisking undersøker systemet observasjonene over tid. En trent modell
kan gi en fiskehendelse hvis den har fiskeklassen. Den eksperimentelle
stangmetoden gir i stedet en tydelig merket kandidat for etterkontroll.

Et annet brukstilfelle er båttelling: operatøren setter en tellingslinje, og en
båt telles én gang per spor når den krysser. Til slutt kan operatøren gjennomgå
hendelser, bilder og CSV og sammenligne med manuell fasit.»

Merking og trening er et utviklingsarbeidsløp, ikke en vanlig operatørhendelse.

## Demo

1. Bruk et opptak og soner som passer kameravinkelen. Forklar hvilke vekter og
   hvilken metode du kjører. Ikke bruk syntetiske treningsvekter i produktdemoen.
2. Vis person/båt-bokser og spor-ID-er. Vis også en kjent feil; forklar at en boks
   fra bevegelsesmodus alene ikke er en semantisk person/båt-klassifikasjon.
3. Vis CSV-kolonnene og matchende snapshot. Forklar video-sekunder og at
   `fishing_candidate` ikke betyr bevist fisking.
4. Vis testene og skill mellom systemtest med kontrollert detektor, faktisk
   modellkjøring, treningskontroll og målt nøyaktighet på egne opptak.

## Rettelsene fra den opprinnelige PDF-en

- Den opprinnelige tabellen hadde 20 krav: 8 implementert, 2 delvis, 10 planlagt.
  Det var ikke 18 krav. Den nye branchen må vurderes separat, ikke ved å gjenbruke
  den gamle opptellingen som dagens status.
- Skyggelabelen fjernes eksplisitt også ved lav terskel i vannsonen.
- Moduler, tilkoblet tracker, modell og logger vises etter faktisk kode.
- Kalman-filter, kroppshøyde og klassifisering ut fra størrelse er ikke beskrevet
  som implementert. En planhomografi gir bare punkter på det målte planet.
- Én boks per fysisk objekt, ingen bølgeutslag, original avspillingshastighet og
  ingen dobbeltelling er ikke ubetingede garantier.
- De tilgjengelige kontrollerte opptakene er 1280×720, mens lagrede soner har
  1920×1080-koordinater. Oppløsning og kameravinkel må håndteres eksplisitt.
- Testplanen kommer før endelig evaluering. Resultater på egne merkede klipp
  fylles inn når de er målt, uten å dikte tall eller bruke syntetiske data som fasit.
