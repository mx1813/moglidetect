# moglidetect
Ein Tool zur Erkennung von KI-generiertem Text bei wissenschaftlichen Arbeiten an der DHBW Heidenheim. Das Projekt ist ein Teil der Studienarbeit 'Erkennung und Analyse von KI-generierten Texten im Wissenschaftsbereich'. Für mehr Informationen und eine detaillierte Dokumentation über die Entstehung dieses Tools siehe diese Studienarbeit.

---

## Aufsetzen der Umgebung
Um *MogliDetect* lokal ausführen zu können müssen die folgenden Schritte VOR dem Starten der Anwendung erledigt werden müssen:
1. Aktuellsten Release der Software herunterladen
    - Die aktuellste Version der [Releases](https://github.com/mx1813/moglidetect/releases) herunterladen
    - Ein Release enthält neben dem Source Code auch noch die zip-Dateien `mogli_gbert.zip` und `mogli_dbmz.zip`
    - Diese Dateien enthalten die Fine-Tuned LLMs und sind aufgrund der Größe nicht in das Repository direkt eingecheckt
1. Source Code Verzeichnis öffnen
    - hierzu entweder die *Source Code*-zip Datei entpacken oder das Repository klonen
1. Fine-Tuned LLMs entpacken
    - Die beiden folgenden zip-Dateien entpacken
        - `mogli_gbert.zip`
        - `mogli_dbmz.zip`
1. Den Inhalt dieser Dateien in das Projekt einfügen. Dabei die folgende Ordnerstruktur beachten:
    - `moglidetect/fine-tuning/model/mogli_dbmz`
    - `moglidetect/fine-tuning/model/mogli_gbert`

Überprüfen Sie, dass die Funktionen in der `llmdetection.py` Datei der `pipeline`-Funktion in dem model-Parameter das richtige Verzeichnis für das Sprachmodell übergibt, sodass die Texterkennung erfolgreich stattfinden kann.
```
def llm_pipeline(inputText):
    pipe = pipeline("text-classification", model="./fine-tuning/model/mogli_gbert")
    result = pipe(inputText)
    json.dumps(result)
    print(result[0])
    return result[0]
```

---

## Starten der Anwendung
Voraussetzung für das Ausführen der Anwendung: [Python](https://www.python.org/downloads/) ist installiert und eingrichtet.
### Installieren der Requirements
Vor dem ersten Starten der Anwendung müssen die für die Anwendung benötigten Requirements heruntergeladen werde. Hierzu empfiehlt es sich, zuerst eine virtuelle Umgebung in Python zu erstellen, um mögliche Konflikte zu vermeiden. 
#### Virtuelle Umgebung erstellen und aktivieren
Einen detaillierten Guide zum Erstellen einer virtuellen Umgebung findet sich [hier](https://docs.python.org/3/library/venv.html). 
Im folgenden wird ein Beispiel zur Erstellung auf Windows gegeben:
```bash
python -m venv venv
```
Nachdem auf diese Weise eine virtuelle Umgebung erstellt wurde, muss diese anschließend noch aktiviert werden. Hierzu in dem obersten Verzeichnis des Projekts, in dem auch die virtuelle Umgebug befinet den folgenden Befehl ausführen:
```bash
venv\Scripts\activate
```

#### Installieren der Requirements
Die Requirements können anschließend mit dem folgenden Befehl heruntergeladen werden:
```bash
pip install -r requirements.txt
```

### Starten der Benutzeroberfläche
Als Benutzeroberfläche vsetzt dieses Tool eine Web-Anwendung ein, welche mit [Flask](https://flask.palletsprojects.com/en/3.0.x/) umgesetzt wurde. Die Anwendung kann also einfach mithilfe der `app.py`-Datei gestartet werden:
```bash
python app.py
```
Die Anwendung läuft dann auf dem [localhost:5000](http://localhost:5000/) und kann dort abgerufen werden.

---

## Verwendung
Über ein Dropdown-Menü kann das gewünschte Erkennungsverfahren ausgewählt werden. Anschließend kann der zu überprüfende Text in das große Textfeld eingefügt werden. Der Text sollte dabei nicht länger als 2048 Zeichen sein. Danach wird mit einem Klick auf den *Check Text!*-Button die Texterkennung gestartet. Je nach gewähltem Erkennungsverfahren muss etwas gewartet werden, bis schlussendlich das Ergebnis der Texterkennung präsentiert wird. 