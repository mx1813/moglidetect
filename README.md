# moglidetect
Ein Tool zur Erkennung von KI-generiertem Text bei wissenschaftlichen Arbeiten an der DHBW Heidenheim. Das Projekt ist ein Teil der Studienarbeit 'Erkennung und Analyse von KI-generierten Texten im Wissenschaftsbereich'. Für mehr Informationen und eine detaillierte Dokumentation über die Entstehung dieses Tools siehe diese Studienarbeit.

---
## Starten der Anwendung
### Installieren der Requirements
Vor dem ersten Starten der Anwendung müssen die für die Anwendung benötigten Requirements heruntergeladen werde. Hierzu empfiehlt es sich, zuerst eine [virtuelle Umgebung](https://docs.python.org/3/library/venv.html) in Python zu erstellen, um mögliche Konflikte zu vermeiden. Die Requirements können anschließend mit dem folgenden Befehl heruntergeladen werden:
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