# Use a pipeline as a high-level helper
from transformers import pipeline

inputText = "Der erste Schritt bei der Anwendung der wissenschaftlichen Methode zur Lösung eines schwierigen Problems besteht darin, Beobachtungen zu machen und Daten zu sammeln. ⁤⁤Dies beinhaltet das Sammeln von Informationen über das Problem und das Sammeln relevanter Fakten, die dazu beitragen können, mögliche Ursachen oder Faktoren zu identifizieren. ⁤⁤Diese Informationen können auf verschiedene Weise gesammelt werden, einschließlich Experimenten, Umfragen und Beobachtungen. ⁤"

pipe = pipeline("text-classification", model="./fine-tuning/model/mogli_sec")
result = pipe(inputText)
print(result)
print(result[0])