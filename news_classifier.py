import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Creando un DataFrame con los titulares y sus respectivas etiquetas
titulares_data = {
    "Titular": [
        "Nuevo algoritmo de IA predice el cambio climático con una precisión sin precedentes",
        "Inteligencia Artificial revoluciona la medicina: Diagnósticos en segundos",
        "IA en el arte: Cómo los robots están creando obras maestras",
        "La ética de la IA: Líderes tecnológicos debaten sobre el futuro del desarrollo inteligente",
        "Educación personalizada a través de la Inteligencia Artificial: El futuro de la enseñanza",
        "Robots dotados de IA para realizar tareas domésticas: ¿El fin de las labores del hogar?",
        "Inteligencia Artificial en la agricultura: Aumentando los rendimientos de las cosechas",
        "La IA detecta fraudes en tiempo real: Un nuevo horizonte para la banca y finanzas",
        "El papel de la Inteligencia Artificial en la lucha contra pandemias globales",
        "Derechos de los robots: La creciente demanda de legislación para la Inteligencia Artificial avanzada",
        "6 trabajos que la Inteligencia Artificial está creando y qué tipo de preparación requieren",
        "Sam Altman regresará como jefe de OpenAI días después de ser despedido",
        "Qué está pasando realmente en OpenAI, la creadora de ChatGPT inmersa en el caos tras el despido de Sam Altman",
        "Microsoft contrata a Sam Altman luego de su polémico despido de OpenAI (por el que los empleados piden la dimisión de la junta directiva)",
        "Un robot levanta un cráneo",
        "Nigromancia digital: cómo la inteligencia artificial cambia nuestra relación con los muertos",
        "Los tokens y otras novedades que están mejorando los chatbots de inteligencia artificial",
        "Cómo DeepMind, la inteligencia artificial de Google, está acelerando la identificación de genes que causan enfermedades",
        "Cómo puede ayudar la inteligencia artificial en la búsqueda de la eterna juventud",
        "El escándalo en un pequeño pueblo de España por las imágenes de decenas de niñas y jóvenes desnudas generadas por IA",

        "Tensiones escalan: Ejércitos en alerta máxima en la frontera de X e Y",
        "Nuevas alianzas militares se forman en respuesta a las crecientes amenazas globales",
        "Guerra cibernética: El nuevo frente de batalla entre naciones",
        "Avances en drones de combate: El futuro de la guerra aérea",
        "La crisis de refugiados se intensifica a medida que los conflictos armados continúan",
        "Sanciones económicas impuestas como respuesta a la agresión militar en Z",
        "Víctimas civiles aumentan: El trágico costo humano de la guerra en A",
        "Protestas globales exigen el fin de la guerra y la violencia",
        "Revelan el uso de armas prohibidas en el conflicto de B: Comunidad internacional en alerta",
        "Qué buscaba EE.UU. al demorar la respuesta al ataque mortal con un dron sobre una de sus bases",
        "Soldados de EE.UU. en Siria a principios de enero",
        "EE.UU. lanza ataques contra objetivos vinculados con Irán en Irak y Siria y Bagdad advierte de 'consecuencias desastrosas' para la región",
        "Una mujer palestina frente a un edificio destruido en el campo de refugiados de al Maghazi en el centro de la Franja de Gaza el 16 de enero de 2024.",
        "Los gráficos que muestran que al menos la mitad de los edificios de Gaza fueron dañados o destruidos",
        "4 frentes donde la guerra en Medio Oriente se ha expandido más allá del conflicto Israel-Hamás",
        "Las preguntas sin respuesta que deja la caída de un avión militar en Rusia en medio de las acusaciones entre Moscú y Kyiv",
        "Moscú acusa a Ucrania de derribar un avión militar ruso con 65 prisioneros ucranianos a bordo",
        "Los palestinos que enfrentan amenazas de muerte en la ciudad de EE.UU. en la que un niño de su comunidad fue asesinado",
        "Estados Unidos lanza un nuevo ataque militar contra los combatientes hutíes en Yemen en una expansión de la guerra en Medio Oriente",
        "Negociaciones de paz fracasan: Se teme la escalada del conflicto en C"

    ],
    "Etiqueta": [
        "IA", "IA", "IA", "IA", "IA",
        "IA", "IA", "IA", "IA", "IA",
        "IA", "IA", "IA", "IA", "IA",
        "IA", "IA", "IA", "IA", "IA",

        "Guerra", "Guerra", "Guerra", "Guerra", "Guerra",
        "Guerra", "Guerra", "Guerra", "Guerra", "Guerra",
        "Guerra", "Guerra", "Guerra", "Guerra", "Guerra",
        "Guerra", "Guerra", "Guerra", "Guerra", "Guerra"
    ]
}

df_titulares = pd.DataFrame(titulares_data)

# Preprocesamiento de datos y división en conjuntos de entrenamiento y prueba
X = df_titulares['Titular']  # Las características (titulares)
y = df_titulares['Etiqueta']  # Las etiquetas (IA o Guerra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un pipeline que incluye TF-IDF vectorization y Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42))
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo
y_pred = pipeline.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Función para categorizar un nuevo titular
def categorizar_titular(titular):
    pred = pipeline.predict([titular])[0]
    return pred

# Ejemplo de uso
nuevo_titular = input("Introduce un titular para categorizar: ")
categoria = categorizar_titular(nuevo_titular)
print(f'El titular es sobre: {categoria}')
