import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Predicción Fitness", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("fitness_dt.csv")

df = load_data()
modelo = joblib.load("modelo_random_forest.pkl")

st.title("✨ POSH COLLECTIVE LAB!")
st.markdown("Conectar cuerpo y mente. Esta app predice tu estado post-entrenamiento para ayudarte a encontrar equilibrio, sentirte mejor y cuidar tu salud mental. 🧘‍♀️💚")

# --- BLOQUE DE BIENVENIDA VISUAL Y GUÍA DE USO ---

# 1. Tipos de entrenamiento disponibles (tarjetas visuales)
st.markdown("### 🏃‍♀️ Tipos de entrenamiento disponibles")

cols = st.columns(5)
workouts = [
    ("Yoga", "🧘🏾"),
    ("Running", "🏃‍♂️"),
    ("Cycling", "🚴‍♂️"),
    ("Strength", "🏋️"),
    ("HIIT", "🔥")
]

for col, (name, emoji) in zip(cols, workouts):
    with col:
        st.markdown(f"### {emoji}")
        st.markdown(f"**{name}**")

# 2. Visualización filtrada de popularidad
st.markdown("### 📊 Popularidad en los datos reales")

try:
    import plotly.express as px
    df = pd.read_csv("fitness_dt.csv")

    # ✅ Filtrar SOLO las 5 prácticas que usás
    valid_workouts = ["Yoga", "Running", "Cycling", "Strength", "HIIT"]
    df_filtered = df[df["Workout Type"].isin(valid_workouts)]
    pie_data = df_filtered["Workout Type"].value_counts().reset_index()
    pie_data.columns = ["Workout Type", "Count"]

    fig_pie = px.pie(pie_data, values="Count", names="Workout Type",
                     title="Distribución de tipos de entrenamiento (datos reales)",
                     color_discrete_sequence=px.colors.sequential.RdBu)

    st.plotly_chart(fig_pie, use_container_width=True)

except Exception as e:
    st.warning("⚠️ No se pudo cargar el dataset para mostrar el gráfico.")

st.markdown("---")

# 3. Guía de pasos
st.markdown("### 🧠 ¿Cómo funciona esta app?")
st.markdown("""
1. ✍️ Ingresá tus datos de entrenamiento (versión principiante o avanzada)
2. 🔮 El modelo predice tu estado de ánimo después de entrenar
3. 🎯 Recibís recomendaciones personalizadas con nutrición, descanso y cambios inteligentes
""")

st.markdown("---")

# 4. Transición clara hacia el formulario
st.markdown("### 🚀 ¡Listo para comenzar?")
st.markdown("Seleccioná tu versión preferida abajo para empezar la predicción 👇")

st.divider()
# Selector de versión
st.subheader("📋 Datos del Usuario")

if "previous_modo" not in st.session_state:
    st.session_state["previous_modo"] = ""

modo = st.radio(
    "Seleccioná tu versión:",
    ["", "👶 Versión Principiante", "🚀 Versión Avanzada"],
    index=0
)

# Check if 'modo' has changed
if modo != st.session_state["previous_modo"]:
    # Reset prediction-related session state variables
    st.session_state["prediccion_hecha"] = False
    if "entrada_codificada" in st.session_state:
        del st.session_state["entrada_codificada"]
    if "prediccion" in st.session_state:
        del st.session_state["prediccion"]

st.session_state["previous_modo"] = modo

if modo:
    st.markdown("---")

if modo == "👶 Versión Principiante":
    with st.form("formulario_principiante"):
        st.subheader("👶 Versión Principiante")
        age = st.slider("Edad", 15, 80, 30)
        gender = st.selectbox("Género", ["Male", "Female"])
        workout_type = st.selectbox("Tipo de clase", ["Cycling", "HIIT", "Running", "Strength", "Yoga"])
        intensity = st.selectbox("Intensidad", ["Low", "Medium", "High"])
        submit = st.form_submit_button("🎯 Predecir")

    if submit:
        user_id = 1
        height = 170
        weight = 70
        duration = 45
        calories = 500
        hr = 120
        steps = 8000
        distance = 5
        sleep = 7
        intake = 2200
        resting_hr = 65
        mood_before = "Neutral"
        modelo = joblib.load("modelo_random_forest.pkl")

        entrada = pd.DataFrame([{
            "User ID": user_id,
            "Age": age,
            "Height (cm)": height,
            "Weight (kg)": weight,
            "Workout Duration (mins)": duration,
            "Calories Burned": calories,
            "Heart Rate (bpm)": hr,
            "Steps Taken": steps,
            "Distance (km)": distance,
            "Sleep Hours": sleep,
            "Daily Calories Intake": intake,
            "Resting Heart Rate (bpm)": resting_hr,
            "Gender": gender,
            "Workout Type": workout_type,
            "Workout Intensity": intensity,
            "Mood Before Workout": mood_before
        }])

        # Codificación
        dummies_cols = ["Gender", "Workout Type", "Workout Intensity", "Mood Before Workout"]
        entrada_codificada = pd.get_dummies(entrada, columns=dummies_cols, drop_first=False)

        columnas_modelo = [
            'User ID', 'Age', 'Height (cm)', 'Weight (kg)', 'Workout Duration (mins)',
            'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)',
            'Sleep Hours', 'Daily Calories Intake', 'Resting Heart Rate (bpm)',
            'Gender_Male', 'Gender_Other',
            'Workout Type_Cycling', 'Workout Type_HIIT', 'Workout Type_Running',
            'Workout Type_Strength', 'Workout Type_Yoga',
            'Workout Intensity_Low', 'Workout Intensity_Medium',
            'Mood Before Workout_Neutral', 'Mood Before Workout_Stressed',
            'Mood Before Workout_Tired'
        ]

        for col in columnas_modelo:
            if col not in entrada_codificada.columns:
                entrada_codificada[col] = 0

        entrada_codificada = entrada_codificada[columnas_modelo]
        prediccion = modelo.predict(entrada_codificada)[0]
        st.success(f"🎉 Estado de ánimo estimado: **{prediccion}**")

        # Guardar en session_state
        st.session_state["user_id"] = user_id
        st.session_state["height"] = height
        st.session_state["weight"] = weight
        st.session_state["duration"] = duration
        st.session_state["mood_before"] = mood_before

        st.session_state["prediccion_hecha"] = True
        st.session_state["prediccion"] = prediccion
        st.session_state["entrada_codificada"] = entrada_codificada
        st.session_state["calories"] = calories
        st.session_state["sleep"] = sleep
        st.session_state["steps"] = steps
        st.session_state["gender"] = gender
        st.session_state["age"] = age
        st.session_state["intensity"] = intensity
        st.session_state["workout_type"] = workout_type
        st.session_state["hr"] = hr
        st.session_state["distance"] = distance
        st.session_state["intake"] = intake
        st.session_state["resting_hr"] = resting_hr

elif modo == "🚀 Versión Avanzada":
    with st.form("formulario_avanzado"):
        st.subheader("🚀 Versión Avanzada")
        col1, col2, col3 = st.columns(3)
        with col1:
            user_id = st.number_input("User ID", 1, 9999, 1)
            age = st.slider("Edad", 15, 80, 30)
            height = st.slider("Altura (cm)", 120, 220, 170)
            weight = st.slider("Peso (kg)", 40, 150, 70)
        with col2:
            duration = st.slider("Duración (min)", 10, 120, 45)
            calories = st.slider("Calorías", 100, 1000, 500)
            hr = st.slider("FC (bpm)", 60, 200, 120)
            steps = st.slider("Pasos", 0, 20000, 8000)
        with col3:
            distance = st.slider("Distancia (km)", 0.0, 20.0, 5.0)
            sleep = st.slider("Sueño (h)", 0.0, 12.0, 7.0)
            intake = st.slider("Calorías ingeridas", 1000, 4000, 2200)
            resting_hr = st.slider("FC reposo (bpm)", 40, 100, 65)

        col4, col5 = st.columns(2)
        with col4:
            gender = st.selectbox("Género", ["Male", "Female", "Other"])
            workout_type = st.selectbox("Tipo", ["Cycling", "HIIT", "Running", "Strength", "Yoga"])
        with col5:
            intensity = st.selectbox("Intensidad", ["Low", "Medium", "High"])
            mood_before = st.selectbox("Ánimo antes", ["Neutral", "Stressed", "Tired", "Motivated"])

        submit = st.form_submit_button("🎯 Predecir")

    if submit:
        entrada_data = [{
            "User ID": user_id,
            "Age": age,
            "Height (cm)": height,
            "Weight (kg)": weight,
            "Workout Duration (mins)": duration,
            "Calories Burned": calories,
            "Heart Rate (bpm)": hr,
            "Steps Taken": steps,
            "Distance (km)": distance,
            "Sleep Hours": sleep,
            "Daily Calories Intake": intake,
            "Resting Heart Rate (bpm)": resting_hr,
            "Gender": gender,
            "Workout Intensity": intensity,
            "Mood Before Workout": mood_before
        }]
        entrada = pd.DataFrame([{
            "User ID": user_id,
            "Age": age,
            "Height (cm)": height,
            "Weight (kg)": weight,
            "Workout Duration (mins)": duration,
            "Calories Burned": calories,
            "Heart Rate (bpm)": hr,
            "Steps Taken": steps,
            "Distance (km)": distance,
            "Sleep Hours": sleep,
            "Daily Calories Intake": intake,
            "Resting Heart Rate (bpm)": resting_hr,
            "Gender": gender,
            "Workout Type": workout_type,
            "Workout Intensity": intensity,
            "Mood Before Workout": mood_before
        }])

        st.subheader("📊 Resumen de tus datos")
        st.dataframe(entrada)

        # Codificar variables categóricas
        dummies_cols = ["Gender", "Workout Type", "Workout Intensity", "Mood Before Workout"]
        entrada_codificada = pd.get_dummies(entrada, columns=dummies_cols, drop_first=False)

        # 🔧 Aquí va la definición dentro del if
        columnas_modelo = [
            'User ID', 'Age', 'Height (cm)', 'Weight (kg)', 'Workout Duration (mins)',
            'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)',
            'Sleep Hours', 'Daily Calories Intake', 'Resting Heart Rate (bpm)',
            'Gender_Male', 'Gender_Other',
            'Workout Type_Cycling', 'Workout Type_HIIT', 'Workout Type_Running',
            'Workout Type_Strength', 'Workout Type_Yoga',
            'Workout Intensity_Low', 'Workout Intensity_Medium',
            'Mood Before Workout_Neutral', 'Mood Before Workout_Stressed',
            'Mood Before Workout_Tired'
        ]

        for col in columnas_modelo:
            if col not in entrada_codificada.columns:
                entrada_codificada[col] = 0

        entrada_codificada = entrada_codificada[columnas_modelo]

        # Predecir
        prediccion = modelo.predict(entrada_codificada)[0]
        # st.success(f"🎉 Estado de ánimo estimado después del entrenamiento: **{prediccion}**")
        st.markdown("---")

        # Guardar en session_state para recomendaciones
        st.session_state["user_id"] = user_id
        st.session_state["height"] = height
        st.session_state["weight"] = weight
        st.session_state["duration"] = duration
        st.session_state["mood_before"] = mood_before

        st.session_state["prediccion"] = prediccion
        st.session_state["gender"] = gender
        st.session_state["workout_type"] = workout_type
        st.session_state["age"] = age
        st.session_state["intensity"] = intensity
        st.session_state["sleep"] = sleep
        st.session_state["calories"] = calories
        st.session_state["hr"] = hr
        st.session_state["steps"] = steps
        st.session_state["distance"] = distance
        st.session_state["intake"] = intake
        st.session_state["resting_hr"] = resting_hr
        st.session_state["entrada_codificada"] = entrada_codificada
        st.session_state["prediccion_hecha"] = True

        # Gráfico de distribución de predicción
        probas = modelo.predict_proba(st.session_state["entrada_codificada"])[0]
        clases = modelo.classes_

        fig_pred = go.Figure(data=[
            go.Bar(
                x=clases,
                y=probas,
                marker_color=['#36A2EB', '#4CAF50', '#FFB74D'],
                text=[f"{p:.2%}" for p in probas],
                textposition='auto'
            )
        ])

        fig_pred.update_layout(
            title='🔮 Distribución de la Predicción',
            yaxis_title='Probabilidad',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            margin=dict(t=50, b=30)
        )

        st.plotly_chart(fig_pred, use_container_width=True)

# --- BLOQUE DE RECOMENDACIONES AVANZADAS ---
if "prediccion_hecha" in st.session_state and st.session_state["prediccion_hecha"]:
    mood = st.session_state["prediccion"]
    age = st.session_state["age"]
    gender = st.session_state["gender"]
    workout_type = st.session_state["workout_type"]
    intensity = st.session_state["intensity"]
    sleep = st.session_state["sleep"]
    calories = st.session_state["calories"]
    hr = st.session_state["hr"]
    steps = st.session_state["steps"]
    distance = st.session_state["distance"]
    intake = st.session_state["intake"]
    resting_hr = st.session_state["resting_hr"]

    st.markdown("## 🎯 Recomendaciones Profesionales Personalizadas")

    if mood == "Fatigued":
        st.warning("😴 Estás fatigado. Necesitás recuperar energía y evitar el sobreentrenamiento.")
        st.markdown("""
        ### 🛠️ Recomendaciones clave
        - 🍌 Snack ideal: Banana + 1 cda de mantequilla de maní antes del entreno.
        - 💧 Hidratate con 500ml de agua + limón + pizca de sal.
        - 🛌 Dormí al menos 7.5h esta noche. Evitá pantallas 30 minutos antes.
        - 🍲 Cena recomendada: Omelette + palta + tostadas integrales.
        - 💓 FC en reposo >75? Día de descanso activo: caminata o movilidad ligera.
        """)

    elif mood == "Neutral":
        st.info("😐 Estás en un punto intermedio. ¡Perfecto para potenciar tu energía!")
        st.markdown("""
        ### ⚡ Cómo llegar a Energized
        - 🍽️ Pre-entreno 90min antes: Avena + arándanos + 1 café solo.
        - 🎧 Playlist recomendada: música a 140-160 bpm.
        - 📈 Intensidad sugerida: Media-Alta, pero sin pasarte del 80% FC máxima.
        """)

    elif mood == "Energized":
        st.success("🚀 ¡Estás con toda la energía! Ideal para mantener ese estado.")
        st.markdown("""
        ### 🥇 Mantené tu nivel óptimo
        - 🔁 Alterná días de cardio y fuerza cada 48h.
        - 🧃 Post-entreno: Smoothie de proteínas + mango + cúrcuma.
        - 🥗 Almuerzo ideal: Pollo grillado + arroz integral + palta.
        - 🎯 Agregá 1 sesión de movilidad o respiración cada 3 días.
        """)

    # --- COMPARADOR AJUSTADO A PREPROCESAMIENTO ORIGINAL ---

if modo == "🚀 Versión Avanzada" and "prediccion_hecha" in st.session_state and st.session_state["prediccion_hecha"]:
    st.markdown("## 🧪 Probabilidades de estado de ánimo según el tipo de entrenamiento")
    workout_types = ['Yoga', 'Strength', 'Cycling', 'HIIT', 'Running', 'Walking']
    results = []

    for w_type in workout_types:
        entrada_data_copy = entrada_data[0].copy()
        entrada_data_copy["Workout Type"] = w_type

        entrada_sim = pd.DataFrame([entrada_data_copy])

        dummies_cols = ["Gender", "Workout Type", "Workout Intensity", "Mood Before Workout"]
        entrada_cod = pd.get_dummies(entrada_sim, columns=dummies_cols, drop_first=False)

        for col in columnas_modelo:
            if col not in entrada_cod.columns:
                entrada_cod[col] = 0

        entrada_cod = entrada_cod[columnas_modelo]

        modelo = joblib.load("modelo_random_forest.pkl")
        probas = modelo.predict_proba(entrada_cod)[0]
        result = {
            "Workout Type": w_type,
            "Energized %": (probas[modelo.classes_.tolist().index("Energized")] * 100),
            "Fatigued %": (probas[modelo.classes_.tolist().index("Fatigued")] * 100),
            "Neutral %": (probas[modelo.classes_.tolist().index("Neutral")] * 100),
        }
        results.append(result)

    df_prob = pd.DataFrame(results)
    st.dataframe(df_prob)

    # Highlight best/worst
    best = df_prob.sort_values("Energized %", ascending=False).iloc[0]
    worst = df_prob.sort_values("Fatigued %", ascending=False).iloc[0]

    st.success(f"🏆 Best energy boost: {best['Workout Type']} ({best['Energized %']}%)")
    st.warning(f"⚠️ Highest fatigue risk: {worst['Workout Type']} ({worst['Fatigued %']}%)")

    # Gráfico Gauge de calorías, sueño y pasos
    import plotly.graph_objects as go
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=calories,
        title={"text": "Calorías Quemadas"},
        gauge={"axis": {"range": [0, 1000]}},
        domain={'row': 0, 'column': 0}
    ))

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sleep,
        title={"text": "Horas de Sueño"},
        gauge={"axis": {"range": [0, 12]}},
        domain={'row': 0, 'column': 1}
    ))

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=steps,
        title={"text": "Pasos Dados"},
        gauge={"axis": {"range": [0, 20000]}},
        domain={'row': 0, 'column': 2}
    ))

    fig.update_layout(
        grid={'rows': 1, 'columns': 3},
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )

    st.plotly_chart(fig, use_container_width=True)


    if calories < 300:
        st.warning("🔥 Tu sesión fue de baja exigencia. Podés aumentar la duración o la intensidad si buscás mayor activación.")
    elif calories > 700:
        st.info("🔥 Quemaste muchas calorías. Asegurate de reponer energías con buena hidratación y carbohidratos complejos.")
    else:
        st.success("🔥 Tu gasto calórico está en un rango saludable para la mayoría de los entrenamientos.")

    if sleep < 6:
        st.warning("😴 Dormiste poco. Esto puede afectar tu recuperación y tu estado de ánimo post-entrenamiento.")
    elif 6 <= sleep < 8:
        st.info("😴 Tu descanso fue aceptable, pero podrías beneficiarte con 7-8h de sueño regular.")
    else:
        st.success("😴 Excelente descanso. Esto mejora directamente tu rendimiento y bienestar emocional.")

    if steps < 5000:
        st.warning("🚶‍♂️ Estuviste poco activo durante el día. Intentá sumar más movimiento fuera del entrenamiento.")
    elif steps < 10000:
        st.info("🚶‍♂️ Actividad diaria moderada. Muy bien, ¡vas por buen camino!")
    else:
        st.success("🚶‍♂️ Excelente nivel de pasos diarios. Es ideal para mantener una buena salud física y mental.")


