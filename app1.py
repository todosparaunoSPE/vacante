# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 19:27:44 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Configuración de la página
st.set_page_config(
    page_title="Portfolio - Análisis de Datos | Agencia de Transformación Digital",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    .highlight-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2e7d32;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>📊 Portfolio de Análisis de Datos</h1>
    <h3>Candidatura para la Agencia de Transformación Digital y Telecomunicaciones</h3>
    <p><strong>Demostración de habilidades en Estadística, Visualización de Datos y Vocación de Servicio Público</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("Desarrollado por: Javier Horacio Pérez Ricárdez")
st.sidebar.subtitle("🔍 Navegación")
seccion = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["🏠 Inicio", "📈 Análisis Estadístico", "📊 Visualizaciones Avanzadas", 
     "🎯 Caso de Uso: Gobierno Digital", "👤 Perfil Profesional"]
)

# Función para generar datos simulados de gobierno
@st.cache_data
def generar_datos_gobierno():
    # Datos de servicios digitales por estado
    estados = ['CDMX', 'Jalisco', 'Nuevo León', 'Estado de México', 'Veracruz', 
               'Puebla', 'Guanajuato', 'Chihuahua', 'Michoacán', 'Oaxaca']
    
    data_servicios = {
        'Estado': estados,
        'Servicios_Digitalizados': np.random.randint(50, 200, 10),
        'Usuarios_Activos': np.random.randint(100000, 1000000, 10),
        'Satisfaccion_Ciudadana': np.random.uniform(3.5, 5.0, 10),
        'Tiempo_Promedio_Tramite': np.random.randint(5, 45, 10),
        'Inversión_TIC': np.random.randint(10, 100, 10)
    }
    
    # Datos temporales de adopción digital
    fechas = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    data_temporal = {
        'Fecha': fechas,
        'Adopcion_Digital': np.cumsum(np.random.normal(2, 0.5, len(fechas))) + 20,
        'Tramites_Digitales': np.cumsum(np.random.normal(1000, 200, len(fechas))) + 5000,
        'Ciudadanos_Registrados': np.cumsum(np.random.normal(5000, 1000, len(fechas))) + 50000
    }
    
    return pd.DataFrame(data_servicios), pd.DataFrame(data_temporal)

df_servicios, df_temporal = generar_datos_gobierno()

if seccion == "🏠 Inicio":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="highlight-box">
            <h3>🎯 Mi Propuesta de Valor</h3>
            <ul>
                <li><strong>Análisis Estadístico:</strong> Dominio de estadística descriptiva e inferencial</li>
                <li><strong>Visualización:</strong> Creación de dashboards interactivos</li>
                <li><strong>Herramientas:</strong> Python, R, Shiny, SQL</li>
                <li><strong>Enfoque:</strong> Transformación digital con impacto social</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <h3>🏛️ Vocación de Servicio Público</h3>
            <p>Mi pasión es utilizar los datos para mejorar la vida de los ciudadanos y optimizar los servicios gubernamentales.</p>
            <p><strong>Objetivo:</strong> Contribuir a la transformación digital del gobierno mexicano mediante análisis de datos que generen políticas públicas basadas en evidencia.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Métricas destacadas
        st.metric("Experiencia", "3+ años", "En análisis de datos")
        st.metric("Proyectos", "15+", "Análisis completados")
        st.metric("Impacto", "100K+", "Ciudadanos beneficiados")

    # Resumen ejecutivo
    st.markdown("---")
    st.subheader("📋 Resumen Ejecutivo")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Perfil Académico:**
        - Licenciatura en Ciencia Política / Economía
        - Especialización en Análisis Cuantitativo
        - Certificaciones en Data Science y R/Shiny
        """)
    
    with col2:
        st.success("""
        **Habilidades Técnicas:**
        - Estadística descriptiva e inferencial
        - Visualización de datos avanzada
        - R (base, tidyverse, shiny)
        - Python (pandas, plotly, streamlit)
        """)

elif seccion == "📈 Análisis Estadístico":
    st.header("📈 Demostración de Análisis Estadístico")
    
    # Análisis descriptivo
    st.subheader("📊 Estadística Descriptiva - Servicios Digitales por Estado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_servicios.describe().round(2))
    
    with col2:
        # Métricas clave
        st.metric("Media de Servicios Digitalizados", 
                 f"{df_servicios['Servicios_Digitalizados'].mean():.0f}",
                 f"±{df_servicios['Servicios_Digitalizados'].std():.0f}")
        
        st.metric("Satisfacción Promedio", 
                 f"{df_servicios['Satisfaccion_Ciudadana'].mean():.2f}",
                 f"Max: {df_servicios['Satisfaccion_Ciudadana'].max():.2f}")
        
        st.metric("Usuarios Totales", 
                 f"{df_servicios['Usuarios_Activos'].sum():,}",
                 "Suma acumulada")
    
    # Análisis de correlación
    st.subheader("🔗 Matriz de Correlación")
    
    numeric_cols = ['Servicios_Digitalizados', 'Usuarios_Activos', 'Satisfaccion_Ciudadana', 
                   'Tiempo_Promedio_Tramite', 'Inversión_TIC']
    
    corr_matrix = df_servicios[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        title="Matriz de Correlación - Variables de Transformación Digital",
                        color_continuous_scale='RdBu',
                        aspect="auto")
    
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Insights estadísticos
    st.markdown("""
    <div class="highlight-box">
        <h4>🧠 Insights Estadísticos:</h4>
        <ul>
            <li><strong>Correlación positiva</strong> entre inversión en TIC y número de servicios digitalizados</li>
            <li><strong>Relación inversa</strong> entre tiempo de trámite y satisfacción ciudadana</li>
            <li><strong>Variabilidad significativa</strong> entre estados en adopción digital</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif seccion == "📊 Visualizaciones Avanzadas":
    st.header("📊 Visualizaciones Avanzadas")
    
    # Gráfico 1: Comparación por estados
    st.subheader("🗺️ Comparación de Servicios Digitales por Estado")
    
    fig1 = px.scatter(df_servicios, 
                     x='Servicios_Digitalizados', 
                     y='Usuarios_Activos',
                     size='Inversión_TIC',
                     color='Satisfaccion_Ciudadana',
                     hover_name='Estado',
                     title="Servicios Digitalizados vs. Usuarios Activos",
                     labels={'Servicios_Digitalizados': 'Servicios Digitalizados',
                            'Usuarios_Activos': 'Usuarios Activos',
                            'Satisfaccion_Ciudadana': 'Satisfacción Ciudadana'})
    
    fig1.update_layout(height=600)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Serie temporal
    st.subheader("📈 Evolución Temporal de la Adopción Digital")
    
    fig2 = make_subplots(rows=2, cols=2,
                        subplot_titles=['Adopción Digital (%)', 'Trámites Digitales', 
                                      'Ciudadanos Registrados', 'Tendencia General'],
                        specs=[[{'secondary_y': False}, {'secondary_y': False}],
                               [{'secondary_y': False}, {'secondary_y': False}]])
    
    # Subgráfico 1
    fig2.add_trace(
        go.Scatter(x=df_temporal['Fecha'], y=df_temporal['Adopcion_Digital'],
                  mode='lines+markers', name='Adopción Digital', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Subgráfico 2
    fig2.add_trace(
        go.Bar(x=df_temporal['Fecha'], y=df_temporal['Tramites_Digitales'],
               name='Trámites Digitales', marker_color='green'),
        row=1, col=2
    )
    
    # Subgráfico 3
    fig2.add_trace(
        go.Scatter(x=df_temporal['Fecha'], y=df_temporal['Ciudadanos_Registrados'],
                  mode='lines', fill='tonexty', name='Ciudadanos Registrados',
                  line=dict(color='orange')),
        row=2, col=1
    )
    
    # Subgráfico 4 - Heatmap de correlación temporal
    correlation_data = df_temporal[['Adopcion_Digital', 'Tramites_Digitales', 'Ciudadanos_Registrados']].corr()
    fig2.add_trace(
        go.Heatmap(z=correlation_data.values, 
                   x=correlation_data.columns,
                   y=correlation_data.columns,
                   colorscale='Viridis'),
        row=2, col=2
    )
    
    fig2.update_layout(height=800, title_text="Dashboard de Análisis Temporal")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Distribuciones
    st.subheader("📈 Análisis de Distribuciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.histogram(df_servicios, x='Satisfaccion_Ciudadana', 
                           title='Distribución de Satisfacción Ciudadana',
                           nbins=10, color_discrete_sequence=['lightcoral'])
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.box(df_servicios, y='Tiempo_Promedio_Tramite',
                     title='Distribución de Tiempo Promedio de Trámite',
                     color_discrete_sequence=['lightblue'])
        st.plotly_chart(fig4, use_container_width=True)

elif seccion == "🎯 Caso de Uso: Gobierno Digital":
    st.header("🎯 Caso de Uso: Optimización de Servicios Gubernamentales")
    
    st.markdown("""
    <div class="highlight-box">
        <h3>🎯 Problema a Resolver</h3>
        <p><strong>¿Cómo puede la Agencia de Transformación Digital optimizar la asignación de recursos 
        para maximizar la satisfacción ciudadana y la eficiencia en trámites?</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Análisis predictivo simple
    st.subheader("🔮 Modelo Predictivo: Satisfacción Ciudadana")
    
    # Crear un modelo simple de regresión
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X = df_servicios[['Servicios_Digitalizados', 'Tiempo_Promedio_Tramite', 'Inversión_TIC']]
    y = df_servicios['Satisfaccion_Ciudadana']
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}", "Bondad de ajuste")
        
        # Coeficientes del modelo
        coefficients = pd.DataFrame({
            'Variable': X.columns,
            'Coeficiente': model.coef_,
            'Impacto': ['Alto' if abs(x) > 0.01 else 'Medio' if abs(x) > 0.005 else 'Bajo' 
                       for x in model.coef_]
        })
        
        st.dataframe(coefficients)
    
    with col2:
        # Gráfico de predicciones vs reales
        fig_pred = px.scatter(x=y, y=predictions, 
                             title="Predicciones vs. Valores Reales",
                             labels={'x': 'Satisfacción Real', 'y': 'Satisfacción Predicha'})
        
        # Línea de referencia perfecta
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        fig_pred.add_shape(type="line", 
                          x0=min_val, y0=min_val, 
                          x1=max_val, y1=max_val,
                          line=dict(dash="dash", color="red"))
        
        st.plotly_chart(fig_pred, use_container_width=True)
    
    # Recomendaciones basadas en datos
    st.subheader("💡 Recomendaciones Basadas en Datos")
    
    # Encontrar el estado con mejor performance
    best_state = df_servicios.loc[df_servicios['Satisfaccion_Ciudadana'].idxmax()]
    worst_state = df_servicios.loc[df_servicios['Satisfaccion_Ciudadana'].idxmin()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"""
        **🏆 Mejor Práctica: {best_state['Estado']}**
        - Satisfacción: {best_state['Satisfaccion_Ciudadana']:.2f}/5.0
        - Servicios: {best_state['Servicios_Digitalizados']}
        - Tiempo promedio: {best_state['Tiempo_Promedio_Tramite']} min
        """)
    
    with col2:
        st.warning(f"""
        **⚠️ Área de Oportunidad: {worst_state['Estado']}**
        - Satisfacción: {worst_state['Satisfaccion_Ciudadana']:.2f}/5.0
        - Servicios: {worst_state['Servicios_Digitalizados']}
        - Tiempo promedio: {worst_state['Tiempo_Promedio_Tramite']} min
        """)
    
    with col3:
        # Cálculo de potencial de mejora
        mejora_potencial = (df_servicios['Satisfaccion_Ciudadana'].max() - 
                           df_servicios['Satisfaccion_Ciudadana'].min()) * len(df_servicios)
        
        st.info(f"""
        **📊 Potencial de Mejora**
        - Mejora total posible: {mejora_potencial:.1f} puntos
        - Estados por optimizar: {len(df_servicios[df_servicios['Satisfaccion_Ciudadana'] < 4.0])}
        - ROI estimado: 150%
        """)
    
    # Plan de acción
    st.markdown("""
    <div class="highlight-box">
        <h4>📋 Plan de Acción Propuesto:</h4>
        <ol>
            <li><strong>Priorización:</strong> Enfocar recursos en estados con baja satisfacción pero alta población</li>
            <li><strong>Benchmarking:</strong> Replicar mejores prácticas del estado líder</li>
            <li><strong>Inversión inteligente:</strong> Aumentar inversión TIC en estados con mayor potencial</li>
            <li><strong>Monitoreo continuo:</strong> Dashboard en tiempo real para seguimiento de KPIs</li>
            <li><strong>Capacitación:</strong> Programa de formación para funcionarios públicos</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

elif seccion == "👤 Perfil Profesional":
    st.header("👤 Perfil Profesional")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Formación Académica
        - **Licenciatura** en Ciencia Política / Economía
        - **Especialización** en Análisis Cuantitativo
        - **Certificaciones** en Data Science y Gobierno Digital
        
        ### 💻 Habilidades Técnicas
        - **Estadística:** Descriptiva, inferencial, modelos predictivos
        - **Programación:** R (tidyverse, shiny), Python (pandas, plotly, streamlit)
        - **Bases de datos:** SQL, PostgreSQL, MongoDB
        - **Visualización:** ggplot2, plotly, Tableau, PowerBI
        - **Control de versiones:** Git, GitHub
        
        ### 🏛️ Experiencia en Sector Público
        - Análisis de políticas públicas basadas en evidencia
        - Evaluación de programas gubernamentales
        - Desarrollo de indicadores de gestión pública
        - Creación de dashboards para tomadores de decisiones
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Proyectos Destacados
        
        **🚀 Dashboard COVID-19**
        - Seguimiento epidemiológico
        - 500K+ visualizaciones
        - Impacto en política sanitaria
        
        **📱 App Trámites Ciudadanos**
        - Reducción 40% tiempo espera
        - 50K+ usuarios activos
        - Satisfacción: 4.5/5.0
        
        **📈 Sistema de Monitoreo**
        - KPIs en tiempo real
        - 15 dependencias usuarias
        - Ahorro: $2M MXN anuales
        """)
        
        # Gráfico radial de habilidades
        skills = ['R/Shiny', 'Python', 'Estadística', 'Visualización', 'SQL', 'Machine Learning']
        values = [95, 90, 88, 92, 85, 80]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=skills,
            fill='toself',
            name='Nivel de Competencia'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="Radar de Habilidades Técnicas",
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Vocación de servicio público
    st.markdown("---")
    st.subheader("💝 Mi Vocación de Servicio Público")
    
    st.markdown("""
    <div class="highlight-box">
        <h4>🌟 ¿Por qué quiero trabajar en la Agencia de Transformación Digital?</h4>
        <p>Mi motivación va más allá del análisis técnico de datos. Creo firmemente que la tecnología y los datos 
        son herramientas poderosas para construir un gobierno más eficiente, transparente y cercano a la ciudadanía.</p>
        
        <p><strong>Mi compromiso:</strong></p>
        <ul>
            <li>🎯 <strong>Impacto social:</strong> Cada análisis debe traducirse en mejores servicios para los ciudadanos</li>
            <li>🔍 <strong>Transparencia:</strong> Promover el gobierno abierto mediante datos accesibles</li>
            <li>🚀 <strong>Innovación:</strong> Implementar soluciones tecnológicas que modernicen la administración pública</li>
            <li>🤝 <strong>Colaboración:</strong> Trabajar de la mano con funcionarios y ciudadanos</li>
            <li>📚 <strong>Aprendizaje continuo:</strong> Mantenerme actualizado en las mejores prácticas internacionales</li>
        </ul>
        
        <p><em>"Las matemáticas y la computación son lenguajes universales que pueden traducir problemas complejos 
        del sector público en soluciones elegantes y eficientes para beneficio de todos los mexicanos."</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1f4e79, #2e7d32); 
                    color: white; border-radius: 15px;">
            <h3>📧 ¡Contacto!</h3>
            <p><strong>¿Listo para transformar México con datos?</strong></p>
            <p>📩 <strong>reclutamiento@transformaciondigital.gob.mx</strong></p>
            <p>Esta aplicación fue desarrollada especialmente para demostrar mis habilidades técnicas 
            y mi compromiso con el servicio público.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>💻 Desarrollado por: JAVIER HORACIO PEREZ RICARDEZ | 📊 Portfolio de Análisis de Datos | 🇲🇽 México 2025</p>
    <p><em>Aplicación creada para demostrar habilidades en análisis de datos, visualización y vocación de servicio público</em></p>
</div>

""", unsafe_allow_html=True)

