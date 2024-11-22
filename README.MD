# AquaChile Document Retrieval and Q&A System

Este proyecto es un sistema **RAG** de preguntas y respuestas (Q&A) para AquaChile. Utiliza varias herramientas y bibliotecas para proporcionar respuestas a preguntas basadas en documentos reglamentarios de la empresa.

## Instalación

1. Clona el repositorio:

    ```sh
    git clone https://github.com/agutierrezmorag/sharepoint-loader.git
    cd sharepoint-loader
    ```

2. Crea y activa un entorno virtual:

    ```sh
    python -m venv venv
    source venv/bin/activate    # En unix
    venv\Scripts\activate       # En Windows
    ```

3. Instala las dependencias:

    ```sh
    pip install -r requirements.txt
    ```

4. Actualiza el archivo `example.env` con los valores correspondientes a las variables de entorno y renombra el archivo a `.env`.

## Uso

### Ingesta de Documentos

1. Abre el archivo `doc_ingestion/doc_loader.ipynb` en Jupyter Notebook o JupyterLab.
2. Ejecuta todas las celdas para cargar y procesar los documentos desde SharePoint.

**Solo ejecutar la ingesta de documentos cuando se suban nuevos documentos en el sharepoint *Aqua_PDF***.

### Ejecución de la Aplicación

1. Ejecuta la aplicación Streamlit:

    ```sh
    streamlit run app.py
    ```

2. Abre tu navegador web y navega a `http://localhost:8501` para interactuar con el asistente virtual.
