# AquaChile Document Retrieval and Q&A System

Este proyecto es un sistema **RAG** de preguntas y respuestas (Q&A) para AquaChile. Utiliza varias herramientas y bibliotecas para proporcionar respuestas a preguntas basadas en documentos reglamentarios de la empresa.

## Instalación

1. Clona el repositorio:

    ```sh
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
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

4. **Solo en Windows**: ejecutar el siguiente comando:

    ```sh
    pip install pywin32==308
    ```

5. Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

    ```env
    O365_CLIENT_ID=""
    O365_CLIENT_SECRET=""
    DOCUMENT_LIBRARY_ID=""
    AZURE_OPENAI_API_KEY=""
    AZURE_OPENAI_ENDPOINT=""
    OPENAI_API_VERSION=""
    AZURE_AI_SEARCH_SERVICE_NAME=""
    AZURE_AI_SEARCH_ENDPOINT=""
    AZURE_AI_SEARCH_API_KEY=""
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT=""
    LANGCHAIN_API_KEY=""
    LANGCHAIN_PROJECT="Sharepoint Q&A"
    ```

## Uso

### Ingesta de Documentos

1. Abre el archivo `doc_ingestion/doc_loader.ipynb` en Jupyter Notebook o JupyterLab.
2. Ejecuta todas las celdas para cargar y procesar los documentos desde SharePoint.

### Ejecución de la Aplicación

1. Ejecuta la aplicación Streamlit:

    ```sh
    streamlit run app.py
    ```

2. Abre tu navegador web y navega a `http://localhost:8501` para interactuar con el asistente virtual.

## Funciones Principales

### `utils/langchain_utils.py`

- `get_agent()`: Configura y devuelve un ejecutor de agente para manejar las consultas.
- `set_tracer(project_name)`: Configura un tracer de [LangSmith](https://www.langchain.com/langsmith) para el proyecto.
- `_get_llm()`: Configura y devuelve un modelo de lenguaje de Azure OpenAI.
- `_get_tools()`: Configura y devuelve una lista de herramientas para la recuperación de documentos.
- `_get_agent_prompt()`: Recupera el prompt del agente desde *LangChain Hub*.
- `q_suggestion_chain()`: Configura y devuelve una cadena de sugerencias de preguntas. *Sin implementar*.

### `app.py`

- `answer_question(question, agent_thoughts_placeholder, response_placeholder)`: Maneja la lógica asincrónica para responder preguntas utilizando el agente configurado.
- `submit_question(question)`: Envía la pregunta del usuario al sistema.
