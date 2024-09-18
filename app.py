import os
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from flask import Flask, request
from flask_cors import CORS, cross_origin


api_key = os.getenv("GEMINI_API_KEY")
index_storage_dir = "index_storage"
data_dir = "data"

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=api_key
)

Settings.llm = Gemini(api_key=api_key,safety_settings=safety_settings, temperature=0)


if not os.path.exists(index_storage_dir):
    print("Creating new index...")
    # load the documents and create the index
    documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=index_storage_dir)
else:
    # load the existing index
    print("Loading existing index...")
    storage_context = StorageContext.from_defaults(persist_dir=index_storage_dir)
    index = load_index_from_storage(storage_context)
    
print("Index loaded successfully.")

memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        """You are a friendly chatbot, able to have normal interactions.
        You help the users with their questions. Return answers from the stored documents only.
        Do not make up your own answers.
        Summarize the answers in less than or up to three sentences. Return shorter response as possible. """
    ),
)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def home():
    return "Web Scraping API"

@app.route('/web', methods=['GET','POST'])
@cross_origin()
def get_bot_response():
    prompt = request.args.get('prompt')
    print("Received prompt:", prompt)
    response = chat_engine.chat(prompt)
    output_dict = {"response": response.response}
    output_json = json.dumps(output_dict)
    return output_json

if __name__ == "__main__":
    app.run(debug=True, port= 9090)