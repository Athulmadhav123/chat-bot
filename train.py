from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load document
loader = TextLoader("docs/company_policy.txt")

documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in FAISS
db = FAISS.from_documents(docs, embeddings)

# Save FAISS index
db.save_local("faiss_index")

print("Training completed!")