from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

app = Flask(__name__)

# Groq client
client = Groq(
    api_key="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"
)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""

    if request.method == "POST":

        user_query = request.form["query"]

        # Search relevant docs
        docs = db.similarity_search(
            user_query,
            k=1
        )

        # If no docs found
        if not docs:
            answer = "Escalating to human agent."

        else:

            context = docs[0].page_content

            # Simple escalation check
            if len(context.strip()) < 20:

                answer = "Escalating to human agent."

            else:

                prompt = f"""
You are a customer support assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{user_query}
"""

                # Updated Groq model
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3
                )

                answer = response.choices[0].message.content

    return render_template(
        "index.html",
        answer=answer
    )

if __name__ == "__main__":
    app.run(debug=True)