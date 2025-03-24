# LLM-Powered Booking Analytics & QA System

## 🚀 Project Overview  
This project is a **FastAPI-based Booking Analytics & Question Answering (QA) System** that:  
- 📊 **Processes hotel booking data** to extract insights.  
- 🔍 **Implements analytics** (e.g., revenue trends, cancellation rates, geographical distribution).  
- 🤖 **Uses Retrieval-Augmented Generation (RAG)** with a vector database for intelligent Q&A.  
- 🌐 **Exposes a REST API** for querying analytics and retrieving answers.  
- ☁️ **Deployed on Render** for accessibility.  

---

## 🛠️ Features  
✅ **Data Preprocessing** – Cleans and structures hotel booking data.  
✅ **Analytics & Reporting** – Extracts key insights like revenue trends and cancellation rates.  
✅ **RAG-based QA System** – Uses **FAISS/ChromaDB** for fast Q&A retrieval.  
✅ **FastAPI REST API** – Provides endpoints for analytics & question answering.  
✅ **Easy Deployment** – Can be deployed on **Render or any cloud service**.  

---

## 📌 Technologies Used  
- **FastAPI** – For the REST API.  
- **Python (Pandas, NumPy, Matplotlib, Seaborn)** – For data processing & analytics.  
- **FAISS/ChromaDB/Weaviate** – For vector-based search in RAG.  
- **Sentence Transformers** – To generate embeddings for Q&A.  
- **Uvicorn** – To run the FastAPI server.  
- **Docker (Optional)** – For containerization.  

---

## 🖥️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2️⃣ Create a Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the API Locally  
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Now, the API should be accessible at:  
👉 **http://127.0.0.1:8000**

---

## 🌍 API Endpoints  

| Endpoint   | Method | Description |
|------------|--------|-------------|
| `/analytics` | **POST** | Returns booking analytics insights. |
| `/ask` | **POST** | Answers booking-related queries using LLM & RAG. |

### 🔹 Example Query for `/ask`
```json
{
  "question": "Which locations had the highest booking cancellations?"
}
```

### 🔹 Example Response
```json
{
  "answer": ["New York", "Los Angeles", "Paris", "London", "Tokyo"]
}
```

---

## 🚀 Deploying on Render  

### 1️⃣ Push Your Code to GitHub  
Ensure your latest code is pushed to GitHub:  
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2️⃣ Create a New Web Service on Render  
- Go to **Render** and create a **New Web Service**.
- Select your **GitHub repository** and deploy.
- Use the following **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- Use the following **Start Command**:
  ```bash
  uvicorn api:app --host 0.0.0.0 --port 8000
  ```

### 3️⃣ Get the Public URL  
Once deployed, Render will provide a **public URL** like:  
👉 **https://your-app-name.onrender.com**  

Now, you can make API requests using this **public URL**.

---

## ⚡ Challenges Faced & Implementation Choices  

### ✅ Challenges Faced  
- **Data Cleaning** – Handling missing values and inconsistent data formats.  
- **Optimizing RAG Performance** – Fine-tuning the vector database for fast retrieval.  
- **LLM Integration** – Selecting an open-source LLM for accurate answers.  
- **Deployment Issues** – Managing dependency versions for smooth deployment.  

### ✅ Implementation Choices  
- Used **FAISS** for efficient vector search.  
- Stored insights in **JSON format** for quick retrieval in API responses.  
- Used **FastAPI** for its **asynchronous capabilities** and fast performance.  

---

## 🔮 Future Improvements  
🚀 **Enhance LLM Performance** – Fine-tune the model for better accuracy.  
🚀 **Add More Analytics** – Include seasonal trends, customer segmentation, etc.  
🚀 **Database Integration** – Replace JSON storage with a SQL/NoSQL database.  

---

## 👨‍💻 Contributor  
**Tanishq Mahapatra**  

---

## 📝 License  
📜 **MIT License** – Open for modifications and improvements.  
