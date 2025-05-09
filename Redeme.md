# 📊 Vector Spaces and Subspaces Visualizer (Streamlit App)

This interactive Streamlit app demonstrates the core concepts of **Vector Spaces** and **Subspaces** in Linear Algebra. It allows you to explore closure properties, test for subspace criteria, and visualize vector span relationships step-by-step.

---

## 🚀 Features

- ✅ **Input Custom Vectors in ℝⁿ**
- 🔄 Check for:
  - Closure under vector **addition**
  - Closure under **scalar multiplication**
  - Presence of the **zero vector**
- 📌 Determine if the set of vectors forms a **subspace**
- 🧠 Test whether a given vector lies in the **span**
- 🧮 View the **linear combination** that produces it (if it exists)
- 📐 Optional: Extendable to include 3D visualization for ℝ³ (not enabled by default)

---

## 🛠️ Installation & Running (Step-by-Step)

Follow these steps to get the app running on your machine:

✅ 1. On Windows:
```bash
python -m venv .venv

✅ 2. Activate the virtual environment
On Windows:

```bash
.venv\Scripts\activate

📦 4. Install dependencies

```bash
pip install streamlit numpy

▶️ 5. Run the Streamlit app

```bash
python -m streamlit run vector_space_app.py


### 🔁 1. Clone the repository

```bash
git clone https://github.com/yourusername/vector-space-visualizer.git
cd vector-space-visualizer
