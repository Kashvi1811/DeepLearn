# DeepLearn

A premium **interactive deep learning toolbox** built with Streamlit. Master neural networks through hands-on visualization and real-time simulation.

## Features

- **6 Interactive Neural Network Modules:**
  - **Perceptron** — See how a linear separator evolves epoch-by-epoch
  - **Multi-Layer Perceptron (MLP)** — Explore non-linear decision boundaries
  - **Sentiment Analysis** — Classify text as positive, neutral, or negative with token highlighting
  - **Computer Vision** — Recognize handwritten digits using a CNN (TensorFlow) or sklearn fallback
  - **Hopfield Network** — Watch associative memory restore corrupted patterns
  - **LSTM** — Learn temporal patterns through sequence prediction

- **Step-by-Step Simulations** — Use sliders to inspect model behavior at each training epoch
- **Live Visualizations** — Plotly charts update in real-time as you adjust hyperparameters
- **Premium UI** — Dark mode with gradient accents, smooth animations, and premium typography
- **Offline-Ready** — All models train locally; no external API calls

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kashvi1811/DeepLearn.git
cd DeepLearn
```

2. Create a Python 3.11+ virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and explore the modules!

## Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push this repo to GitHub (already done ✓)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New App"**
4. Select:
   - **Repository:** `Kashvi1811/DeepLearn`
   - **Branch:** `master`
   - **Main file path:** `app.py`
5. Click **"Deploy"** — the app will be live in ~2-3 minutes

### Option 2: Other Platforms

- **Heroku:** Use `Procfile` with `streamlit run app.py`
- **Railway:** Deploy directly from GitHub
- **Docker:** Build and run with a Dockerfile

## Architecture

- **app.py** — Single-file Streamlit application with:
  - Premium CSS theming (dark mode, gradients, fonts)
  - 6 interactive neural network demos
  - Real-time epoch snapshots and simulations
  - Optional TensorFlow CNN fallback

- **requirements.txt** — All dependencies pinned for reproducibility

## Technologies

- **Framework:** Streamlit (interactive web apps)
- **ML/DL:** scikit-learn, TensorFlow/Keras
- **Visualization:** Plotly, Matplotlib
- **Image Processing:** Pillow, SciPy

## Learning Path

1. Start with **Home** — overview and quick launch buttons
2. Try **Perceptron** — easiest, shows linear separation
3. Move to **MLP** — understand non-linearity
4. Explore **Sentiment Analysis** — text classification in action
5. Advance to **Computer Vision** — digit recognition pipeline
6. Experiment with **Hopfield Network** — associative memory concepts
7. Master **LSTM** — temporal patterns and sequence models

## Author

**Kashvi** — Interactive Deep Learning Toolbox  
[GitHub](https://github.com/Kashvi1811) | [Email](mailto:kashvisoni2005@gmail.com)

## License

MIT — Feel free to fork, modify, and share!

---

**Happy learning! 🧠✨**
