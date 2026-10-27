const STEPS = [
  ["Input", "Browser records or decodes the file, converts it to 16 kHz mono WAV, and sends it to the API."],
  ["Pre-processing", "Silence is trimmed, volume is normalised, and audio is cut into 3-second windows."],
  ["Features", "Each window becomes a log-mel spectrogram: a 64 x 301 picture of which frequencies are loud over time."],
  ["Model", "An ONNX model turns the spectrogram into a probability for each of the 6 emotions."],
  ["Output", "Probabilities are averaged over all windows; the highest one is the predicted emotion."],
];

const MODELS = [
  ["SVM, Random Forest", "Classical ML on 208 hand-crafted numbers (mean and spread of MFCC, chroma, spectral shape, energy)."],
  ["BiLSTM", "Reads the spectrogram frame by frame, like a sentence, with attention on key moments."],
  ["CNN", "Treats the spectrogram as an image and finds local patterns."],
  ["CNN-LSTM", "CNN for local patterns plus LSTM for how they change over time."],
  ["CRNN + Attention", "CNN plus GRU plus attention pooling."],
  ["wav2vec2, HuBERT, WavLM", "Large transformers pre-trained on hundreds to thousands of hours of unlabelled speech, then fine-tuned for emotion."],
];

export default function About() {
  return (
    <div className="grid">
      <section className="card">
        <h2>How a prediction is made</h2>
        <ol className="steps">
          {STEPS.map(([t, d]) => (
            <li key={t}>
              <b>{t}</b>
              <span className="muted">{d}</span>
            </li>
          ))}
        </ol>
      </section>
      <section className="card">
        <h2>Models compared</h2>
        <dl className="defs">
          {MODELS.map(([t, d]) => (
            <div key={t}>
              <dt>{t}</dt>
              <dd className="muted">{d}</dd>
            </div>
          ))}
        </dl>
        <h2>Datasets</h2>
        <dl className="defs">
          <div>
            <dt>RAVDESS</dt>
            <dd className="muted">24 professional actors, 1,056 clips used (calm and surprised removed).</dd>
          </div>
          <div>
            <dt>CREMA-D</dt>
            <dd className="muted">91 actors of different ages and ethnicities, 7,442 clips.</dd>
          </div>
        </dl>
      </section>
    </div>
  );
}
