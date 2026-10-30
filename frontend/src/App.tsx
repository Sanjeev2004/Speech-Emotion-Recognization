import { useState } from "react";
import About from "./components/About";
import Analyze from "./components/Analyze";
import Compare from "./components/Compare";

const TABS = { analyze: "Analyze", compare: "Model comparison", about: "How it works" } as const;
type Tab = keyof typeof TABS;
const GITHUB = "https://github.com/Sanjeev2004/Speech-Emotion-Recognization";

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (location.hash.slice(1) as Tab) in TABS ? (location.hash.slice(1) as Tab) : "analyze");
  const go = (t: Tab) => {
    setTab(t);
    history.replaceState(null, "", `#${t}`);
  };

  return (
    <>
      <header>
        <div className="wrap head">
          <div>
            <h1>Speech Emotion Recognition</h1>
            <p className="muted">Detect neutral, happy, sad, angry, fear or disgust from a voice clip.</p>
          </div>
          <nav>
            {(Object.keys(TABS) as Tab[]).map((t) => (
              <button key={t} className={t === tab ? "on" : ""} onClick={() => go(t)}>
                {TABS[t]}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="wrap">
        {tab === "analyze" && <Analyze />}
        {tab === "compare" && <Compare />}
        {tab === "about" && <About />}
      </main>
      <footer className="wrap muted small">
        Trained on RAVDESS and CREMA-D. Audio is processed for the prediction and not stored.{" "}
        <a href={GITHUB} target="_blank" rel="noreferrer">
          Source on GitHub
        </a>
      </footer>
    </>
  );
}
