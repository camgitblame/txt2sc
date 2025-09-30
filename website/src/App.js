import logo from "./logo.svg";
import * as styles from "./App.css.ts";
import { Carousel } from "antd";
import "./App.css";
import { Github, ScrollText } from "lucide-react";

function App() {
  const onChange = (currentSlide) => {
    console.log(currentSlide);
  };

  return (
    <div className={styles.app}>
      <h1>Text2Scene: Text-Driven Movie Scene Generation</h1>
      <h2>Project Page</h2>
      <div
        style={{
          margin: "30px 0",
          textAlign: "center",
          display: "flex",
          justifyContent: "center",
          gap: "30px",
        }}
      >
        <a
          href="https://github.com/camgitblame/txt2sc"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: "inherit",
            textDecoration: "none",
            display: "inline-flex",
            alignItems: "center",
            gap: "4px",
          }}
        >
          <Github /> Github
        </a>
        <a
          href="#"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: "inherit",
            textDecoration: "none",
            display: "inline-flex",
            alignItems: "center",
            gap: "4px",
          }}
        >
          <ScrollText /> Paper
        </a>
      </div>
      <div className={styles.nameContainer}>
        <div className={styles.nameDiv}>
          <h3>Cam Nguyen</h3>
          <code>camng44@gmail.com</code>
        </div>
      </div>
      <div>
        <div className={styles.abstract}>
          <h2>Abstract</h2>
          Text2Scene is a novel approach for generating cinematic movie scenes from text descriptions. 
          Our method leverages advanced diffusion models and 3D scene understanding to create immersive, 
          story-driven visual experiences. By combining text-to-image generation with spatial consistency 
          and temporal coherence, we enable the creation of dynamic movie scenes that faithfully represent 
          narrative descriptions while maintaining visual quality and cinematic appeal.
        </div>
      </div>
      <div style={{ margin: "50px 0", textAlign: "center" }}>
        <h2>Coming Soon</h2>
        <p>More content and examples will be added soon.</p>
      </div>
    </div>
  );
}

export default App;
