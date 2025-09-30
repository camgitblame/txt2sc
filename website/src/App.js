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
        <button
          onClick={() => alert("Paper coming soon!")}
          style={{
            background: "none",
            border: "none",
            color: "inherit",
            textDecoration: "none",
            display: "inline-flex",
            alignItems: "center",
            gap: "4px",
            cursor: "pointer",
            fontSize: "inherit",
            fontFamily: "inherit",
          }}
        >
          <ScrollText /> Paper
        </button>
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
          Our project presents a text-to-3D scene pipeline that generates aesthetically pleasing, perpetual walkthroughs aligned with the visual style of target films. 
          Building on the SceneScape baseline, we fine-tune Stable Diffusion with DreamBooth for few-shot, scene-focused synthesis that captures each film’s color palette, materials, and set dressing. 
          To keep geometry stable under camera motion, we guide inpainting with a multi-ControlNet setup that conditions masks using ControlNet-Inpaint and ControlNet-Depth. At test time, we add four lightweight stabilizers, namely EMA-smoothed depth, seam-aware mask morphology, immediate mesh accumulation, and a short camera-motion warm-up, which improves structural consistency over long sequences.

          We evaluate on five stylistically distinct movies. In pairwise comparisons, human experts and GPT-4V prefer our outputs over the baseline for film likeness, visual quality, 3D structural consistency, and prompt alignment. Quantitatively, CLIP-AS and reconstructed 3D density increase, indicating more appealing frames and fuller coverage, while reprojection error and CLIP-TS remain comparable to SceneScape. Overall, our results improve on the baseline and provide a practical route to film-specific, 3D-plausible walkthroughs that require no 3D or multiview training data.
        </div>
      </div>
      <Carousel arrows className={styles.carousel} afterChange={onChange}>
        <div className={styles.contentStyle}>
          <h2>
            Scene from <span className={styles.italic}>Alien</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>Few-shot training images:</h3>
              <div className={styles.placeholderImage}>
                <p>Training Image 1</p>
                <small>Placeholder for Alien training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 2</p>
                <small>Placeholder for Alien training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 3</p>
                <small>Placeholder for Alien training data</small>
              </div>
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated walkthrough:</h3>
              <div className={styles.placeholderVideo}>
                <p>🎬 Video Placeholder</p>
                <small>Generated Alien scene walkthrough</small>
              </div>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            Patrick Bateman's apartment from{" "}
            <span className={styles.italic}>American Psycho</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>Few-shot training images:</h3>
              <div className={styles.placeholderImage}>
                <p>Training Image 1</p>
                <small>Placeholder for American Psycho training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 2</p>
                <small>Placeholder for American Psycho training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 3</p>
                <small>Placeholder for American Psycho training data</small>
              </div>
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated walkthrough:</h3>
              <div className={styles.placeholderVideo}>
                <p>🎬 Video Placeholder</p>
                <small>Generated American Psycho scene walkthrough</small>
              </div>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            The Overlook Hotel from <span className={styles.italic}>The Shining</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>Few-shot training images:</h3>
              <div className={styles.placeholderImage}>
                <p>Training Image 1</p>
                <small>Placeholder for The Shining training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 2</p>
                <small>Placeholder for The Shining training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 3</p>
                <small>Placeholder for The Shining training data</small>
              </div>
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated walkthrough:</h3>
              <div className={styles.placeholderVideo}>
                <p>🎬 Video Placeholder</p>
                <small>Generated The Shining scene walkthrough</small>
              </div>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            Spaceship interior from <span className={styles.italic}>Passengers</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>Few-shot training images:</h3>
              <div className={styles.placeholderImage}>
                <p>Training Image 1</p>
                <small>Placeholder for Passengers training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 2</p>
                <small>Placeholder for Passengers training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 3</p>
                <small>Placeholder for Passengers training data</small>
              </div>
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated walkthrough:</h3>
              <div className={styles.placeholderVideo}>
                <p>🎬 Video Placeholder</p>
                <small>Generated Passengers scene walkthrough</small>
              </div>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            Beauty clinic from <span className={styles.italic}>The Substance</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>Few-shot training images:</h3>
              <div className={styles.placeholderImage}>
                <p>Training Image 1</p>
                <small>Placeholder for The Substance training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 2</p>
                <small>Placeholder for The Substance training data</small>
              </div>
              <div className={styles.placeholderImage}>
                <p>Training Image 3</p>
                <small>Placeholder for The Substance training data</small>
              </div>
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated walkthrough:</h3>
              <div className={styles.placeholderVideo}>
                <p>🎬 Video Placeholder</p>
                <small>Generated The Substance scene walkthrough</small>
              </div>
            </div>
          </div>
        </div>
      </Carousel>
    </div>
  );
}

export default App;
