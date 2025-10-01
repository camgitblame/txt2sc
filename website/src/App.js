import * as styles from "./App.css.ts";
import { Carousel } from "antd";
import "./App.css";
import { Github, ScrollText, Play } from "lucide-react";
import { useState } from "react";

// Import training images
import alienTrain1 from "./assets/alien/train1.jpg";
import alienTrain2 from "./assets/alien/train2.jpg";
import alienTrain3 from "./assets/alien/train3.jpg";

import asTrain1 from "./assets/american_psycho/train1.jpg";
import asTrain2 from "./assets/american_psycho/train2.jpg";
import asTrain3 from "./assets/american_psycho/train3.png";

import shiningTrain1 from "./assets/the_shining/train1.jpg";
import shiningTrain2 from "./assets/the_shining/train2.jpg";
import shiningTrain3 from "./assets/the_shining/train3.jpg";

import pasTrain1 from "./assets/passengers/train1.jpg";
import pasTrain2 from "./assets/passengers/train2.jpg";
import pasTrain3 from "./assets/passengers/train3.jpg";

import subTrain1 from "./assets/substance/train1.jpg";
import subTrain2 from "./assets/substance/train2.jpg";
import subTrain3 from "./assets/substance/train3.jpg";

// Import videos
import alienVideo from "./assets/alien/video_alien.mp4";
import asVideo from "./assets/american_psycho/video_american_psycho.mp4";
import shiningVideo from "./assets/the_shining/video_the_shining.mp4";
import passengersVideo from "./assets/passengers/video_passengers.mp4";
import substanceVideo from "./assets/substance/video_the_substance.mp4";

function App() {
  const [currentView, setCurrentView] = useState('main');
  
  const onChange = (currentSlide) => {
    console.log(currentSlide);
  };

  const renderResultsPage = () => {
    const movies = [
      { name: "The Shining", subtitle: "The Overlook Hotel hallway" },
      { name: "The Substance", subtitle: "Elisabeth Sparkle's apartment" },
      { name: "American Psycho", subtitle: "Patrick Bateman's apartment" },
      { name: "Passengers", subtitle: "The Vienna Suite" },
      { name: "Alien", subtitle: "The Nostromo corridor" }
    ];

    const variants = ["Baseline", "DB+1CN", "DB+2CN"];

    return (
      <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "20px" }}>
        <div className={styles.abstract} style={{ marginBottom: "40px" }}>
          <h2>Results Comparison</h2>
          <p style={{ marginBottom: "20px" }}>
            Below are the generated walkthrough videos for all five movies using three different methods:
          </p>
          <ul style={{ textAlign: "left", margin: "0", padding: "0 20px" }}>
            <li><strong>Baseline</strong>: SceneScape (original method)</li>
            <li><strong>DB+1CN</strong>: DreamBooth + ControlNet Inpaint</li>
            <li><strong>DB+2CN</strong>: DreamBooth + ControlNet Inpaint + ControlNet Depth</li>
          </ul>
        </div>

        {movies.map((movie, movieIndex) => (
          <div key={movie.name} style={{ marginBottom: "60px" }}>
            <h2 style={{ textAlign: "center", marginBottom: "30px" }}>
              {movie.name} - <span className={styles.italic}>{movie.subtitle}</span>
            </h2>
            <div style={{ 
              display: "grid", 
              gridTemplateColumns: "repeat(3, 1fr)", 
              gap: "20px",
              maxWidth: "1200px",
              margin: "0 auto"
            }}>
              {variants.map((variant, variantIndex) => (
                <div key={variant} style={{ textAlign: "center" }}>
                  <h3 style={{ marginBottom: "15px", fontSize: "18px" }}>{variant}</h3>
                  <div style={{
                    width: "100%",
                    height: "250px",
                    backgroundColor: "#404040",
                    border: "2px dashed #666",
                    borderRadius: "8px",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#ccc",
                    fontSize: "14px"
                  }}>
                    <div style={{ fontSize: "40px", marginBottom: "10px" }}>🎬</div>
                    <div>Video Placeholder</div>
                    <div style={{ fontSize: "12px", marginTop: "5px" }}>
                      {movie.name} - {variant}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className={styles.app}>
      <h1>Text-to-3D Scene Generation for Movie Walkthroughs</h1>
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
        <button
          onClick={() => setCurrentView(currentView === 'main' ? 'results' : 'main')}
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
          <Play /> {currentView === 'main' ? 'Results' : 'Back to Main'}
        </button>
      </div>
      
      {currentView === 'main' ? (
        <>
          <div className={styles.nameContainer}>
        <div className={styles.nameDiv}>
          <h3>Cam Nguyen</h3>
          <code>camng44@gmail.com</code>
        </div>
      </div>
      <div>
        <div className={styles.abstract}>
          <h2>Abstract</h2>
          We present a text-to-3D pipeline that generates aesthetically pleasing, perpetual scene walkthroughs aligned with the visual identity of target films. Building on the SceneScape baseline, we fine-tune Stable Diffusion with DreamBooth for few-shot, scene-focused synthesis that recreates each film’s color palette, materials, and set dressing.
          {/* <br /><br/> */}

          To maintain stable geometry under camera motion, we guide inpainting with a multi-ControlNet setup that conditions masks using ControlNet-Inpaint and ControlNet-Depth. At test time, we add four lightweight stabilizers, namely EMA-smoothing for depth, seam-aware mask morphology, immediate mesh accumulation, and a short camera-motion warm-up, which improves structural consistency over long video sequences.
          {/* <br /><br /> */}

          We evaluate the results on five stylistically distinct movies. In qualitative analysis, both human experts and GPT-4V strongly prefer our outputs over the baseline for film likeness, visual quality, 3D structural consistency, and prompt alignment. Quantitatively, CLIP-AS and reconstructed 3D density increase over the baseline, indicating more appealing frames and fuller coverage, while reprojection error and CLIP-TS remain comparable to SceneScape. Overall, our results improve on the baseline and provide a practical path to film-specific, 3D-plausible walkthroughs that require no 3D or multiview training data.
        </div>
      </div>
      <Carousel arrows className={styles.carousel} afterChange={onChange}>
        <div className={styles.contentStyle}>
          <h2>
            The Nostromo corridor from <span className={styles.italic}>Alien (1979)</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>DreamBooth Training Images</h3>
              <img 
                src={alienTrain1} 
                alt="Alien training 1" 
                className={styles.trainingImage}
              />
              <img 
                src={alienTrain2} 
                alt="Alien training 2" 
                className={styles.trainingImage}
              />
              <img 
                src={alienTrain3} 
                alt="Alien training 3" 
                className={styles.trainingImage}
              />
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated Walkthrough Video</h3>
              <video 
                src={alienVideo}
                controls
                muted
                loop
                style={{ width: '100%', maxWidth: '500px', height: 'auto' }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            Patrick Bateman's apartment from{" "}
            <span className={styles.italic}>American Psycho (2000)</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>DreamBooth Training Images</h3>
              <img 
                src={asTrain1} 
                alt="American Psycho training 1" 
                className={styles.trainingImage}
              />
              <img 
                src={asTrain2} 
                alt="American Psycho training 2" 
                className={styles.trainingImage}
              />
              <img 
                src={asTrain3} 
                alt="American Psycho training 3" 
                className={styles.trainingImage}
              />
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated Walkthrough Video</h3>
              <video 
                src={asVideo}
                controls
                muted
                loop
                style={{ width: '100%', maxWidth: '500px', height: 'auto' }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            The Overlook Hotel hallway from <span className={styles.italic}>The Shining (1980)</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>DreamBooth Training Images</h3>
              <img 
                src={shiningTrain1} 
                alt="The Shining training 1" 
                className={styles.trainingImage}
              />
              <img 
                src={shiningTrain2} 
                alt="The Shining training 2" 
                className={styles.trainingImage}
              />
              <img 
                src={shiningTrain3} 
                alt="The Shining training 3" 
                className={styles.trainingImage}
              />
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated Walkthrough Video</h3>
              <video 
                src={shiningVideo}
                controls
                muted
                loop
                style={{ width: '100%', maxWidth: '500px', height: 'auto' }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            The Vienna Suite from <span className={styles.italic}>Passengers (2016)</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>DreamBooth Training Images</h3>
              <img 
                src={pasTrain1} 
                alt="Passengers training 1" 
                className={styles.trainingImage}
              />
              <img 
                src={pasTrain2} 
                alt="Passengers training 2" 
                className={styles.trainingImage}
              />
              <img 
                src={pasTrain3} 
                alt="Passengers training 3" 
                className={styles.trainingImage}
              />
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated Walkthrough Video</h3>
              <video 
                src={passengersVideo}
                controls
                muted
                loop
                style={{ width: '100%', maxWidth: '500px', height: 'auto' }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
        <div className={styles.contentStyle}>
          <h2>
            Elisabeth Sparkle’s apartment from <span className={styles.italic}>The Substance (2024)</span>
          </h2>
          <div className={styles.carouselContentFlex}>
            <div className={styles.carouselContentDivOne}>
              <h3>DreamBooth Training Images</h3>
              <img 
                src={subTrain1} 
                alt="The Substance training 1" 
                className={styles.trainingImage}
              />
              <img 
                src={subTrain2} 
                alt="The Substance training 2" 
                className={styles.trainingImage}
              />
              <img 
                src={subTrain3} 
                alt="The Substance training 3" 
                className={styles.trainingImage}
              />
            </div>
            <div className={styles.carouselContentDivTwo}>
              <h3>Generated walkthrough video:</h3>
              <video 
                src={substanceVideo}
                controls
                muted
                loop
                style={{ width: '100%', maxWidth: '500px', height: 'auto' }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
      </Carousel>
        </>
      ) : (
        renderResultsPage()
      )}
    </div>
  );
}

export default App;
