import * as styles from "./App.css.ts";
import { Carousel } from "antd";
import "./App.css";
import { Github, ScrollText, Play, Database, X } from "lucide-react";
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
import shiningTrain4 from "./assets/the_shining/train4.png";
import shiningTrain5 from "./assets/the_shining/train5.png";

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

// Import baseline videos
import alienBaseline from "./assets/alien/video_baseline.mp4";
import asBaseline from "./assets/american_psycho/video_baseline.mp4";
import shiningBaseline from "./assets/the_shining/video_baseline.mp4";
import passengersBaseline from "./assets/passengers/video_baseline.mp4";
import substanceBaseline from "./assets/substance/video_baseline.mp4";

// Import DB+1CN videos
import alienDb1cn from "./assets/alien/video_db_1cn.mp4";
import asDb1cn from "./assets/american_psycho/video_db_1cn.mp4";
import shiningDb1cn from "./assets/the_shining/video_db_1cn.mp4";
import passengersDb1cn from "./assets/passengers/video_db_1cn.mp4";
import substanceDb1cn from "./assets/substance/video_db_1cn.mp4";

// Import DB+2CN videos
import alienDb2cn from "./assets/alien/video_db_2cn.mp4";
import asDb2cn from "./assets/american_psycho/video_db_2cn.mp4";
import shiningDb2cn from "./assets/the_shining/video_db_2cn.mp4";
import passengersDb2cn from "./assets/passengers/video_db_2cn.mp4";
import substanceDb2cn from "./assets/substance/video_db_2cn.mp4";

function App() {
  const [currentView, setCurrentView] = useState('main');
  const [zoomedImage, setZoomedImage] = useState(null);
  
  const onChange = (currentSlide) => {
    console.log(currentSlide);
  };

  const renderDataPage = () => {
    const movies = [
      { name: "The Shining", subtitle: "The Overlook Hotel hallway" },
      { name: "The Substance", subtitle: "Elisabeth Sparkle's apartment" },
      { name: "American Psycho", subtitle: "Patrick Bateman's apartment" },
      { name: "Passengers", subtitle: "The Vienna Suite" },
      { name: "Alien", subtitle: "The Nostromo corridor" }
    ];

    // Training images mapping - you'll need to import additional images
    const trainingImagesMap = {
      "The Shining": [shiningTrain1, shiningTrain2, shiningTrain3, shiningTrain4, shiningTrain5],
      "The Substance": [subTrain1, subTrain2, subTrain3],
      "American Psycho": [asTrain1, asTrain2, asTrain3],
      "Passengers": [pasTrain1, pasTrain2, pasTrain3],
      "Alien": [alienTrain1, alienTrain2, alienTrain3]
    };

    return (
      <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "20px" }}>
        <div className={styles.abstract} style={{ marginBottom: "40px" }}>
          <h2>Training Data</h2>
          <p style={{ marginBottom: "20px" }}>
            DreamBooth training images extracted from each movie to capture the distinctive visual style and characteristics of each scene. Click on any image to view full size.
          </p>
        </div>

        {movies.map((movie, movieIndex) => (
          <div key={movie.name} style={{ marginBottom: "60px" }}>
            <h2 style={{ textAlign: "center", marginBottom: "30px" }}>
              {movie.name} - <span className={styles.italic}>{movie.subtitle}</span>
            </h2>
            <div style={{ 
              display: "grid", 
              gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", 
              gap: "20px",
              maxWidth: "1200px",
              margin: "0 auto"
            }}>
              {trainingImagesMap[movie.name].map((image, imageIndex) => (
                <div 
                  key={imageIndex} 
                  style={{ 
                    textAlign: "center",
                    cursor: "pointer",
                    transition: "transform 0.2s ease"
                  }}
                  onClick={() => setZoomedImage(image)}
                  onMouseEnter={(e) => e.target.style.transform = "scale(1.05)"}
                  onMouseLeave={(e) => e.target.style.transform = "scale(1)"}
                >
                  <img
                    src={image}
                    alt={`${movie.name} training ${imageIndex + 1}`}
                    style={{
                      width: "100%",
                      aspectRatio: "1 / 1",
                      borderRadius: "8px",
                      objectFit: "cover",
                      border: "2px solid #666",
                      backgroundColor: "#2a2a2a",
                      transition: "transform 0.2s ease, box-shadow 0.2s ease"
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.boxShadow = "0 8px 25px rgba(255, 255, 255, 0.1)";
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.boxShadow = "none";
                    }}
                  />
                  <p style={{ 
                    marginTop: "10px", 
                    fontSize: "14px", 
                    color: "#ccc" 
                  }}>
                    Training Image {imageIndex + 1}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* Image Zoom Modal */}
        {zoomedImage && (
          <div 
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(0, 0, 0, 0.9)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
              cursor: "pointer"
            }}
            onClick={() => setZoomedImage(null)}
          >
            <button
              style={{
                position: "absolute",
                top: "20px",
                right: "20px",
                background: "rgba(255, 255, 255, 0.2)",
                border: "none",
                borderRadius: "50%",
                width: "50px",
                height: "50px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                color: "white",
                fontSize: "24px"
              }}
              onClick={(e) => {
                e.stopPropagation();
                setZoomedImage(null);
              }}
            >
              <X />
            </button>
            <img
              src={zoomedImage}
              alt="Zoomed training image"
              style={{
                maxWidth: "90%",
                maxHeight: "90%",
                objectFit: "contain",
                borderRadius: "8px"
              }}
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        )}
      </div>
    );
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

    // Video mapping
    const videoMap = {
      "The Shining": {
        "Baseline": shiningBaseline,
        "DB+1CN": shiningDb1cn,
        "DB+2CN": shiningDb2cn
      },
      "The Substance": {
        "Baseline": substanceBaseline,
        "DB+1CN": substanceDb1cn,
        "DB+2CN": substanceDb2cn
      },
      "American Psycho": {
        "Baseline": asBaseline,
        "DB+1CN": asDb1cn,
        "DB+2CN": asDb2cn
      },
      "Passengers": {
        "Baseline": passengersBaseline,
        "DB+1CN": passengersDb1cn,
        "DB+2CN": passengersDb2cn
      },
      "Alien": {
        "Baseline": alienBaseline,
        "DB+1CN": alienDb1cn,
        "DB+2CN": alienDb2cn
      }
    };

    return (
      <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "20px" }}>
        <div className={styles.abstract} style={{ marginBottom: "40px" }}>
          <h2>Results Comparison</h2>
          <p style={{ marginBottom: "20px" }}>
            Below are the generated walkthrough videos for all five movies using three different methods:
          </p>
          <ul style={{ textAlign: "left", margin: "0", padding: "0 20px" }}>
            <li><strong>Baseline</strong>: SceneScape</li>
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
                  <video
                    controls
                    muted
                    loop
                    style={{
                      width: "100%",
                      maxWidth: "300px",
                      aspectRatio: "1 / 1",
                      borderRadius: "8px",
                      objectFit: "contain"
                    }}
                  >
                    <source src={videoMap[movie.name][variant]} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
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
        <button
          onClick={() => setCurrentView(currentView === 'main' ? 'data' : 'main')}
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
          <Database /> {currentView === 'main' ? 'Data' : 'Back to Main'}
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
      ) : currentView === 'results' ? (
        renderResultsPage()
      ) : (
        renderDataPage()
      )}
    </div>
  );
}

export default App;
