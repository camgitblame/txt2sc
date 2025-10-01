import { style } from "@vanilla-extract/css";

export const app = style({
  textAlign: "center",
  backgroundColor: "#282c34",
  color: "white",
  minHeight: "100vh",
  padding: "20px",
  fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif",
});

export const nameContainer = style({
  display: "flex",
  justifyContent: "center",
  margin: "30px 0",
});

export const nameDiv = style({
  textAlign: "center",
});

export const abstract = style({
  maxWidth: "800px",
  margin: "0 auto",
  textAlign: "left",
  padding: "20px",
  backgroundColor: "#1e2329",
  borderRadius: "8px",
  marginBottom: "50px",
});

export const carousel = style({
  maxWidth: "1200px",
  margin: "0 auto",
  backgroundColor: "#1e2329",
  borderRadius: "8px",
  padding: "20px",
});

export const contentStyle = style({
  padding: "40px",
  textAlign: "center",
  minHeight: "600px",
  color: "white",
});

export const carouselContentFlex = style({
  display: "flex",
  justifyContent: "center",
  alignItems: "flex-start",
  gap: "15px",
  marginTop: "20px",
  "@media": {
    "screen and (max-width: 768px)": {
      flexDirection: "column",
      gap: "15px",
    },
  },
});

export const carouselContentDivOne = style({
  flex: "1",
  display: "flex",
  flexDirection: "column",
  gap: "8px",
  color: "white",
  alignItems: "center",
});

export const carouselContentDivTwo = style({
  flex: "1",
  display: "flex",
  flexDirection: "column",
  gap: "8px",
  color: "white",
  alignItems: "center",
});

export const placeholderImage = style({
  width: "100%",
  height: "150px",
  backgroundColor: "#404040",
  border: "2px dashed #666",
  borderRadius: "8px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  color: "#ccc",
  fontSize: "14px",
});

export const placeholderVideo = style({
  width: "100%",
  height: "300px",
  backgroundColor: "#404040",
  border: "2px dashed #666",
  borderRadius: "8px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  color: "#ccc",
  fontSize: "18px",
});

export const italic = style({
  fontStyle: "italic",
});

export const trainingImage = style({
  width: "100%",
  maxWidth: "150px",
  aspectRatio: "1 / 1",
  objectFit: "cover",
  borderRadius: "8px",
  border: "2px solid #666",
  backgroundColor: "#2a2a2a",
});

export const dreamboothImage = style({
  margin: "0 auto",
});
