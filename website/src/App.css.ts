import { style } from "@vanilla-extract/css";

export const app = style({
  textAlign: "center",
  paddingTop: "2em",
});

export const nameContainer = style({
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  flexDirection: "row",
  "@media": {
    "(max-width: 800px)": {
      flexDirection: "column",
    },
  },
});

export const nameDiv = style({
  width: "50%",
});

export const contentStyle = style({
  textAlign: "center",

  //   height: "60vh",
  padding: "1em 1em 3em 1em",
});

export const abstract = style({
  textAlign: "center",
  //   height: "60vh",
  //   padding: "0em 1em 3em 1em",
  margin: "3em auto 0 auto",
  width: "50%",

  "@media": {
    "(max-width: 800px)": {
      width: "80%",
    },
  },
});

export const carousel = style({
  background: "#B4CAD1",
  margin: "5%",
  "@media": {
    "(max-width: 800px)": {
      margin: "5% 0",
    },
  },
});

export const carouselContentFlex = style({
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-around",
  alignItems: "center",
  padding: "1em",
  "@media": {
    "(max-width: 800px)": {
      flexDirection: "column",
    },
  },
});

export const carouselContentDivOne = style({
  textAlign: "center",
  width: "40%",
  "@media": {
    "(max-width: 800px)": {
      width: "90%",
    },
  },
});

export const carouselContentDivTwo = style({
  textAlign: "center",
  width: "55%",
  "@media": {
    "(max-width: 800px)": {
      width: "90%",
    },
  },
});
export const italic = style({
  fontStyle: "italic",
});

export const dreamboothImage = style({
  margin: "0 auto",
});
