import express from "express";
import cors from "cors";
import multer from "multer";
import dotenv from "dotenv";
import { analyzeAndMatchNeume } from "./neumeAnalysis.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 3001;

// Configure multer for memory storage
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Health check endpoint
app.get("/health", (_req, res) => {
  res.json({ status: "ok" });
});

// Analyze neume image endpoint
app.post("/api/analyze", upload.single("image"), async (req, res) => {
  try {
    let imageBase64: string;

    // Handle both file upload and base64 string
    if (req.file) {
      imageBase64 = req.file.buffer.toString("base64");
    } else if (req.body.image) {
      // Remove data URL prefix if present
      imageBase64 = req.body.image.replace(/^data:image\/\w+;base64,/, "");
    } else {
      res.status(400).json({ error: "No image provided" });
      return;
    }

    const result = await analyzeAndMatchNeume(imageBase64);
    res.json(result);

    // const result = await analyze(imageBase64);
    // res.send(result);
  } catch (error) {
    console.error("Analysis error:", error);
    res.status(500).json({
      error: "Failed to analyze image",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

app.listen(port, () => {
  console.log(`Neumes Agent backend running on port ${port}`);
});
