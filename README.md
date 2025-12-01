# Neumes Agent

AI-powered tool for identifying medieval music notation (neumes) in images.

## Features

- **Image Upload**: Upload images containing neume notation
- **Region Selection**: Click and drag to select specific regions containing neumes
- **AI Analysis**: Uses OpenAI's vision API to describe the neume shape
- **Embedding-based Matching**: Matches descriptions against a neumes lexicon using embeddings
- **Disambiguation**: When multiple neumes match, the user can select the best fit

## Architecture

```
┌─────────────────┐       ┌─────────────────┐
│    Frontend     │◄─────►│     Backend     │
│   React + TS    │       │    Node.js      │
│   (Vite)        │       │    (Express)    │
└─────────────────┘       └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │   OpenAI API    │
                          │ - Vision (GPT-4o)│
                          │ - Embeddings    │
                          └─────────────────┘
```

## Prerequisites

- Node.js 18+
- OpenAI API key

## Setup

### Backend

```bash
cd backend
npm install
cp .env.example .env
# Edit .env and add your OpenAI API key
npm run dev
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at http://localhost:5173 and the backend at http://localhost:3001.

## Environment Variables

### Backend (.env)

- `OPENAI_API_KEY`: Your OpenAI API key
- `PORT`: Server port (default: 3001)

### Frontend (.env)

- `VITE_API_URL`: Backend API URL (default: http://localhost:3001)

## How It Works

1. **Image Upload**: User uploads an image containing neume notation
2. **Region Selection**: User selects a region containing a specific neume
3. **AI Analysis**: The selected region is sent to the backend, which:
   - Uses OpenAI's vision API (GPT-4o) to describe the neume shape
   - Generates embeddings for the description
   - Compares with pre-computed embeddings of neume descriptions in the lexicon
   - Returns matching neumes ranked by similarity
4. **Disambiguation**: If multiple neumes match with similar scores, the user is asked to select the best fit

## Neumes Lexicon

The lexicon includes common neume types:

- **Punctum**: Single dot/square representing one note
- **Virga**: Single note with vertical stem
- **Pes (Podatus)**: Two ascending notes
- **Clivis**: Two descending notes
- **Scandicus**: Three or more ascending notes
- **Climacus**: Three or more descending notes
- **Torculus**: Low-high-low pattern
- **Porrectus**: High-low-high pattern
- **Quilisma**: Wavy/zigzag shaped neume
- And more...

## License

ISC