# yt2report

Transform YouTube videos into structured reports through an AI pipeline.

## Overview
This tool takes YouTube videos (individual or channel) and processes them into well-structured, readable reports through several stages:

1. Audio extraction from YouTube
2. Raw transcript generation via AssemblyAI
3. Enhanced transcript creation using Google Gemini (prompt adapted from [Dwarkesh Patel](https://gist.github.com/dwarkeshsp/65c232298781c86b33f5e32065152f1e))
4. Final structured report generation via OpenRouter (for flexible model selection)

## Requirements 
- AssemblyAI API key (https://www.assemblyai.com/)
- Google API key (https://aistudio.google.com/)
- Openrouter API key (https://openrouter.ai/settings/keys)

## Usage
uv run process.py
