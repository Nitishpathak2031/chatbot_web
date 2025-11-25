import dotenv from "dotenv";
import axios from "axios";
import * as cheerio from "cheerio";
import { ChromaClient } from "chromadb";
import { GoogleGenerativeAI } from "@google/generative-ai";
import puppeteer from "puppeteer-core"; // Use puppeteer-core for external Chrome

dotenv.config();

// Gemini Init
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

// Chroma Init
const chromaClient = new ChromaClient({
  host: "localhost",
  port: 8000,
  ssl: false,
});
const heartbeat = await chromaClient.heartbeat();
console.log("✅ Chroma heartbeat:", heartbeat);

const WebCollections = "WEB_SCAPED_DATA_COLLECTION-1";

// ------------------- SCRAPER -------------------
async function scrapeWebPage(url = "") {
  try {
    console.log(`🕸 Scraping ${url} ...`);

    const browser = await puppeteer.launch({
      headless: true,
      executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", // Path change if needed
    });

    const page = await browser.newPage();
    await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });

    const body = await page.evaluate(() => {
      const removeTags = ["script", "style", "nav", "footer", "header"];
      removeTags.forEach((tag) =>
        document.querySelectorAll(tag).forEach((el) => el.remove())
      );
      return document.body.innerText;
    });

    const head = await page.evaluate(() => document.title);

    await browser.close();

    if (!body || body.trim().length < 50) {
      throw new Error("Scraped content is too short or empty.");
    }

    console.log("🟢 Scraping success");
    return {
      head,
      body: body.replace(/\s+/g, " ").trim(),
    };
  } catch (error) {
    console.error("🔴 Puppeteer scraping error:", error.message);
    throw error;
  }
}

// ------------------- EMBEDDING -------------------
async function generateVectorEmbeddings(text) {
  try {
    const embeddingModel = genAI.getGenerativeModel({
      model: "text-embedding-004",
    });

    const result = await embeddingModel.embedContent({
      content: { parts: [{ text: String(text) }] },
    });

    return result.embedding.values;
  } catch (error) {
    console.error("❌ Embedding error:", error.message);
    return [];
  }
}

// ------------------- INSERT INTO DB -------------------
async function insertIntoDB(id, embedding, url, body = "", head = "") {
  if (!embedding || embedding.length === 0) return;

  const collection = await chromaClient.getOrCreateCollection({
    name: WebCollections,
    embeddingFunction: null,
  });

  await collection.add({
    ids: [id],
    embeddings: [embedding],
    metadatas: [{ url, head, body }],
  });
}

// ------------------- TEXT CHUNKER -------------------
function chunkText(text, chunkSize = 500) {
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize)
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  return chunks;
}

// ------------------- INGEST -------------------
async function ingest(url = "") {
  console.log(`🧠 Ingesting: ${url}`);
  try {
    const { head, body } = await scrapeWebPage(url);
    const bodyChunks = chunkText(body, 500);

    for (let i = 0; i < bodyChunks.length; i++) {
      const chunkId = `${url}-chunk-${i}`;
      const bodyEmbedding = await generateVectorEmbeddings(bodyChunks[i]);
      await insertIntoDB(chunkId, bodyEmbedding, url, bodyChunks[i], head);
    }

    console.log(`✅ Ingested: ${url}`);
  } catch {
    console.error(`❌ Failed to ingest ${url}`);
  }
}

// ------------------- CHAT QUERY -------------------
async function chat(question = "") {
  console.log("💬 Query:", question);

  const questionEmbedding = await generateVectorEmbeddings(question);
  const collection = await chromaClient.getOrCreateCollection({
    name: WebCollections,
    embeddingFunction: null,
  });

  const results = await collection.query({
    nResults: 3,
    queryEmbeddings: [questionEmbedding],
  });

  const metadatas = results?.metadatas?.[0] || [];
  if (metadatas.length === 0) {
    console.log("🚫 No relevant context found.");
    return;
  }

  const bodies = metadatas.map((e) => e.body).filter(Boolean);
  const urls = [...new Set(metadatas.map((e) => e.url))];

  const prompt = `
You are an AI agent.
Based on the context, answer the user.

Query: ${question}
URL(s): ${urls.join(", ")}
Context:
---
${bodies.join("\n---\n")}
---
Answer:`;  

  const response = await model.generateContent(prompt);
  console.log("🤖:", response.response.text());
}

// ------------------- RUN -------------------
console.log("--- Starting Ingestion ---");
// await ingest("https://innovatex-technology.com");
// await ingest("https://innovatex-technology.com/about");
// await ingest("https://innovatex-technology.com/service");
// await ingest("https://innovatex-technology.com/contact"); 
// console.log("--- Ingestion Complete ---");

console.log("--- Starting Chat ---");
await chat("what are the contact detail provideed like gmail address mobile no");
