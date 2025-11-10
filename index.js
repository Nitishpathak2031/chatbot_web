import Dotenv from "dotenv";
import axios from "axios";
import * as cheerio from "cheerio";
import { ChromaClient } from "chromadb";
import { GoogleGenerativeAI } from "@google/generative-ai";

Dotenv.config();

// ✅ Initialize Gemini client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// ✅ Initialize Chroma client
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
  const { data } = await axios.get(url);
  const $ = cheerio.load(data);

  const pageHead = $("head").text();
  const pageBody = $("body").text();

  const internalLinks = new Set();
  const externalLinks = new Set();

  $("a").each((_, el) => {
    const link = $(el).attr("href");
    if (!link || link === "/") return;

    if (link.startsWith("http") || link.startsWith("https")) {
      externalLinks.add(link);
    } else {
      internalLinks.add(link);
    }
  });

  return {
    head: pageHead,
    body: pageBody,
    internalLinks: Array.from(internalLinks),
    externalLinks: Array.from(externalLinks),
  };
}

// ------------------- EMBEDDING (using Gemini) -------------------
async function generateVectorEmbeddings(text) {
  try {
    const model = genAI.getGenerativeModel({ model: "text-embedding-004" });

    const result = await model.embedContent({
      content: {
        parts: [
          {
            text: typeof text === "string" ? text : JSON.stringify(text),
          },
        ],
      },
    });

    return result.embedding.values;
  } catch (error) {
    console.error("❌ Error generating embedding:", error.message);
    return [];
  }
}

// ------------------- INSERT INTO DB -------------------
async function insertIntoDB(embedding, url, body = "", head = "") {
  if (!embedding || embedding.length === 0) {
    console.warn("⚠️ Skipped empty embedding for:", url);
    return;
  }

  const collection = await chromaClient.getOrCreateCollection({
    name: WebCollections,
    embeddingFunction: null,
  });

  await collection.add({
    ids: [url],
    embeddings: [embedding],
    metadatas: [{ url, head, body }],
  });
}

// ------------------- TEXT CHUNKER -------------------
function chunkText(text, chunkSize) {
  if (!text || chunkSize <= 0) throw new Error("Invalid input text");
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

// ------------------- INGEST -------------------
async function ingest(url = "") {
  console.log(`🧠 Ingesting: ${url}`);
  const { head, body } = await scrapeWebPage(url);
  const bodyChunks = chunkText(body, 1000);

  for (const chunk of bodyChunks) {
    const bodyEmbedding = await generateVectorEmbeddings(chunk);
    await insertIntoDB(bodyEmbedding, url, chunk, head);
  }

  console.log(`✅ Ingested: ${url}`);
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

  const Bodyresponse = results.metadatas[0]
  .map((e) => e.body)
  .filter((e) => e.trim() !== '' && !!e);

  const urlresponse = results.metadatas[0]
  .map((e) => e.url)
  .filter((e) => e.trim() !== '' && !!e);

  console.log("📄 Top results:", urlresponse);
//   console.log("📄 Top results:", Bodyresponse);

  const response = await model.generateContent([
  {
    role: "system",
    parts: [
      {
        text:
          "You are an AI agent expert in providing support to users on behalf of a webpage. Given the page content, reply accordingly.",
      },
    ],
  },
  {
    role: "user",
    parts: [
      {
        text: `Query: ${question}\n\nURL: ${urlresponse.join(
          ", "
        )}\n\nRetrieved Context: ${Bodyresponse.join(", ")}`,
      },
    ],
  },
]);

console.log("🤖:", response.response.text());
}

// ------------------- RUN -------------------
// Uncomment this once to scrape and store data
// await ingest("https://www.piyushgarg.dev");
// await ingest("https://www.piyushgarg.dev/cohort");
// await ingest("https://www.piyushgarg.dev/about");

// Then run chat after ingestion
await chat("What is cohort?");
