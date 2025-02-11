import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import OpenAI from "openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import "dotenv/config";

// Type definitions for better type safety
type SimilarityMetric = "dot_product" | "cosine" | "euclidean";

interface EnvironmentVariables {
  ASTRA_DB_API_ENDPOINT: string;
  ASTRA_DB_APPLICATION_TOKEN: string;
  ASTRA_DB_NAMESPACE: string;
  ASTRA_DB_COLLECTION: string;
  OPENAI_API_KEY: string;
}

// Validate environment variables
const validateEnvVariables = (): EnvironmentVariables => {
  const required = [
    "ASTRA_DB_API_ENDPOINT",
    "ASTRA_DB_APPLICATION_TOKEN",
    "ASTRA_DB_NAMESPACE",
    "ASTRA_DB_COLLECTION",
    "OPENAI_API_KEY",
  ];

  const missing = required.filter((key) => !process.env[key]);
  if (missing.length > 0) {
    throw new Error(
      `Missing required environment variables: ${missing.join(", ")}`
    );
  }

  return {
    ASTRA_DB_API_ENDPOINT: process.env.ASTRA_DB_API_ENDPOINT!,
    ASTRA_DB_APPLICATION_TOKEN: process.env.ASTRA_DB_APPLICATION_TOKEN!,
    ASTRA_DB_NAMESPACE: process.env.ASTRA_DB_NAMESPACE!,
    ASTRA_DB_COLLECTION: process.env.ASTRA_DB_COLLECTION!,
    OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
  };
};

class F1DataLoader {
  private client: DataAPIClient;
  private db: any;
  private openai: OpenAI;
  private splitter: RecursiveCharacterTextSplitter;
  private readonly f1Data: string[];

  constructor(env: EnvironmentVariables) {
    this.client = new DataAPIClient(env.ASTRA_DB_APPLICATION_TOKEN);
    this.db = this.client.db(env.ASTRA_DB_API_ENDPOINT, {
      namespace: env.ASTRA_DB_NAMESPACE,
    });
    this.openai = new OpenAI({ apiKey: env.OPENAI_API_KEY });
    this.splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 512,
      chunkOverlap: 100,
    });

    this.f1Data = [
      "https://en.wikipedia.org/wiki/Formula_One",
      "https://f1.fandom.com/wiki/Formula_1_Wiki",
      "https://www.formula1.com/en/results/driver-standings",
      "https://www.formula1.com/en/racing/2025",
      "https://www.skysports.com/f1",
      "https://www.bbc.com/sport/formula1",
      "https://www.espn.com/f1/",
      "https://www.motorsport.com/f1/",
      "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020",
      "https://www.statista.com/topics/3899/motor-sports/",
      "https://www.statsf1.com/en/default.aspx",
      "https://www.formula1.com/en/timing/f1-live",
      "https://en.wikipedia.org/wiki/List_of_Formula_One_World_Drivers%27_Champions",
      "https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship",
      "https://www.formula1.com/en/page/what-is-f1",
      "https://f1chronicle.com/a-beginners-guide-to-formula-1/",
      "https://www.cnet.com/culture/sports/f1-101-heres-everything-i-wish-i-knew-about-formula-1-when-i-started-watching/",
    ];
  }

  async createCollection(collectionName: string): Promise<void> {
    try {
      const similarityMetric: SimilarityMetric = "dot_product";
      await this.db.createCollection(collectionName, {
        vector: {
          dimension: 1536,
          metric: similarityMetric,
        },
      });
      console.log(`Collection ${collectionName} created successfully`);
    } catch (error) {
      if ((error as any)?.message?.includes("already exists")) {
        console.log(`Collection ${collectionName} already exists`);
      } else {
        throw error;
      }
    }
  }

  private async scrapePage(url: string): Promise<string> {
    try {
      console.log(`Starting to scrape ${url}`);

      const loader = new PuppeteerWebBaseLoader(url, {
        launchOptions: {
          headless: "new",
          args: [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-accelerated-2d-canvas",
            "--disable-gpu",
            "--window-size=1920x1080",
          ],
        },
        gotoOptions: {
          waitUntil: "networkidle0",
          timeout: 60000,
        },
        evaluate: async (page, browser) => {
          try {
            // Wait for some common selectors that might indicate content is loaded
            await Promise.race([
              page.waitForSelector("p", { timeout: 5000 }).catch(() => null),
              page
                .waitForSelector("article", { timeout: 5000 })
                .catch(() => null),
              page
                .waitForSelector(".content", { timeout: 5000 })
                .catch(() => null),
            ]);

            const result = await page.evaluate(() => {
              // Remove unwanted elements
              const elementsToRemove = document.querySelectorAll(
                "nav, footer, header, script, style, iframe, .advertisement, .ads, .cookie-notice"
              );
              elementsToRemove.forEach((el) => el.remove());

              // Try to get the main content first
              const mainContent = document.querySelector(
                "main, article, .content, #mw-content-text"
              );
              if (mainContent && mainContent instanceof HTMLElement) {
                return mainContent.textContent || "";
              }

              // Fallback to body content
              return document.body.textContent || "";
            });

            await browser.close();
            return result;
          } catch (error) {
            await browser.close();
            throw error;
          }
        },
      });

      const content = await loader.scrape();

      if (!content) {
        throw new Error("No content retrieved");
      }

      const cleanedContent = content
        .replace(/\s+/g, " ")
        .replace(/(\r\n|\n|\r)/gm, " ")
        .trim();

      if (cleanedContent.length < 100) {
        throw new Error("Retrieved content is too short to be valid");
      }

      console.log(
        `Successfully scraped ${url} - Content length: ${cleanedContent.length} characters`
      );
      return cleanedContent;
    } catch (error) {
      console.error(`Error scraping ${url}:`, error);
      throw error;
    }
  }

  private async createEmbedding(text: string): Promise<number[]> {
    try {
      const embedding = await this.openai.embeddings.create({
        input: text,
        model: "text-embedding-3-small",
        encoding_format: "float",
      });
      return embedding.data[0].embedding;
    } catch (error) {
      console.error("Error creating embedding:", error);
      throw error;
    }
  }

  async loadSampleData(collectionName: string): Promise<void> {
    const collection = await this.db.collection(collectionName);
    let processedUrls = 0;
    let totalChunks = 0;
    let failedUrls: string[] = [];

    for (const url of this.f1Data) {
      try {
        console.log(
          `\nProcessing ${url} (${processedUrls + 1}/${this.f1Data.length})...`
        );

        let content: string | undefined;
        // Add retry logic for scraping
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            content = await this.scrapePage(url);
            break;
          } catch (error) {
            if (attempt === 3) {
              throw error;
            }
            console.log(`Attempt ${attempt} failed, retrying...`);
            await new Promise((resolve) => setTimeout(resolve, 5000 * attempt));
          }
        }

        if (!content) {
          throw new Error("No content retrieved after retries");
        }

        const chunks = await this.splitter.splitText(content);
        console.log(`Split into ${chunks.length} chunks`);

        for (const chunk of chunks) {
          try {
            const vector = await this.createEmbedding(chunk);
            await collection.insertOne({
              $vector: vector,
              text: chunk,
              source_url: url,
              timestamp: new Date().toISOString(),
            });
            totalChunks++;
          } catch (error) {
            console.error(`Error processing chunk from ${url}:`, error);
            continue;
          }
        }
        processedUrls++;
        console.log(
          `Successfully processed ${url}. Total progress: ${processedUrls}/${this.f1Data.length} URLs, ${totalChunks} chunks inserted`
        );
      } catch (error) {
        console.error(`Failed to process ${url}:`, error);
        failedUrls.push(url);
        continue;
      }
    }

    console.log("\n=== Processing Summary ===");
    console.log(
      `Total URLs processed successfully: ${processedUrls}/${this.f1Data.length}`
    );
    console.log(`Total chunks inserted: ${totalChunks}`);
    if (failedUrls.length > 0) {
      console.log("\nFailed URLs:");
      failedUrls.forEach((url) => console.log(`- ${url}`));
    }
  }
}

// Main execution
async function main() {
  try {
    const env = validateEnvVariables();
    const loader = new F1DataLoader(env);

    await loader.createCollection(env.ASTRA_DB_COLLECTION);
    await loader.loadSampleData(env.ASTRA_DB_COLLECTION);
  } catch (error) {
    console.error("Error in main execution:", error);
    process.exit(1);
  }
}

main();
