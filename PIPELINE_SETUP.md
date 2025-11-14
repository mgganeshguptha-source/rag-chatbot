# RAG Chatbot Embedding Pipeline Setup Guide

This guide explains how to set up the persistent embedding pipeline with Supabase PostgreSQL + pgvector.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Embedding Pipeline (Background)                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Google Drive │───▶│ embed_       │───▶│  Supabase    │  │
│  │  Documents   │    │ pipeline.py  │    │  PostgreSQL  │  │
│  └──────────────┘    └──────────────┘    │  + pgvector  │  │
│                             │             └──────────────┘  │
│                             ▼                                │
│                    sentence-transformers                    │
│                    (local, 768-dim, $0)                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Chat Application (Real-time)                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   User       │───▶│  Streamlit   │───▶│  Supabase    │  │
│  │   Query      │    │     App      │◀───│  PostgreSQL  │  │
│  └──────────────┘    └──────────────┘    │  (pgvector)  │  │
│                             │             └──────────────┘  │
│                             ▼                                │
│                       Gemini 2.0 Flash                      │
│                    (LLM only, FREE tier)                    │
└─────────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- ✅ **Zero cold start** - Embeddings pre-generated, instant app startup
- ✅ **$0/month cost** - Supabase free tier + local embeddings
- ✅ **Scalable** - Database handles millions of vectors efficiently
- ✅ **Incremental updates** - Only processes changed documents
- ✅ **Automated** - GitHub Actions runs pipeline automatically

---

## Step 1: Create Supabase Database

### 1.1 Sign Up for Supabase (FREE)
1. Go to [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Create account (free tier includes 500MB database)

### 1.2 Create New Project
1. Click "New Project"
2. Choose project name (e.g., "rag-chatbot-vectors")
3. Set database password (save this!)
4. Select region (choose closest to you)
5. Choose "Free" tier
6. Click "Create new project" (takes ~2 minutes)

### 1.3 Enable pgvector Extension
1. Once project is ready, go to **SQL Editor** (left sidebar)
2. Click "New Query"
3. Paste this command:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. Click **Run** (or press F5)
5. You should see: `Success. No rows returned`

### 1.4 Create Database Schema
1. Still in SQL Editor, click "New Query" again
2. Open the file `supabase_schema.sql` in your code editor
3. Copy **all contents** of `supabase_schema.sql`
4. Paste into Supabase SQL Editor
5. Click **Run**
6. Verify success: Go to **Table Editor**, you should see:
   - `document_metadata`
   - `document_embeddings`

---

## Step 2: Get Supabase Credentials

### 2.1 Database Connection URL
1. In Supabase dashboard, go to **Settings** → **Database**
2. Scroll to "Connection string"
3. Select **Session pooler** tab
4. Copy the **Connection string** (looks like):
   ```
   postgresql://postgres.xxx:[YOUR-PASSWORD]@aws-x-xx-xxx-x.pooler.supabase.com:6543/postgres
   ```
5. Replace `[YOUR-PASSWORD]` with your actual database password
6. Save this as `SUPABASE_DATABASE_URL`

### 2.2 Project URL
1. Go to **Settings** → **API**
2. Copy **Project URL** (e.g., `https://xxx.supabase.co`)
3. Save this as `SUPABASE_URL`

### 2.3 Service Role Key
1. Still in **Settings** → **API**
2. Under "Project API keys", find **service_role** key
3. Click "Reveal" and copy the long key
4. Save this as `SUPABASE_SERVICE_KEY`

⚠️ **Important:** The service_role key is secret! Never commit it to git.

---

## Step 3: Configure Secrets

### For Replit (Development)
1. Click **Tools** → **Secrets** in Replit sidebar
2. Add these secrets:
   - `SUPABASE_DATABASE_URL`: Your connection string
   - `SUPABASE_URL`: Your project URL
   - `SUPABASE_SERVICE_KEY`: Your service role key

### For Streamlit Cloud (Production)
1. Go to your app dashboard on [share.streamlit.io](https://share.streamlit.io)
2. Click **Settings** (gear icon) → **Secrets**
3. Add all Supabase credentials in TOML format:
   ```toml
   SUPABASE_DATABASE_URL = "postgresql://..."
   SUPABASE_URL = "https://xxx.supabase.co"
   SUPABASE_SERVICE_KEY = "eyJ..."
   GEMINI_API_KEY = "AIza..."
   GOOGLE_SERVICE_ACCOUNT_KEY = '''{"type":"service_account",...}'''
   GOOGLE_DRIVE_FOLDER_ID = "1BEf..."
   ```

### For GitHub Actions (Automated Pipeline)
1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each:
   - `SUPABASE_DATABASE_URL`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `GOOGLE_SERVICE_ACCOUNT_KEY`
   - `GOOGLE_DRIVE_FOLDER_ID`
   - `GEMINI_API_KEY` (optional, only if pipeline needs LLM)

---

## Step 4: Run Embedding Pipeline

### Option A: Manual Run (First Time)

```bash
# Install dependencies
pip install sentence-transformers psycopg2-binary google-api-python-client google-auth requests beautifulsoup4 pypdf pytesseract pillow

# Run pipeline (incremental update)
python embed_pipeline.py

# Or force rebuild all embeddings
python embed_pipeline.py --force-rebuild
```

Expected output:
```
==================================================
🚀 RAG Chatbot Embedding Pipeline
==================================================

📦 Initializing components...
✅ Connected to Supabase PostgreSQL database
🔄 Loading sentence-transformers model...
✅ Embedding model loaded (768-dim, local)
✅ Pipeline initialization complete

📥 Loading documents from Google Drive...
✅ Loaded 15 documents

🔄 Processing 15 changed/new documents:

📄 Processing: WhatsApp Chat with...
   ✅ Created 145 chunks
  🔄 Generating 145 embeddings locally...
  ✅ Generated 145 embeddings (768-dim, 0 API calls)
   ✅ Stored 145 embeddings in database

...

==================================================
📊 Pipeline Statistics:
   Total documents: 15
   Total chunks: 181
   Avg chunks/doc: 12.1
==================================================
✅ Pipeline execution complete!
```

### Option B: GitHub Actions (Automated)

1. Push your code to GitHub
2. Go to **Actions** tab
3. Select "Update RAG Embeddings" workflow
4. Click **Run workflow** button
5. Optionally check "Force rebuild all embeddings"
6. Click green **Run workflow** button
7. Wait ~5 minutes, check logs

The workflow will:
- ✅ Install Python dependencies
- ✅ Download sentence-transformers model
- ✅ Load documents from Google Drive
- ✅ Generate embeddings locally (no API cost!)
- ✅ Update PostgreSQL database

---

## Step 5: Verify Database

### Via Supabase Dashboard
1. Go to **Table Editor** → `document_embeddings`
2. You should see rows with:
   - chunk_id
   - document_id
   - content
   - embedding (vector)
   - metadata

### Via SQL Query
```sql
-- Check statistics
SELECT * FROM document_stats;

-- Test similarity search (replace with your actual embedding)
SELECT * FROM cosine_similarity_search(
    (SELECT embedding FROM document_embeddings LIMIT 1),
    0.5,
    10
);

-- Count total chunks
SELECT COUNT(*) FROM document_embeddings;
```

---

## Step 6: Update Streamlit App

The app will automatically use the persistent PostgreSQL database instead of in-memory Chroma.

**What changed:**
- ❌ No more cold-start embedding generation
- ❌ No more in-memory Chroma database
- ✅ Fast startup (~2 seconds instead of ~20 seconds)
- ✅ Persistent storage across restarts
- ✅ Semantic search powered by pgvector

---

## Troubleshooting

### "Extension vector does not exist"
- Make sure you ran: `CREATE EXTENSION IF NOT EXISTS vector;`
- Check that you're using PostgreSQL 16+ (Supabase uses this by default)

### "Permission denied for table document_embeddings"
- Verify you're using the `service_role` key, not the `anon` key
- Check row-level security policies in Supabase dashboard

### Pipeline fails with "sentence-transformers not found"
- Run: `pip install sentence-transformers`
- For Streamlit Cloud, add to `requirements.txt`

### "Failed to connect to PostgreSQL"
- Verify `SUPABASE_DATABASE_URL` is correct
- Make sure you replaced `[YOUR-PASSWORD]` with actual password
- Check that database is not paused (Supabase free tier pauses after inactivity)

### Embeddings not showing in chat
- Verify pipeline completed successfully
- Check `document_embeddings` table has rows
- Restart Streamlit app to reload configuration

---

## Monitoring & Maintenance

### Check Pipeline Runs (GitHub Actions)
- Go to **Actions** tab in GitHub
- View workflow run history
- Check logs for errors

### Monitor Database Usage (Supabase)
- Go to **Settings** → **Usage**
- Free tier limits:
  - 500 MB database size
  - 2 GB bandwidth/month
  - Unlimited API requests

### Update Schedule
The GitHub Action runs:
- **Daily at 2 AM UTC** (automatic)
- **On code changes** (when you push updates)
- **Manual trigger** (click "Run workflow" button)

You can modify the schedule in `.github/workflows/update-embeddings.yml`:
```yaml
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM UTC
  # - cron: '0 */6 * * *'  # Every 6 hours
  # - cron: '0 0 * * 0'  # Weekly on Sunday
```

---

## Cost Analysis

| Component | Service | Cost |
|-----------|---------|------|
| Vector Database | Supabase (500MB) | **$0/month** |
| Embeddings | sentence-transformers (local) | **$0/month** |
| LLM Generation | Gemini 2.0 Flash (free tier) | **$0/month** |
| Hosting | Streamlit Cloud | **$0/month** |
| Pipeline Execution | GitHub Actions (2000 min/month) | **$0/month** |
| **TOTAL** | | **$0/month** 🎉 |

**Scalability Notes:**
- Free tier supports ~500,000 chunks (500MB / ~1KB per chunk)
- For larger datasets, upgrade to Supabase Pro ($25/month for 8GB)
- sentence-transformers runs on GitHub Actions runners (no cost)

---

## Next Steps

1. ✅ Run the pipeline for the first time
2. ✅ Verify embeddings in Supabase dashboard
3. ✅ Test the Streamlit app (should start instantly)
4. ✅ Set up GitHub Actions for automation
5. 📚 Read architecture documentation in `replit.md`

---

## Support

- **Supabase Docs:** https://supabase.com/docs/guides/database/extensions/pgvector
- **sentence-transformers:** https://www.sbert.net/
- **GitHub Actions:** https://docs.github.com/en/actions

Happy building! 🚀
