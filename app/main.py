from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openai import OpenAI
from openai import AsyncOpenAI
from typing import List, Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import json
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

async def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI's text-embedding-3-small model"""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextItem(BaseModel):
    text: str

class QdrantUploadResponse(BaseModel):
    collection_name: str
    vectors_uploaded: int
    collection_info: dict

class InferenceRequest(BaseModel):
    message: str

class InferenceResponse(BaseModel):
    final_verdict: dict = {  # Combined prediction with confidence
        "is_sarcastic": bool,
        "confidence": float,
        "explanation": str
    }
    is_llm_predicted_sarcasm: int  # LLM's prediction
    is_embedded_predicted_sarcasm: bool  # Prediction based on embedding similarity
    reply: str
    similar_texts: List[dict] = []  # For debugging/transparency
    analysis: dict = {  # Include full analysis details
        "confidence": float,
        "reasoning": str,
        "cultural_context": str,
        "tone_analysis": str
    }

class SarcasmAnalysis(BaseModel):
    is_sarcastic: bool
    confidence: float
    reasoning: str
    cultural_context: str
    reply: str
    tone_analysis: str

class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int

async def generate_sarcastic_reply(text: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a sarcastic Bengali person. Generate a witty and sarcastic reply in Bengali to the given text. Keep the reply short and funny."},
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating reply for text: {text}")
        print(f"Error: {str(e)}")
        return "Error generating reply"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read the frontend/index.html file
    with open("frontend/index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.post("/generate-reply")
async def generate_reply(item: TextItem):
    try:
        reply = await generate_sarcastic_reply(item.text)
        return {"original_text": item.text, "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-csv")
async def process_csv(limit: Optional[int] = None, offset: Optional[int] = 0):
    try:
        # Read the CSV file
        df = pd.read_csv("bensarc.csv")
        
        # Create a new column for replies if it doesn't exist
        if 'reply' not in df.columns:
            df['reply'] = ''
        
        # Apply offset and limit
        start_idx = offset
        end_idx = len(df) if limit is None else min(offset + limit, len(df))
        
        # Process each row
        for index, row in df.iloc[start_idx:end_idx].iterrows():
            if pd.isna(df.at[index, 'reply']) or df.at[index, 'reply'] == '':
                print(f"working on {index} {row['Text']}")
                reply = await generate_sarcastic_reply(row['Text'])
                df.at[index, 'reply'] = reply
                print(f"reply: {reply}")
                # Save after each 10 rows to prevent data loss
                if index % 10 == 0:
                    df.to_csv("bensarc.csv", index=False)
                    print(f"saving till {index}")
                    time.sleep(1)  # Rate limiting
        
        # Save the final results
        df.to_csv("bensarc.csv", index=False)
        
        return {
            "message": "CSV processing completed",
            "processed_range": f"{start_idx} to {end_idx-1}",
            "total_rows": len(df),
            "processed_rows": end_idx - start_idx
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/regenerate-replies")
async def regenerate_replies(limit: Optional[int] = None, offset: Optional[int] = 0):
    try:
        # Read the CSV file
        df = pd.read_csv("bensarc.csv")
        
        # Create a new column for replies if it doesn't exist
        if 'reply' not in df.columns:
            df['reply'] = ''
        
        # Apply offset and limit
        start_idx = offset
        end_idx = len(df) if limit is None else min(offset + limit, len(df))
        
        # Process each row
        for index, row in df.iloc[start_idx:end_idx].iterrows():
            if not pd.isna(df.at[index, 'reply']) and df.at[index, 'reply'] != '':
                print(f"Regenerating reply for {index} {row['Text']}")
                print(f"Old reply: {df.at[index, 'reply']}")
                reply = await generate_sarcastic_reply(row['Text'])
                df.at[index, 'reply'] = reply
                print(f"New reply: {reply}")
                # Save after each 10 rows to prevent data loss
                if index % 10 == 0:
                    df.to_csv("bensarc.csv", index=False)
                    print(f"saving till {index}")
                    time.sleep(1)  # Rate limiting
        
        # Save the final results
        df.to_csv("bensarc.csv", index=False)
        
        return {
            "message": "Reply regeneration completed",
            "processed_range": f"{start_idx} to {end_idx-1}",
            "total_rows": len(df),
            "processed_rows": end_idx - start_idx
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/split-csv")
async def split_csv(chunk_size: int = 500):
    try:
        # Read the CSV file
        df = pd.read_csv("bensarc.csv")
        total_rows = len(df)
        
        # Calculate number of chunks
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        # Create chunks directory if it doesn't exist
        os.makedirs("chunks", exist_ok=True)
        
        chunk_files = []
        # Split into chunks and save
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            
            # Create chunk dataframe
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            # Generate filename
            filename = f"chunks/bensarc_chunk_{i+1}_of_{num_chunks}.csv"
            
            # Save chunk to CSV
            chunk_df.to_csv(filename, index=False)
            chunk_files.append(filename)
            
            print(f"Created chunk {i+1} of {num_chunks}: {filename}")
        
        return {
            "message": "CSV split completed",
            "total_rows": total_rows,
            "number_of_chunks": num_chunks,
            "chunk_size": chunk_size,
            "chunk_files": chunk_files
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_chunk(filename: str):
    try:
        print(f"Processing {filename}")
        # Read the chunk
        df = pd.read_csv(filename)
        
        # Create a new column for replies if it doesn't exist
        if 'reply' not in df.columns:
            df['reply'] = ''
        
        idx = 0
        # Process each row
        for index, row in df.iterrows():
            if row['Polarity'] == 1 and (pd.isna(df.at[index, 'reply']) or df.at[index, 'reply'] == ''):
                print(f"Working on {filename} - row {index}: {row['Text']}")
                reply = await generate_sarcastic_reply(row['Text'])
                df.at[index, 'reply'] = reply
                print(f"Reply: {reply}")
                idx = idx + 1
                # Save periodically
                if idx % 10 == 0:
                    df.to_csv(filename, index=False)
                    await asyncio.sleep(1)  # Rate limiting
        
        # Save final results
        df.to_csv(filename, index=False)
        return {"filename": filename, "status": "completed", "rows_processed": len(df)}
    
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return {"filename": filename, "status": "error", "error": str(e)}

@app.post("/process-chunks")
async def process_chunks(max_concurrent: int = 15):
    try:
        # Get all chunk files
        chunk_files = [f for f in os.listdir("chunks") if f.startswith("bensarc_chunk_") and f.endswith(".csv")]
        chunk_files = [os.path.join("chunks", f) for f in chunk_files]
        
        if not chunk_files:
            raise HTTPException(status_code=404, detail="No chunk files found")
        
        # Process chunks in batches
        results = []
        for i in range(0, len(chunk_files), max_concurrent):
            batch = chunk_files[i:i + max_concurrent]
            # Create tasks for the batch
            tasks = [process_chunk(filename) for filename in batch]
            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Wait a bit between batches to prevent rate limiting
            await asyncio.sleep(2)
        
        # Combine all processed chunks back into one file
        combined_df = pd.concat([pd.read_csv(f) for f in chunk_files])
        combined_df.to_csv("new_processed.csv", index=False)
        
        return {
            "message": "All chunks processed",
            "chunks_processed": len(results),
            "results": results,
            "combined_file": "new_processed.csv"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/merge-completed")
async def merge_completed():
    try:
        # Check if completed directory exists
        if not os.path.exists("completed"):
            raise HTTPException(status_code=404, detail="Completed directory not found")
        
        # Get all CSV files from completed directory
        completed_files = [f for f in os.listdir("completed") if f.startswith("bensarc_chunk_") and f.endswith(".csv")]
        completed_files = [os.path.join("completed", f) for f in completed_files]
        
        if not completed_files:
            raise HTTPException(status_code=404, detail="No completed chunk files found")
        
        print(f"Found {len(completed_files)} files to merge")
        
        # Read and combine all CSV files
        dfs = []
        for file in completed_files:
            try:
                print(f"Reading {file}")
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")
        
        if not dfs:
            raise HTTPException(status_code=500, detail="No valid CSV files could be read")
        
        # Combine all dataframes
        print("Combining dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by original index if it exists
        if 'id' in combined_df.columns:
            combined_df = combined_df.sort_values('id')
        
        # Save combined file
        output_file = "gen_sarc.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined file to {output_file}")
        
        return {
            "message": "Completed files merged successfully",
            "files_processed": len(completed_files),
            "total_rows": len(combined_df),
            "output_file": output_file
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-datasets")
async def create_datasets(test_size: float = 0.01, random_state: int = 42):
    try:
        # Read the generated sarcasm dataset
        if not os.path.exists("gen_sarc.csv"):
            raise HTTPException(status_code=404, detail="gen_sarc.csv not found")
        
        print("Reading gen_sarc.csv...")
        df = pd.read_csv("gen_sarc.csv")
        total_rows = len(df)
        
        # Calculate sizes
        test_rows = int(total_rows * test_size)
        train_rows = total_rows - test_rows
        
        print(f"Total rows: {total_rows}")
        print(f"Test rows: {test_rows}")
        print(f"Train rows: {train_rows}")
        
        # Create stratified sample based on Polarity
        test_indices = df.groupby('Polarity', group_keys=False).apply(
            lambda x: x.sample(n=int(len(x) * test_size), random_state=random_state)
        ).index
        
        # Split into test and train
        test_df = df.loc[test_indices]
        train_df = df.drop(test_indices)
        
        # Create output directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)
        
        # Save datasets
        test_file = "datasets/test_sarcasm.csv"
        train_file = "datasets/train_sarcasm.csv"
        
        test_df.to_csv(test_file, index=False)
        train_df.to_csv(train_file, index=False)
        
        # Calculate statistics
        test_stats = test_df['Polarity'].value_counts().to_dict()
        train_stats = train_df['Polarity'].value_counts().to_dict()
        
        return {
            "message": "Datasets created successfully",
            "total_rows": total_rows,
            "test_rows": len(test_df),
            "train_rows": len(train_df),
            "test_distribution": test_stats,
            "train_distribution": train_stats,
            "files": {
                "test": test_file,
                "train": train_file
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-create-dataset")
async def bulk_create_dataset(chunk_size: int = 500, max_concurrent: int = 15, test_size: float = 0.01, random_state: int = 123):
    try:
        results = []
        
        # Step 1: Split CSV
        print("Step 1: Splitting CSV...")
        split_result = await split_csv(chunk_size=chunk_size)
        results.append({"step": "split_csv", "result": split_result})
        print("CSV split completed")
        
        # Step 2: Process Chunks
        print("Step 2: Processing chunks...")
        process_result = await process_chunks(max_concurrent=max_concurrent)
        results.append({"step": "process_chunks", "result": process_result})
        print("Chunks processing completed")
        
        # Step 3: Merge Completed Files
        print("Step 3: Merging completed files...")
        merge_result = await merge_completed()
        results.append({"step": "merge_completed", "result": merge_result})
        print("Merging completed")
        
        # Step 4: Create Datasets
        print("Step 4: Creating train/test datasets...")
        dataset_result = await create_datasets(test_size=test_size, random_state=random_state)
        results.append({"step": "create_datasets", "result": dataset_result})
        print("Dataset creation completed")
        
        return {
            "message": "Bulk dataset creation completed successfully",
            "steps_completed": len(results),
            "results": results
        }
    
    except Exception as e:
        # Get the step where the error occurred
        current_step = len(results) + 1 if 'results' in locals() else 1
        error_message = f"Error in step {current_step}: {str(e)}"
        print(error_message)
        
        return {
            "message": "Bulk dataset creation failed",
            "error": error_message,
            "steps_completed": current_step - 1,
            "partial_results": results if 'results' in locals() else []
        }

@app.post("/upload-to-qdrant")
async def upload_to_qdrant(collection_name: str = "bengali_sarcasm"):
    try:
        # Read the training dataset
        if not os.path.exists("datasets/train_sarcasm.csv"):
            raise HTTPException(status_code=404, detail="Training dataset not found")
        
        print("Reading training dataset...")
        df = pd.read_csv("datasets/train_sarcasm.csv")
        
        # Create collection if it doesn't exist
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if not collection_exists:
            print(f"Creating collection {collection_name}...")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # text-embedding-3-small dimension
                    distance=models.Distance.COSINE
                )
            )
        
        # Generate embeddings and prepare points
        print("Generating embeddings...")
        texts = df['Text'].tolist()
        embeddings = []
        # Process in batches to respect rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i + batch_size, len(texts))]
            print(f"Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}...")
            batch_embeddings = await asyncio.gather(*[get_embedding(text) for text in batch])
            embeddings.extend(batch_embeddings)
            await asyncio.sleep(1)  # Rate limiting
        
        # Prepare points for upload
        points = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            point = models.PointStruct(
                id=idx,
                vector=embedding,  # No need for tolist() as it's already a list
                payload={
                    'text': df.at[idx, 'Text'],
                    'polarity': int(df.at[idx, 'Polarity']),
                    'reply': df.at[idx, 'reply']
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            print(f"Uploading batch {i//batch_size + 1}/{len(points)//batch_size + 1}...")
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
        
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        
        return QdrantUploadResponse(
            collection_name=collection_name,
            vectors_uploaded=len(points),
            collection_info=collection_info.dict()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_similar_texts(text: str, n_results: int = 10) -> List[dict]:
    """Get similar texts from Qdrant using embeddings"""
    text_embedding = await get_embedding(text)
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name="bengali_sarcasm_v1",
        query_vector=text_embedding,
        limit=n_results
    )
    
    # Format results
    similar_texts = [
        {
            "text": hit.payload["text"],
            "polarity": hit.payload["polarity"],
            "reply": hit.payload["reply"],
            "score": hit.score
        }
        for hit in search_results
    ]
    
    return similar_texts

@app.post("/evaluate-test")
async def evaluate_test():
    try:
        # Read test dataset
        if not os.path.exists("datasets/test_sarcasm.csv"):
            raise HTTPException(status_code=404, detail="Test dataset not found")
        
        df_test = pd.read_csv("datasets/test_sarcasm.csv")
        
        # Create directories if they don't exist
        os.makedirs("eval", exist_ok=True)
        
        # Initialize lists to store predictions and metrics
        predictions = []
        true_labels = []
        results = []
        
        # Process each test example
        for index, row in df_test.iterrows():
            print(f"Processing test example {index + 1}/{len(df_test)}")
            
            # Get inference result
            inference_request = InferenceRequest(message=row['Text'])
            inference_result = await infer_sarcasm(inference_request)
            
            # Store predictions
            predictions.append(inference_result.final_verdict["is_sarcastic"])
            true_labels.append(row['Polarity'])
            
            # Store detailed results
            results.append({
                "text": row['Text'],
                "true_polarity": row['Polarity'],
                "predicted_sarcastic": inference_result.final_verdict["is_sarcastic"],
                "confidence": inference_result.final_verdict["confidence"],
                "predicted_reply": inference_result.reply,
                "original_reply": row['reply'] if 'reply' in row else "",
                "llm_predicted": inference_result.is_llm_predicted_sarcasm,
                "embedding_predicted": inference_result.is_embedded_predicted_sarcasm,
                "explanation": inference_result.final_verdict["explanation"],
                "reasoning": inference_result.analysis["reasoning"]
            })
            
            # Add delay to respect rate limits
            await asyncio.sleep(1)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, predictions),
            "precision": precision_score(true_labels, predictions, zero_division=1),
            "recall": recall_score(true_labels, predictions, zero_division=1),
            "f1": f1_score(true_labels, predictions, zero_division=1)
        }
        
        # Print debug information
        print(f"True labels distribution: {pd.Series(true_labels).value_counts().to_dict()}")
        print(f"Predictions distribution: {pd.Series(predictions).value_counts().to_dict()}")
        
        # Add distribution information to metrics
        metrics.update({
            "true_positives": len([1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1]),
            "false_positives": len([1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1]),
            "true_negatives": len([1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0]),
            "false_negatives": len([1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0]),
            "total_samples": len(true_labels)
        })
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = f"eval/test_result_{timestamp}.csv"
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_file = f"eval/test_sarcasm_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Evaluate reply relevance using LLM
        system_prompt = """Evaluate if the predicted reply is relevant and appropriate for the given sarcastic text.
        Consider:
        1. Contextual relevance
        2. Sarcastic tone matching
        3. Cultural appropriateness
        4. Humor effectiveness
        
        Return a JSON with:
        {
            "is_relevant": true/false,
            "score": <float between 0 and 1>,
            "feedback": "<brief explanation>"
        }
        """
        
        reply_evaluations = []
        for result in results:
            if result["predicted_reply"]:  # Only evaluate if there's a reply
                eval_response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Text: {result['text']}\nReply: {result['predicted_reply']}"}
                    ],
                    response_format={ "type": "json_object" }
                )
                reply_eval = json.loads(eval_response.choices[0].message.content)
                reply_evaluations.append(reply_eval)
                await asyncio.sleep(1)  # Rate limiting
        
        # Calculate reply relevance metrics
        relevant_replies = sum(1 for eval in reply_evaluations if eval["is_relevant"])
        #avg_reply_score = sum(eval["score"] for eval in reply_evaluations) / len(reply_evaluations) if reply_evaluations else 0
        
        # Add reply metrics to results
        metrics["reply_relevance_rate"] = relevant_replies / len(reply_evaluations) if reply_evaluations else 0
        #metrics["avg_reply_score"] = avg_reply_score
        
        # Update metrics file with reply evaluation
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        
        return {
            "metrics": metrics,
            "files_saved": {
                "metrics": metrics_file,
                "results": results_file
            },
            "total_evaluated": len(df_test),
            "reply_metrics": {
                "relevant_replies": relevant_replies,
                "total_replies": len(reply_evaluations)
                #"avg_score": avg_reply_score
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def infer_sarcasm(request: InferenceRequest, n_embedding: int = 10):
    try:
        # Get similar texts for context
        similar_texts = await get_similar_texts(request.message, n_embedding)
        print(similar_texts)
        
        # Prepare examples for the prompt
        examples = "\n".join([
            f"Text: {item['text']}\n"
            f"Is Sarcastic: {'Yes' if item['polarity'] == 1 else 'No'}\n"
            f"Reply: {item['reply'] if item['polarity'] == 1 else 'N/A'}\n"
            f"Score: {item['score']:.3f}\n"
            for item in similar_texts[:5]  # Use top 5 for examples
        ])
        
        
        # Calculate weighted score considering both similarity and polarity
        weighted_scores = []
        for hit in similar_texts:
            # Weight the score by polarity (1 for sarcastic, -1 for non-sarcastic)
            polarity_weight = 1 if hit["polarity"] == 1 else -1
            weighted_score = hit["score"] * polarity_weight
            weighted_scores.append(weighted_score)
        
        # Calculate normalized weighted average (-1 to 1 range)
        avg_sarcastic_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        # Normalize to 0-1 range for easier interpretation
        normalized_score = (avg_sarcastic_score + 1) / 2
        
        # Determine predicted sarcasm based on similarity score
        is_embedded_predicted_sarcasm = normalized_score >= 0.4
        
        # Create a detailed prompt for sarcasm detection
        system_prompt = """You are an expert in Bengali sarcasm detection and generation, with deep understanding of Bengali culture, humor, and linguistic nuances.
        
        Analyze the given Bengali text for sarcasm and provide a structured response. Consider:
        
        1. Contextual Analysis:
           - Cultural references and their implications
           - Current events or social context
           - Intended vs literal meaning
        
        2. Linguistic Patterns:
           - Use of exaggeration or understatement
           - Ironic praise or mock politeness
           - Bengali-specific idioms and expressions
        
        3. Similar Examples:
        {examples}
        
        Semantic Similarity Analysis:
        - Weighted similarity score (considers both similarity and existing labels): {score:.3f}
        - This score ranges from 0 to 1, where:
          * Higher scores (>0.5) indicate similarity to sarcastic texts
          * Lower scores (<0.5) indicate similarity to non-sarcastic texts
        - Based on our empirical analysis:
          * Scores >= 0.4 strongly indicate sarcasm
          * Scores < 0.4 suggest non-sarcastic content
        - Current text's prediction based on similarity: {prediction}
        
        Important: While this semantic similarity should heavily influence your decision,
        also consider the linguistic patterns and cultural context. If you disagree with
        the similarity-based prediction, provide strong reasoning for your decision.
        
        Provide your analysis in the following JSON format:
        {{
            "is_sarcastic": true/false,
            "confidence": <float between 0 and 1>,
            "reasoning": "<detailed explanation including both semantic similarity and linguistic analysis>",
            "cultural_context": "<relevant Bengali cultural context>",
            "reply": "<if sarcastic, provide a witty Bengali reply that matches the tone>",
            "tone_analysis": "<analysis of tone, style, and how it aligns with similarity scores>"
        }}
        
        Make sure the reply (if sarcastic):
        - Uses appropriate Bengali cultural references
        - Maintains the subtle art of Bengali humor
        - Matches the original text's tone and style
        - Is contextually relevant and witty
        - Always be creative and make punch lines
        """
        
       
        
        # Get sarcasm detection and reply
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.format(
                    examples=examples, 
                    score=avg_sarcastic_score,
                    prediction="sarcastic" if is_embedded_predicted_sarcasm else "non-sarcastic"
                )},
                {"role": "user", "content": request.message}
            ],
            temperature=0.8,
            response_format={ "type": "json_object" }
        )
        
        print(response.choices[0].message.content)
        # Parse the response
        analysis = SarcasmAnalysis.model_validate_json(response.choices[0].message.content)
        
        # Calculate final verdict
        llm_confidence = analysis.confidence
        # Normalize embedding confidence from -1 to 1 range to 0 to 1 range
        embedding_confidence = normalized_score
        embedding_confidence = max(0, min(1, embedding_confidence))  # Clip to [0,1]
        
        # Weight the predictions (can be adjusted based on performance)
        llm_weight = 0.4
        embedding_weight = 0.6
        
        # Calculate weighted confidence for each prediction
        # Convert boolean predictions to confidence scores in [0,1]
        llm_score = llm_confidence if analysis.is_sarcastic else (1 - llm_confidence)
        embedding_score = embedding_confidence
        
        # Calculate weighted average of confidence scores
        llm_weighted = llm_weight * llm_score
        embedding_weighted = embedding_weight * embedding_score
        
        # Combine predictions
        combined_score = (llm_weighted + embedding_weighted)
        final_is_sarcastic = 1 if combined_score >= 0.45 else 0  # Convert to 0/1 integer
        
        # Calculate overall confidence  
        final_confidence = combined_score
        
        # Generate explanation
        final_explanation = (
            f"Final prediction combines LLM analysis (confidence: {llm_confidence:.2f}) and "
            f"embedding similarity (normalized confidence: {embedding_confidence:.2f}). "
            f"{'Both methods agree' if (1 if analysis.is_sarcastic else 0) == (1 if is_embedded_predicted_sarcasm else 0) else 'Methods disagree'} "
            f"on sarcasm detection. Final confidence: {final_confidence:.2f}"
        )
        
        return InferenceResponse(
            is_llm_predicted_sarcasm=1 if analysis.is_sarcastic else 0,
            is_embedded_predicted_sarcasm=1 if is_embedded_predicted_sarcasm else 0,
            reply=analysis.reply if analysis.is_sarcastic else "",
            similar_texts=similar_texts,
            final_verdict={
                "is_sarcastic": final_is_sarcastic,
                "confidence": final_confidence,
                "explanation": final_explanation
            },
            analysis={
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
                "cultural_context": analysis.cultural_context,
                "tone_analysis": analysis.tone_analysis
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calculate-metrics/{timestamp}")
async def calculate_metrics(timestamp: str):
    try:
        # Construct file path
        results_file = f"eval/test_sarcasm_{timestamp}.csv"
        
        # Check if file exists
        if not os.path.exists(results_file):
            raise HTTPException(status_code=404, detail=f"Test results file not found for timestamp {timestamp}")
        
        # Read results
        results_df = pd.read_csv(results_file)
        
        # Extract true labels and predictions
        true_labels = results_df['true_polarity'].tolist()
        predictions = [1 if p else 0 for p in results_df['predicted_sarcastic'].tolist()]
        
        # Calculate confusion matrix components
        true_positives = len([1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1])
        false_positives = len([1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1])
        true_negatives = len([1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0])
        false_negatives = len([1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0])
        total_samples = len(true_labels)
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / total_samples
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Print distributions for debugging
        print(f"True labels distribution: {pd.Series(true_labels).value_counts().to_dict()}")
        print(f"Predictions distribution: {pd.Series(predictions).value_counts().to_dict()}")
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives,
            total_samples=total_samples
        )
        
        return {
            "metrics": metrics.dict(),
            "distributions": {
                "true_labels": pd.Series(true_labels).value_counts().to_dict(),
                "predictions": pd.Series(predictions).value_counts().to_dict()
            },
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 