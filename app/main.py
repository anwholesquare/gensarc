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

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sarcastic Reply Generator API"}

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