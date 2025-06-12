from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import base64
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load data and models
with open('discourse_posts.json', 'r') as f:
    discourse_posts = json.load(f)

with open('course_content.json', 'r') as f:
    course_content = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings
post_contents = [post['content'] for post in discourse_posts]
post_embeddings = model.encode(post_contents)

course_titles = [item['title'] for item in course_content]
course_texts = [item['text'] for item in course_content]
course_embeddings = model.encode(course_texts)

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

def find_similar_posts(question, top_k=3):
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, post_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(discourse_posts[i], similarities[i]) for i in top_indices]

def find_similar_course_content(question, top_k=2):
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, course_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(course_content[i], similarities[i]) for i in top_indices]

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    try:
        # Find relevant discourse posts
        similar_posts = find_similar_posts(request.question)
        post_links = [
            Link(
                url=post['url'],
                text=f"{post['username']}: {post['content'][:100]}..."
            )
            for post, score in similar_posts if score > 0.3
        ]
        
        # Find relevant course content
        similar_content = find_similar_course_content(request.question)
        content_links = [
            Link(
                url=content['url'],
                text=f"Course Content: {content['title']}"
            )
            for content, score in similar_content if score > 0.3
        ]
        
        # Combine all links
        all_links = post_links + content_links
        
        # Generate answer based on most relevant content
        if similar_posts and similar_posts[0][1] > 0.5:
            best_post = similar_posts[0][0]
            answer = f"Based on a similar question in the forum: {best_post['content'][:200]}..."
        elif similar_content and similar_content[0][1] > 0.5:
            best_content = similar_content[0][0]
            answer = f"According to the course materials: {best_content['text'][:200]}..."
        else:
            answer = "I couldn't find a specific answer to your question in the course materials or forum posts. Please provide more details or ask your question differently."
        
        return AnswerResponse(
            answer=answer,
            links=all_links[:3]  # Return top 3 links
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
