import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Load environment variables
# load_dotenv()

# Setup OpenAI and Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index_name = "rags-pdf-chatbot"
index = pc.Index(index_name)

# Function to create OpenAI embeddings
def create_embeddings(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def query_pinecone(query, top_k=10):  
    query_embedding = create_embeddings(query)
    query_response = index.query(vector=query_embedding, top_k=top_k, include_values=False, include_metadata=True)
    return query_response['matches']

# LLM-based re-ranking function
def llm_re_rank(query, results):
    re_ranked_results = []
    for match in results:
        document_text = match['metadata']['text']
        
        prompt = f"Query: {query}\nDocument: {document_text}\n\nRate the relevance of this document to the query from 1 to 10:"
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=5
        ).choices[0].text.strip()

        
        try:
            score = float(response)  
        except ValueError:
            score = 0  # If LLM response is not a valid number, assign a default score

        match['llm_score'] = score
        re_ranked_results.append(match)

    
    return sorted(re_ranked_results, key=lambda x: x['llm_score'], reverse=True)

def chat_with_enterprise_data(query, top_k=4):
    # Stage 1: Retrieve top_k results from Pinecone
    results = query_pinecone(query, top_k=10)  

    # Stage 2: Re-rank the results
    re_ranked_results = llm_re_rank(query, results)

    # Stage 3: Use the top re-ranked results for context
    top_results = re_ranked_results[:top_k]
    context = " ".join([match['metadata']['text'] for match in top_results if 'metadata' in match and 'text' in match['metadata']])

    # Stage 4: Generate a response 
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens=150
    ).choices[0].text.strip()

    return response, [match['metadata']['text'] for match in top_results]

# Define the test function
def test_answer_relevancy():
    # List of predefined test cases (questions)
    test_cases = [
        "What is naive bayes?",
        "Is naive bayes a linear classifier?",
        "Explain Bernoulli model?"
        # "How is feature selection done in naive bayes?",
        # "Explain logistic regression?"
    ]
    
    for user_input in test_cases:
        print(f"Running test case: {user_input}")
        response, retrieved_chunks = chat_with_enterprise_data(user_input, top_k=4)
        
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
        test_case = LLMTestCase(
            input=user_input,
            actual_output=response,
            retrieval_context=retrieved_chunks
        )
        assert_test(test_case, [answer_relevancy_metric])

if __name__ == "__main__":
    test_answer_relevancy()



