import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Turn LangSmith tracking ON
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"   # <-- paste your key
os.environ["LANGSMITH_PROJECT"] = "my-first-app"          # runs are grouped under this name
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"          # <-- paste your key

print("LangSmith is connected!")

def run_traced_chain():
    # Initialize the LangChain ChatOpenAI model
    # Note: Ensure OPENAI_API_KEY is set in your environment
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Define 3 distinct prompts    
    prompts = [
        "What is the capital of France?",
        "Count from 1 to 5 using words.",
        "Tell me a one-sentence joke about a computer."
    ]
    print("--- Starting LLM Runs ---\n")

    for i, prompt_text in enumerate(prompts, start=1):
        print(f"Sending Prompt #{i}: '{prompt_text}'")
        
        # Create a simple prompt template
        prompt_template = ChatPromptTemplate.from_template("{input}")
        
        # Combine into a runnable chain
        chain = prompt_template | llm 
        
        # Invoke the chain
        response = chain.invoke({"input": prompt_text})
        
        # Print the response content
        print(f"Answer #{i}:\n{response.content}\n")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    run_traced_chain()
