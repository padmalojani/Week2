import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Set your API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
def main():
    # 1. Initialize the model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 2. Create a structured prompt template with placeholders ({count}, {diet_type})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional nutritionist."),
        ("user", "Provide exactly {count} {diet_type} breakfast ideas. Format them strictly as a numbered list, one line each, with no preamble.")
    ])

    # 3. Initialize the string output parser to automatically extract .content
    parser = StrOutputParser()

    # 4. Build the LCEL chain: Prompt -> Model -> Parser
    chain = prompt | llm | parser

    # 5. Invoke the chain by passing a dictionary of variables
    result = chain.invoke({
        "count": 5, 
        "diet_type": "healthy and high-protein"
    })

    print(result)

if __name__ == "__main__":
    main()