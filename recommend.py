from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

@tool
def recommend(state):
    """
    Dynamically generates recommendations for any type of question.

    Args:
        state (dict): Contains the current state including messages.

    Returns:
        dict: The updated state with a recommendation response.
    """
    print("---GENERATE RECOMMENDATIONS---")

    messages = state["messages"]
    user_query = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    print(docs)
    print(user_query)
    # LLM reasoning and generation
    model = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        temperature=0,
        streaming=True
    )

    # Prompt for recommendations
    prompt = PromptTemplate(
        template="""
        You are a recommendation assistant. Given the user's query and any retrieved context, generate a relevant recommendation.

        User Query: {user_query}

        Retrieved Context:
        {docs}

        If context is available, use it to make your recommendation. If no context is retrieved, generate a thoughtful response based on general knowledge. Provide reasoning in your answer.
        """,
        input_variables=["user_query", "docs"],
    )

    # Chain LLM with prompt
    recommendation_chain = prompt | model | StrOutputParser()
    response = recommendation_chain.invoke({
        "user_query": user_query,
        "docs":docs
    })

    return {"messages": response}
