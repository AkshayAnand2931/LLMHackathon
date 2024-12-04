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

    # Retrieve context for the query
    retriever = create_retriever_tool(
        retriever, "retrieve_context", "Search for relevant information"
    )
    docs = retriever.invoke(user_query)

    # If retrieval finds results, use them
    if docs:
        retrieved_context = "\n".join(doc.page_content for doc in docs)
    else:
        retrieved_context = None

    # LLM reasoning and generation
    model = ChatOpenAI(
        model_name="gpt-4-0125-preview",
        temperature=0.5,
        streaming=True
    )

    # Prompt for recommendations
    prompt = PromptTemplate(
        template="""
        You are a recommendation assistant. Given the user's query and any retrieved context, generate a relevant recommendation.

        User Query: {user_query}

        Retrieved Context:
        {retrieved_context}

        If context is available, use it to make your recommendation. If no context is retrieved, generate a thoughtful response based on general knowledge. Provide reasoning in your answer.
        """,
        input_variables=["user_query", "retrieved_context"],
    )

    # Chain LLM with prompt
    recommendation_chain = prompt | model
    response = recommendation_chain.invoke({
        "user_query": user_query,
        "retrieved_context": retrieved_context or "No specific information retrieved."
    })

    return {"messages": [{"content": response}]}
