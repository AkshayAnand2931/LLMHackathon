from typing import Annotated, Sequence, Literal
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from retrieval_logic import PharmaceuticalRAG
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
import pprint

pharma_rag = PharmaceuticalRAG(
    data_path="datasets/microlabs_usa",
    lmstudio_api_base="http://localhost:1234/v1"
)

documents = pharma_rag.load_json_files()
retriever = pharma_rag.load_existing_vector_store()

class AgentState(TypedDict):

    messages: Annotated[Sequence[BaseMessage], add_messages]


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_pharmaceutical_data",
    "Search and return information about pharmaceutical medication"
)

tools = [retriever_tool]


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

    print("--CHECK RELEVANCE--")

    class grade(BaseModel):

        binary_score: str = Field(description="Relevance score 'yes' or 'no' ")

    model = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        temperature =0,
        streaming = True
    )

    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state['messages']
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("--DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("--DECISION: DOCS NOT RELEVANT--")
        return "rewrite"

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]

    model = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        temperature=0,
        streaming=True
    )

    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def create_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("agent",agent)
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    graph = workflow.compile()

    return graph


def final_response(user_query):
    inputs = {
        "messages": [
            ("user", user_query),
        ]
    }

    result = list()

    for output in graph.stream(inputs):
        result.append(output["agent"]["messages"][0].content)

    return result[0]


if __name__ == "__main__":

    graph = create_graph()

    while(True):

        print("--------------------------")
        user_prompt = input("Give question please...")
        inputs = {
            "messages": [
                ("user", user_prompt),
            ]
        }

        for output in graph.stream(inputs):
            print(output["agent"]["messages"][0].content)