import json
from langchain.utilities import SerpAPIWrapper

# Add SerpAPI Wrapper
serp_api = SerpAPIWrapper(serpapi_api_key="ed09d22d123a41cab542c563919882c1460269e2b19f9f8803a27c7af888f324")  # Replace with your SerpAPI key

def web_search(question: str):
    """
    Perform a web search using SerpAPI and return the result as a JSON object.

    Args:
        question (str): The user query to search online.

    Returns:
        dict: JSON with "content" (all titles) and "source" (any one source).
    """
    print("---PERFORMING WEB SEARCH---")
    try:
        # Perform the search using SerpAPI
        search_results = serp_api.run(query=question)

        # Parse results into JSON
        if isinstance(search_results, list) and len(search_results) > 0:
            titles = " ".join(item.get("title", "") for item in search_results)
            source = search_results[0].get("source", "Unknown source")
            result_json = {"content": titles, "source": source}
        else:
            result_json = {"content": "No relevant information found online.", "source": "Unknown"}

    except Exception as e:
        result_json = {"content": f"An error occurred during the search: {e}", "source": "Unknown"}
    
    return result_json

if __name__ == "__main__":
    while True:
        print("--------------------------")
        user_prompt = input("Enter your search query: ")
        if user_prompt.lower() in ["exit", "quit"]:
            print("Exiting web search.")
            break
        result = web_search(user_prompt)
        # Save the result to a JSON file
        with open("search_result.json", "w") as json_file:
            json.dump(result, json_file, indent=4)
        print(f"Result stored in 'search_result.json':\n{result}")