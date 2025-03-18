from typing import Annotated, Literal, Dict, Any
from typing_extensions import TypedDict
import os
import pandas as pd
from IPython.display import Image
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import MermaidDrawMethod

# Import custom tools
from tools.mini_tools import mean, stddev, sum_data
from tools.DataFrame import simple_df_summary, analyze_column
from tools.DataLoader import load_file
from tools.eda import basic_eda
from tools.visualization import VisualizationTools
from tools.DataCleaner import handle_missing_values, drop_duplicates, convert_data_types, handle_outliers
# Define State for the graph
class AnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    dataframes: Dict[str, pd.DataFrame]


def load_environment():
    """Load environment variables and return key configurations"""
    if load_dotenv():
        print("Environment variables loaded successfully")
    else:
        print("Failed to load environment variables")

    return {
        "openai_api_key": os.getenv("OPEN_AI_API_KEY")
    }


def to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Convert a CSV file to a pandas DataFrame

    Args:
        file_path: Path to the CSV file

    Returns:
        pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading DataFrame: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe instead of None


def save_graph_visualization(app, filename="statistics_agent_graph.png"):
    """
    Generate and save visualization of the graph

    Args:
        app: Compiled graph application
        filename: Output filename
    """
    output_path = f"diagrams/{filename}"
    os.makedirs("diagrams", exist_ok=True)

    try:
        # Primary method
        img = Image(app.get_graph().draw_mermaid_png())
        with open(output_path, "wb") as png:
            png.write(img.data)
        print(f"Graph visualization saved as '{output_path}'")
    except Exception as e:
        print(f" Error generating graph visualization: {str(e)}")

        # Fallback method
        try:
            mermaid_image = app.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API
            )

            fallback_path = f"diagrams/{filename.replace('.png', '_fallback.png')}"
            with open(fallback_path, "wb") as f:
                f.write(mermaid_image)
            print(f"Fallback graph visualization saved as '{fallback_path}'")
        except Exception as e2:
            print(f"Fallback visualization also failed: {str(e2)}")


def create_statistics_agent(api_key, sample_data_path='./data/rba-subset-int.csv'):
    """
    Create the statistics agent graph

    Args:
        api_key: OpenAI API key
        sample_data_path: Path to sample data

    Returns:
        tuple: (graph, compiled_app)
    """
    # Create the graph
    memory = MemorySaver()
    graph = StateGraph(AnalysisState)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000, api_key=api_key)

    # Set up visualization tools with sample data
    try:
        df = pd.read_csv(sample_data_path)
    except Exception:
        df = pd.DataFrame()  # Use empty dataframe if loading fails

    viz_tools = VisualizationTools(df, output_dir="output/plots")
    visualization_tools = viz_tools.get_tools()

    # Combine all tools
    tools = [
        mean, stddev, sum_data,
        simple_df_summary, analyze_column,
        load_file, basic_eda,
        handle_missing_values, drop_duplicates, convert_data_types, handle_outliers,
        *visualization_tools
    ]

    # Store tool objects in graph for later access
    graph.viz_tools = viz_tools

    # Set up LLM with tools and create tool node
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    graph.add_node("tool_node", tool_node)

    # Define prompt node
    def prompt_node(state: AnalysisState) -> AnalysisState:
        new_message = llm_with_tools.invoke(state["messages"])
        return {"messages": [new_message]}

    graph.add_node("prompt_node", prompt_node)

    # Define conditional edge logic
    def conditional_edge(state: AnalysisState) -> Literal['tool_node', '__end__']:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tool_node"
        else:
            return "__end__"

    # Connect the nodes
    graph.add_conditional_edges('prompt_node', conditional_edge)
    graph.add_edge("tool_node", "prompt_node")
    graph.set_entry_point("prompt_node")

    # Compile the graph
    app = graph.compile()

    return graph, app


def run_query(app, query):
    """
    Run a single query through the app and return the response

    Args:
        app: Compiled graph application
        query: User query string

    Returns:
        Response content
    """
    new_state = app.invoke({"messages": [query]}, {"recursion_limit": 100})
    return new_state["messages"][-1].content


def print_examples():
    """Print example queries for the user"""
    print("\nExample queries:")
    print("- 'What is the mean of [1, 2, 3, 4, 5]?'")
    print("- 'Calculate the standard deviation of [10, 20, 30, 40]'")
    print("- 'Find the sum of [5, 10, 15, 20]'")
    print("- 'Load the CSV file at ./data/rba-subset-int.csv'")
    print("- 'Provide a summary of the dataframe'")
    print("- 'Create a scatter plot of column X vs Y'\n")


def main():
    """Main function to run the app interactively"""
    print("Initializing Statistics Agent...")

    # Load environment and create agent
    env = load_environment()
    graph, app = create_statistics_agent(env["openai_api_key"])
    print("Agent ready!")

    # Generate graph visualization
    save_graph_visualization(app)

    # Load sample dataframe for demonstration
    file_path = './data/rba-subset-int.csv'
    df = to_dataframe(file_path)
    if not df.empty:
        print(f"\nSuccessfully loaded DataFrame from {file_path} with shape: {df.shape}")
        print(df.head(5))

    # Print example queries
    print_examples()

    # Interactive query loop
    while True:
        user_input = input("\nAsk a statistics question (or 'quit' to exit): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        try:
            response = run_query(app, user_input)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()