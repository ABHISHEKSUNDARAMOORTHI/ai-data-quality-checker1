import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# It's good practice to choose a model suitable for text generation
# gemini-pro is generally a good choice for text-based tasks
model = genai.GenerativeModel('gemini-pro')

def generate_explanation(anomaly_data, column_stats=None, df_head=None):
    """
    Generates a human-readable explanation and remediation suggestion for a data quality anomaly.
    anomaly_data: dictionary containing anomaly details (type, column, description, sample_values)
    column_stats: optional dictionary of column statistics (min, max, mean, std, unique count)
    df_head: optional string representation of the first few rows of the DataFrame
    """
    prompt = f"""
    You are an expert Data Quality Analyst and Data Engineer. Your task is to explain data quality issues and suggest concrete remediation steps.

    Here is a detected data quality anomaly:
    - Anomaly Type: {anomaly_data.get('type', 'N/A')}
    - Column: {anomaly_data.get('column', 'N/A')}
    - Description: {anomaly_data.get('description', 'N/A')}
    - Sample Problematic Values: {anomaly_data.get('sample_values', 'N/A')}
    - Original Indices (rows): {anomaly_data.get('original_indices', 'N/A')}

    Additional Context (if available):
    """
    if column_stats:
        prompt += f"\n- Column Statistics: {column_stats}"
    if df_head:
        prompt += f"\n- First few rows of DataFrame:\n{df_head}"

    prompt += f"""

    Based on this information, provide:
    1.  A clear, concise explanation of the data quality issue.
    2.  Possible root causes for this issue.
    3.  Concrete, actionable remediation steps for a data engineer to fix this, including potential Python (Pandas) code snippets or SQL commands where applicable.
    4.  A short, specific title for this anomaly.

    Structure your response as follows:
    ---
    Title: [Your Title Here]
    Explanation: [Your detailed explanation]
    Possible Root Causes: [List of possible causes]
    Remediation Steps:
    [Numbered list of steps with code examples]
    ---
    """

    try:
        response = model.generate_content(prompt)
        # Check if parts exist and concatenate them, or handle a single text response
        if hasattr(response, 'parts') and response.parts:
            full_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            return full_text
        elif hasattr(response, 'text'):
            return response.text
        else:
            return "Could not generate explanation. Response format unexpected."
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"

# Example usage (for testing)
if __name__ == "__main__":
    test_anomaly = {
        "type": "Outlier (Numerical)",
        "column": "age",
        "description": "Potential outliers detected using Isolation Forest. Score below 5th percentile.",
        "sample_values": [300],
        "original_indices": [6]
    }
    test_column_stats = {
        "min": 22,
        "max": 300,
        "mean": 53.6,
        "std": 74.3
    }
    test_df_head = """
    id,name,age,city
    1,Alice,30,New York
    2,Bob,25,London
    ...
    7,Grace,300,New York
    """
    explanation = generate_explanation(test_anomaly, test_column_stats, test_df_head)
    print(explanation)