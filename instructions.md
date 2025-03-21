Here are some instructions/guidelines for the hackathon project. Please read them carefully and follow them step by step. The goal is to create a financial statement analysis tool using Google Gemini 2.0 Flash and LangChain, with a focus on modular design, prompt engineering, and efficient data processing.
# Hackathon Project: AI-Driven Financial Statement Analysis

Okay, let's break down this hackathon project step-by-step. You've got a good grasp of the core challenges, and I'm here to guide you through the technical and strategic aspects.  You're right to focus on Gemini 2.0 Flash and Thinking, given their capabilities and the generous free tier – that's a smart, cost-effective choice for a hackathon.  Let's tackle this systematically:

**1. High-Level Strategy & Architecture**

*   **LLM-Centric Approach:** You're absolutely correct – for this project, leveraging powerful LLMs like Gemini is the most efficient and effective approach.  Forget about complex, fine-tuned NER models, custom spaCy pipelines, or Hugging Face Transformers.  Gemini's large context window, native PDF/image understanding, and general reasoning abilities make it ideal for this task.  We'll focus on *prompt engineering*, not model training.

*   **Modular Design:**  We will break the problem into smaller, manageable chunks.  This means separate prompts for different aspects of the analysis (extraction, ratio calculation, summary generation). This is crucial for:
    *   **Controllability:**  Easier to debug and refine individual parts.
    *   **Output Quality:**  LLMs perform better with focused tasks.
    *   **Maintainability:**  If requirements change, you can modify specific modules.

*   **Data Flow:**
    1.  **Input:** User uploads a PDF, spreadsheet (CSV), or scanned document (image).
    2.  **Preprocessing (Minimal):**  For PDFs, use Gemini's built-in PDF parsing.  For CSV, ensure it's properly formatted. For scanned documents, Gemini should handle it directly due to its image understanding capabilities.
    3.  **LLM Processing (Multiple Stages):** A series of prompts to Gemini will:
        *   Extract key financial data (numbers, tables, text).
        *   Calculate financial ratios.
        *   Generate summaries for each section (Business Overview, Income Statement, etc.).
        *   Identify key trends and insights.
    4.  **Output:**
        *   Structured JSON data containing all extracted information and analysis results.
        *   A final PDF report generated from this JSON data.
    5. **Frontend(Streamlit):**
        *   File upload component for each input option, PDF, Spreadsheet, and Scanned Documents.
        *   Button to initiate the analysis.
        *   Displays and organizes the analyzed financial data.

*   **Tool Use (Important, but Focused):**  Yes, we *will* use tool use, but in a very specific way.  We're not building a general-purpose agent that can browse the web or use arbitrary tools.  Instead, we'll define custom tools that represent the *financial calculations*.  This is where LangChain comes in handy.  It allows us to:

    *   **Structure Calculations:**  Define functions for each ratio (current ratio, debt-to-equity, etc.).  These functions will take the necessary extracted data as input and return the calculated ratio.
    *   **Integrate with LLM:**  LangChain provides the framework to present these calculations as "tools" to Gemini.  The LLM can then decide *when* and *how* to use these tools based on the context of the financial document.  This is *far* more reliable than asking the LLM to perform calculations directly in its prompt.

*   **LangChain vs. LangGraph vs. CrewAI:**

    *   **LangChain:** This is our primary library. We'll use it for:
        *   **Prompt Templates:**  Creating structured, reusable prompts for Gemini.
        *   **Tool Definition:**  Defining the financial calculation tools.
        *   **LLM Chaining:**  Connecting the different prompts and tools in a logical sequence.
    *   **LangGraph:**  Potentially useful for more complex workflows, but likely overkill for this hackathon. LangGraph excels at creating cyclical or stateful agents, which we don't need here.  Let's keep it simple and stick with LangChain's core chaining capabilities.
    *   **CrewAI:**  Also overkill for this project.  CrewAI is designed for coordinating multiple agents with different roles.  We have a single, focused task.

**2. Detailed Implementation Plan**

Let's break down the implementation into specific, actionable steps:

**Step 1: Project Setup & Dependencies**

*   **Create a Python environment:** Use `venv` or `conda` to manage dependencies.
*   **Install Libraries:**
    ```bash
    pip install google-generativeai langchain streamlit PyPDF2 python-dotenv httpx
    ```
    *   `google-generativeai`:  For interacting with the Gemini API.
    *   `langchain`: For prompt engineering, tool definition, and LLM chaining.
    *   `streamlit`: For the initial frontend.
    *   `PyPDF2`: While Gemini handles PDFs natively, PyPDF2 might be useful for *pre-splitting* very long PDFs if you encounter context window limitations (unlikely, but good to have).
    *   `python-dotenv`: To manage your Gemini API key securely (store it in a `.env` file).
    *    `httpx`: For making HTTP requests.

**Step 2:  API Key Setup**

*   Obtain a Gemini API key from Google AI Studio.
*   Create a `.env` file in your project root:
    ```
    GOOGLE_API_KEY=your_actual_api_key_here
    ```
*   Load the API key in your Python code:
    ```python
    import os
    from dotenv import load_dotenv

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    ```

**Step 3: Core Functions (Tools)**

*   Create a Python file (e.g., `financial_tools.py`) to define your financial calculation tools.  These are standard Python functions:

    ```python
    # financial_tools.py

    def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
        """Calculates the current ratio."""
        if current_liabilities == 0:
            return float('inf')  # Handle division by zero
        return current_assets / current_liabilities

    def calculate_debt_to_equity_ratio(total_liabilities: float, shareholders_equity: float) -> float:
        """Calculates the debt-to-equity ratio."""
        if shareholders_equity == 0:
            return float('inf')
        return total_liabilities / shareholders_equity

    # ... Add functions for ALL the required ratios from the document ...
    def calculate_gross_margin_ratio(gross_profit: float, net_sales: float) -> float:
        if net_sales == 0:
            return 0.0
        return gross_profit / net_sales

    def calculate_operating_margin_ratio(operating_income: float, net_sales: float) -> float:
        if net_sales == 0:
            return 0.0
        return operating_income / net_sales

    def calculate_return_on_assets_ratio(net_income: float, total_assets: float) -> float:
        if total_assets == 0:
            return 0.0
        return net_income / total_assets

    def calculate_return_on_equity_ratio(net_income: float, shareholders_equity: float) -> float:
        if shareholders_equity == 0:
            return 0.0
        return net_income / shareholders_equity

    def calculate_asset_turnover_ratio(net_sales: float, average_total_assets: float) -> float:
        if average_total_assets == 0:
            return 0.0
        return net_sales / average_total_assets

    def calculate_inventory_turnover_ratio(cost_of_goods_sold: float, average_inventory: float) -> float:
        if average_inventory == 0:
            return 0.0
        return cost_of_goods_sold / average_inventory

    def calculate_receivables_turnover_ratio(net_credit_sales: float, average_accounts_receivable: float) -> float:
        if average_accounts_receivable == 0:
            return 0.0
        return net_credit_sales / average_accounts_receivable

    def calculate_debt_ratio(total_liabilities: float, total_assets: float) -> float:
        if total_assets == 0:
            return 0.0
        return total_liabilities / total_assets

    def calculate_interest_coverage_ratio(operating_income: float, interest_expenses: float) -> float:
        if interest_expenses == 0:
            return float('inf')
        return operating_income / interest_expenses
    ```
    *   **Type Hinting:** Use type hints (e.g., `current_assets: float`) for clarity and to help catch errors.
    *   **Error Handling:** Include basic error handling, especially for division by zero.
    *   **Docstrings:**  Write clear docstrings explaining what each function does.  This is crucial for the LLM to understand the tool's purpose.

**Step 4: LangChain Tool Integration**

*   Create another Python file (e.g., `llm_chain.py`). This is where you'll define your LangChain prompts and integrate the tools.

    ```python
    # llm_chain.py
    import os
    from dotenv import load_dotenv
    import google.generativeai as genai
    from langchain.prompts import ChatPromptTemplate
    from langchain.tools import StructuredTool
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from financial_tools import *  # Import your financial tools

    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # --- Define LangChain Tools from your functions ---
    tools = [
        StructuredTool.from_function(calculate_current_ratio),
        StructuredTool.from_function(calculate_debt_to_equity_ratio),
        # ... add all your other tools here ...
        StructuredTool.from_function(calculate_gross_margin_ratio),
        StructuredTool.from_function(calculate_operating_margin_ratio),
        StructuredTool.from_function(calculate_return_on_assets_ratio),
        StructuredTool.from_function(calculate_return_on_equity_ratio),
        StructuredTool.from_function(calculate_asset_turnover_ratio),
        StructuredTool.from_function(calculate_inventory_turnover_ratio),
        StructuredTool.from_function(calculate_receivables_turnover_ratio),
        StructuredTool.from_function(calculate_debt_ratio),
        StructuredTool.from_function(calculate_interest_coverage_ratio),

    ]

    # --- Initialize the LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking", google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True) #or gemini-2.0-flash
    llm_with_tools = llm.bind_tools(tools)

    # --- Prompt Templates ---
    # 1. Data Extraction Prompt
    extraction_template = """
    You are a financial analyst tasked with extracting key data from financial statements.
    Please extract the following information from the provided document and output it in JSON format:

    ```json
    {{
        "company_name": "",
        "reporting_period": "",
        "currency": "",
        "income_statement": {{
            "net_sales": ,
            "cost_of_goods_sold": ,
            "gross_profit": ,
            "operating_expenses": ,
            "operating_income": ,
            "interest_expenses": ,
            "net_income": 
        }},
        "balance_sheet": {{
            "current_assets": ,
            "total_assets": ,
            "current_liabilities": ,
            "total_liabilities": ,
            "shareholders_equity": ,
            "average_inventory": ,
            "average_accounts_receivable":
        }},
        "notes": {{
             "adj_ebitda_available": true/false,
             "adj_ebitda_details": "...",
             "adj_working_capital_available": true/false,
             "adj_working_capital_details": "..."
        }}
    }}
    ```
    If any information is not available, leave the value blank of that particular key or write 0.
    If any key has units, then also add it beside the numeric value.
    """
    extraction_prompt = ChatPromptTemplate.from_template(extraction_template)

    # 2. Ratio Calculation Prompt (using tools)
    ratio_template = """
    You are a financial analysis expert. Given the following extracted financial data,
    calculate the required financial ratios using the available tools.

    Extracted Data:
    {extracted_data}

    Calculate ALL of the following ratios and return the results and the input values used for those ratio calculations in JSON format:
    - Current Ratio
    - Debt to Equity Ratio
    - Gross Margin Ratio
    - Operating Margin Ratio
    - Return on Assets Ratio
    - Return on Equity Ratio
    - Asset Turnover Ratio
    - Inventory Turnover Ratio
    - Receivables Turnover Ratio
    - Debt Ratio
    - Interest Coverage Ratio

     ```json
    {{
        "Current Ratio": {{ "current_assets": , "current_liabilities": , "ratio_value":  }},
        "Debt to Equity Ratio": {{ "total_liabilities": , "shareholders_equity": , "ratio_value":  }},
        "Gross Margin Ratio": {{ "gross_profit": , "net_sales": , "ratio_value":  }},
        "Operating Margin Ratio": {{ "operating_income": , "net_sales": , "ratio_value":  }},
        "Return on Assets Ratio": {{ "net_income": , "total_assets": , "ratio_value":  }},
        "Return on Equity Ratio": {{ "net_income": , "shareholders_equity": , "ratio_value":  }},
        "Asset Turnover Ratio": {{ "net_sales": , "average_total_assets": , "ratio_value":  }},
        "Inventory Turnover Ratio": {{ "cost_of_goods_sold": , "average_inventory": , "ratio_value":  }},
        "Receivables Turnover Ratio": {{ "net_credit_sales": , "average_accounts_receivable": , "ratio_value":  }},
        "Debt Ratio": {{ "total_liabilities": , "total_assets": , "ratio_value":  }},
        "Interest Coverage Ratio": {{"operating_income": , "interest_expenses":, "ratio_value": }}
    }}
    ```
    """
    ratio_prompt = ChatPromptTemplate.from_template(ratio_template)


   # 3.  Summary Generation Prompts (Separate prompts for each section)

    # 3.a Business Overview
    business_overview_template = """
    Based on the extracted financial data, provide a concise business overview:
    {extracted_data}

    Output just the business overview text, and no extra text.
    """
    business_overview_prompt = ChatPromptTemplate.from_template(business_overview_template)

    # 3.b Key Findings
    key_findings_template = """
    Analyze the extracted data and calculated ratios, and provide key findings related to
    financial due diligence.  Focus on profitability, liquidity, solvency, key risks, and
    any notable trends.

    Extracted Data:
    {extracted_data}

    Calculated Ratios:
    {calculated_ratios}

    Output just the key findings text, and no extra text.
    """
    key_findings_prompt = ChatPromptTemplate.from_template(key_findings_template)

    # 3.c Income Statement Overview
    income_statement_template = """
    Provide a summary of the income statement, highlighting key trends and performance indicators.

    Extracted Data (Income Statement):
    {income_statement_data}

    Output just the income statement overview text, and no extra text.
    """
    income_statement_prompt = ChatPromptTemplate.from_template(income_statement_template)

    #3.d Balance Sheet Overview
    balance_sheet_template = """
    Provide a summary of the balance sheet, highlighting key trends and performance indicators.

    Extracted Data (Balance Sheet):
    {balance_sheet_data}

    Output just the balance sheet overview text, and no extra text.
    """
    balance_sheet_prompt = ChatPromptTemplate.from_template(balance_sheet_template)
    
    #3.e Adj EBITDA Overview
    adj_ebitda_template = """
    Analyze the details regarding adjusted EBITDA:
    {adj_ebitda_details}

    Output just the Adjusted EBITDA Overview text, and no extra text.
    """
    adj_ebitda_prompt = ChatPromptTemplate.from_template(adj_ebitda_template)

    #3.f Adj Working Capital Overview
    adj_working_capital_template = """
        Analyze the details regarding adjusted Working Capital:
        {adj_working_capital_details}
    
        Output just the Adjusted Working Capital Overview text, and no extra text.
        """
    adj_working_capital_prompt = ChatPromptTemplate.from_template(adj_working_capital_template)

    # --- Chain Definition ---

    # 1. Data Extraction Chain
    extraction_chain = extraction_prompt | llm_with_tools | JsonOutputParser()

    # 2. Ratio Calculation Chain
    ratio_chain = ratio_prompt | llm_with_tools | JsonOutputParser()

    # 3. Summary Chains
    business_overview_chain = business_overview_prompt | llm | StrOutputParser()
    key_findings_chain = key_findings_prompt | llm | StrOutputParser()
    income_statement_chain = income_statement_prompt | llm | StrOutputParser()
    balance_sheet_chain = balance_sheet_prompt | llm | StrOutputParser()
    adj_ebitda_chain = adj_ebitda_prompt | llm | StrOutputParser()
    adj_working_capital_chain = adj_working_capital_prompt | llm | StrOutputParser()



    # --- Main Function for Processing ---
    def process_financial_document(file_content, file_type):
        """Processes a financial document and returns the analysis results."""

        # Create a LangChain "Part" object for the document.
        part = genai.Part(data=file_content, mime_type=file_type)
        # 1. Extract Data
        extracted_data = extraction_chain.invoke({"input": [part]})
        #print(extracted_data)

        #2. Check for adj_ebitda and adj_working_capital
        adj_ebitda_available = extracted_data.get("notes", {}).get("adj_ebitda_available", False)
        adj_working_capital_available = extracted_data.get("notes", {}).get("adj_working_capital_available", False)
        adj_ebitda_details = extracted_data.get("notes",{}).get("adj_ebitda_details","")
        adj_working_capital_details = extracted_data.get("notes",{}).get("adj_working_capital_details","")
        # 3. Calculate Ratios
        calculated_ratios = ratio_chain.invoke({"extracted_data": extracted_data})
        #print(calculated_ratios)

        # 4. Generate Summaries

        business_overview = business_overview_chain.invoke({"extracted_data": extracted_data})
        key_findings = key_findings_chain.invoke({"extracted_data": extracted_data, "calculated_ratios": calculated_ratios})
        income_statement_overview = income_statement_chain.invoke({"income_statement_data": extracted_data["income_statement"]})
        balance_sheet_overview = balance_sheet_chain.invoke({"balance_sheet_data": extracted_data["balance_sheet"]})
        if adj_ebitda_available:
            adj_ebitda_overview = adj_ebitda_chain.invoke({"adj_ebitda_details":adj_ebitda_details})
        else:
            adj_ebitda_overview = "Adjusted EBITDA information not available in the provided document."

        if adj_working_capital_available:
            adj_working_capital_overview = adj_working_capital_chain.invoke({"adj_working_capital_details":adj_working_capital_details})
        else:
            adj_working_capital_overview = "Adjusted Working Capital information not available in the provided document."


        # 5. Combine Results into a single JSON object
        final_results = {
            "business_overview": business_overview,
            "key_findings": key_findings,
            "income_statement_overview": income_statement_overview,
            "balance_sheet_overview": balance_sheet_overview,
            "adj_ebitda_overview": adj_ebitda_overview,
            "adj_working_capital_overview": adj_working_capital_overview,
            "extracted_data": extracted_data,
            "calculated_ratios": calculated_ratios
        }

        return final_results
    ```

Key improvements and explanations in this code:

*   **Tool Integration:**  The `StructuredTool.from_function()` call wraps each of your calculation functions, making them available to the LLM.
*   **LLM Binding:**  `llm.bind_tools(tools)` creates a new LLM instance that is *aware* of your tools.  This is crucial.  The LLM will now include tool calls in its output when appropriate.
*   **Prompt Templates (Multiple):** We've defined separate, well-structured prompts for:
    *   **Data Extraction:**  Asks the LLM to extract specific data points and format them as JSON.  This is the most critical prompt.
    *   **Ratio Calculation:**  Provides the extracted data to the LLM and instructs it to use the available *tools* to calculate the ratios.  This prompt receives the *output* of the extraction prompt.
    *   **Summary Generation (x5):** Separate prompts for each section of the final report.  These prompts are designed to generate concise, human-readable text.
*   **Chain Definition:**  We use LangChain's simple chaining syntax (`|`) to connect the prompts and the LLM.  The output of one step becomes the input of the next.
*   **Output Parsers:**
    *    `JsonOutputParser()`: Used for extraction and ratio chains, as their output from the LLM is expected to be in JSON format.
    *   `StrOutputParser()`: Used for summary chains to get the raw text output.
*   **`process_financial_document` Function:** This is the main function that orchestrates the entire process:
    1.  Takes the file content and type as input.
    2.  Calls the `extraction_chain` to get the initial data.
    3.  Calls the `ratio_chain` to calculate ratios, passing in the extracted data.
    4.  Calls the summary chains to generate the report sections.
    5.  Combines *everything* into a single, comprehensive JSON object.
*   **JSON Output:** The entire analysis result is returned as a single JSON object.  This is *essential* for:
    *   **Frontend Integration:**  Streamlit (and later, Next.js) can easily consume this JSON to display the results in a structured way.
    *   **Report Generation:**  You'll use this JSON to generate the final PDF report.
* **Adj EBITDA and Working Capital:** Added handling of the `adj_ebitda_available` and `adj_working_capital_available` from the notes section of the extracted data to conditionally generate overviews for Adj EBITDA and Adj Working Capital.

**Step 5: Streamlit Frontend**

```python
# app.py
import streamlit as st
from llm_chain import process_financial_document  # Import your processing function
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO
import base64
import pathlib
import httpx


def generate_pdf_report(analysis_results):
    """Generates a PDF report from the analysis results."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(Paragraph("Financial Statement Analysis Report", styles['h1']))
    elements.append(Spacer(1, 12))

    # Add Business Overview
    elements.append(Paragraph("Business Overview", styles['h2']))
    elements.append(Paragraph(analysis_results['business_overview'], styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add Key Findings
    elements.append(Paragraph("Key Findings, Financial Due Diligence", styles['h2']))
    elements.append(Paragraph(analysis_results['key_findings'], styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Add Income Statement Overview
    elements.append(Paragraph("Income Statement Overview", styles['h2']))
    elements.append(Paragraph(analysis_results['income_statement_overview'], styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Add Balance Sheet Overview
    elements.append(Paragraph("Balance Sheet Overview", styles['h2']))
    elements.append(Paragraph(analysis_results['balance_sheet_overview'], styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add Adj EBITDA Overview
    elements.append(Paragraph("Adjusted EBITDA Overview", styles['h2']))
    elements.append(Paragraph(analysis_results['adj_ebitda_overview'], styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add Adj Working Capital Overview
    elements.append(Paragraph("Adjusted Working Capital Overview", styles['h2']))
    elements.append(Paragraph(analysis_results['adj_working_capital_overview'], styles['Normal']))
    elements.append(Spacer(1, 12))

    # Add Extracted Data (as a table)
    elements.append(Paragraph("Extracted Data", styles['h2']))
    data = [["Metric", "Value"]]
    for key, value in analysis_results['extracted_data'].items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                data.append([f"{key}.{sub_key}", str(sub_value)])
        else:
            data.append([key, str(value)])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Add Calculated Ratios (as a table)
    elements.append(Paragraph("Calculated Ratios", styles['h2']))
    ratio_data = [["Ratio", "Value"]]
    for ratio_name, values in analysis_results['calculated_ratios'].items():
         ratio_data.append([ratio_name, str(values["ratio_value"])])
    ratio_table = Table(ratio_data)
    ratio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(ratio_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

def get_pdf_download_link(pdf_bytes, filename):
    """Generates a download link for the PDF."""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href

st.title("AI-Driven Financial Statement Analysis")

uploaded_file = st.file_uploader("Upload a financial statement (PDF, CSV, or Image)", type=["pdf", "csv", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        # Already in bytes, no need to read again
        pass
    elif file_type == "text/csv":
         file_bytes = file_bytes #already in bytes
    elif file_type.startswith("image"):
         file_bytes = file_bytes #already in bytes
    else:
        st.error("Unsupported file type.")
        st.stop()

    with st.spinner("Analyzing the document..."):
        analysis_results = process_financial_document(file_bytes, file_type)

    st.subheader("Analysis Results:")
    st.write(analysis_results)  # Display the raw JSON for now

    # Generate and display PDF report
    with st.spinner("Generating PDF report..."):
        pdf_report = generate_pdf_report(analysis_results)
        st.markdown(get_pdf_download_link(pdf_report.read(), "financial_analysis_report.pdf"), unsafe_allow_html=True)

```

Key features of the Streamlit app:

*   **File Uploader:**  Allows the user to upload a file (PDF, CSV, or image).
*   **File Type Handling:** Correctly determines the file type and reads the content as bytes.
*   **Processing Call:**  Calls your `process_financial_document` function with the file content and type.
*   **Result Display:**  Displays the JSON output from the analysis. This is good for debugging and showing the raw data.  You can improve the formatting later.
*   **PDF Generation:** Uses the `generate_pdf_report` function (defined below) to create a PDF.
* **ReportLab PDF Generation** Included a basic report generation function.

**Step 6: PDF Report Generation**

*   The `generate_pdf_report` function uses the `reportlab` library to create a basic PDF report from the JSON data.  This is a *very simple* example.  You'll need to customize this significantly to match the desired output format from the hackathon problem statement:

    *   **Structure:**  Add sections for Business Overview, Key Findings, Income Statement Overview, Balance Sheet Overview, Adj EBITDA, and Adj Working Capital.
    *   **Formatting:** Use appropriate headings, fonts, and spacing.
    *   **Tables:**  Present the extracted data and calculated ratios in well-formatted tables. ReportLab provides excellent table styling options.
    *   **Error Handling:**  Add error handling in case certain data is missing from the JSON.

**Step 7: Testing and Refinement**

*   **Test with Sample Data:**  Use the provided sample input document to test the entire pipeline thoroughly.
*   **Iterate on Prompts:**  Refine your prompts based on the results.  Prompt engineering is an iterative process.  Pay close attention to:
    *   **Clarity:**  Make sure the instructions are unambiguous.
    *   **Specificity:**  Be precise about what you want the LLM to extract and how to format the output.
    *   **Tool Usage:**  Observe how the LLM uses the tools.  If it's not using them correctly, adjust the prompts or tool descriptions.
*   **Error Handling:**  Test with various edge cases (missing data, unusual formatting) to ensure your code is robust.

**3. Requirements Document (for LLM and Team)**


Project: AI-Driven Financial Statement Analysis

Goal: Develop a platform that automates the extraction, analysis, and interpretation of financial data from various financial documents, generating a comprehensive summary report.

Input: Financial statements in PDF, CSV, and scanned image formats (JPG, JPEG, PNG).  The example input document is a digitized image-based PDF of an annual report, containing a balance sheet, income statement, cash flow statement, and notes.

Output:  A structured JSON object containing all extracted data, calculated ratios, and generated summaries.  A PDF report generated from this JSON, mirroring the structure and content of the provided sample output document.

Core Requirements (from Problem Statement):

1.  Automated Data Extraction:
    *   Extract financial data from balance sheets, income statements, cash flow statements, and accompanying notes.
    *   Support PDF, spreadsheet (CSV), and scanned document (image) formats.

2.  NLP Capabilities:
    *   Read and extract relevant financial data from the supported document formats.
    *   Understand and interpret accounting terms and policies.
    *   Generate summary reports with key findings.

3.  AI-Powered Insights:
    *   Calculate key financial ratios (profitability, liquidity, solvency):
        *   Liquidity Ratios: Current ratio, Cash ratio
        *   Leverage Ratios: Debt to equity ratio, Debt ratio, Interest coverage ratio
        *   Efficiency Ratios: Asset turnover ratio, Inventory turnover ratio, Receivables turnover ratio
        *   Profitability Ratios: Gross margin ratio, Operating margin ratio, Return on assets ratio, Return on equity ratio
    *   Perform trend analysis across multiple financial periods (if data is available).

4.  Customization & Integration (Future - not for initial Streamlit version):
    *   API integration with financial databases and accounting software (not for this hackathon).
    *   Configurable report templates (basic implementation for this hackathon).

5. Output Sections
    *   Business Overview: Brief company description.
    *   Key Findings, Financial Due Diligence: Important insights from the analysis.
    *   Income Statement Overview: Summary of the income statement.
    *   Balance Sheet Overview: Summary of the balance sheet.
    *   Adj EBITDA: Analysis of Adjusted EBITDA (if data is available).
    *   Adj Working Capital: Analysis of Adjusted Working Capital (if data is available).

Technical Implementation:

*   Programming Language: Python
*   LLM: Google Gemini 2.0 Flash and Gemini 2.0 Flash Thinking (for reasoning)
*   Libraries:
```python
    *   google-generativeai: For interacting with the Gemini API.
    *   langchain: For prompt engineering, tool definition, and LLM chaining.
    *   streamlit: For the initial frontend.
    *   PyPDF2: For potential PDF pre-processing (if needed).
    *   python-dotenv: For secure API key management.
    *   reportlab: For PDF report generation.
    *   httpx: For making HTTP requests.
```
*   API Key: Securely stored and accessed using environment variables.

Architecture:

1.  **Input:** User uploads a document (PDF, CSV, or Image).
2.  **Preprocessing:** Minimal preprocessing, relying on Gemini's native capabilities.
3.  **LLM Processing (Modular):**
    *   **Data Extraction:**  Use a LangChain prompt to instruct Gemini to extract key financial data into a structured JSON format.
    *   **Ratio Calculation:**  Define Python functions for each financial ratio.  Use LangChain to expose these functions as "tools" to Gemini.  A separate prompt instructs Gemini to *use these tools* to calculate the ratios based on the extracted data.
    *   **Summary Generation:**  Use separate LangChain prompts for each section of the output report (Business Overview, Key Findings, etc.).  These prompts use the extracted data and calculated ratios to generate the summaries.
4.  **Output:**
    *   A comprehensive JSON object containing all extracted data, calculated ratios, and generated text.
    *   A PDF report generated from this JSON using ReportLab.
5. **Frontend (Streamlit):**
        *   File upload component for each input option, PDF, Spreadsheet, and Scanned Documents.
        *   Button to initiate the analysis.
        *   Displays and organizes the analyzed financial data.
        *   Button to download the output PDF.
Development Steps:

1.  **Project Setup:**
    *   Create a Python virtual environment.
    *   Install the required libraries.
    *   Set up the Gemini API key securely using `python-dotenv`.

2.  **Define Financial Calculation Tools:**
    *   Create a Python file (`financial_tools.py`) containing functions for each required financial ratio.
    *   Ensure each function has a clear docstring and handles potential errors (e.g., division by zero).

3.  **LangChain Integration (`llm_chain.py`):**
    *   Create LangChain `StructuredTool` objects from your calculation functions.
    *   Initialize the Gemini LLM and bind the tools to it.
    *   Create LangChain `ChatPromptTemplate` objects for:
        *   Data Extraction (outputting JSON).
        *   Ratio Calculation (using tools and outputting JSON).
        *   Summary Generation (separate prompts for each report section, outputting text).
    *   Define LangChain chains to connect the prompts and LLM in the correct sequence.
    *   Create a main function (`process_financial_document`) to handle the entire workflow:
        *   Takes file content and type as input.
        *   Executes the LangChain chains.
        *   Combines all results into a single JSON object.

4.  **Streamlit Frontend (`app.py`):**
    *   Create a Streamlit app with:
        *   A file uploader for PDF, CSV, and image files.
        *   File type handling.
        *   A call to the `process_financial_document` function.
        *   Display of the JSON output.
        *   PDF report generation and download link.

5.  **PDF Report Generation (in `app.py`):**
    *   Create a function (`generate_pdf_report`) that uses ReportLab to generate a PDF report from the JSON data.
    *   Implement the required report structure and formatting.

6.  **Testing and Refinement:**
    *   Thoroughly test the application with the provided sample input document and other test cases.
    *   Iteratively refine the LangChain prompts to improve accuracy and output quality.
    *   Add error handling for various scenarios.

Detailed Prompt Breakdown (Conceptual):

*   **Data Extraction Prompt:**
    *   Role: Financial analyst.
    *   Task: Extract specific data points from the financial document.
    *   Output Format: JSON (specified structure).
    *   Context: Provide the document content.
    *   Constraints: Handle missing data gracefully.

*   **Ratio Calculation Prompt:**
    *   Role: Financial analysis expert.
    *   Task: Calculate financial ratios using provided tools.
    *   Input: Extracted data (JSON from the previous step).
    *   Tools: List the available financial calculation tools.
    *   Output Format: JSON (specified structure with ratio values and inputs).

*   **Summary Generation Prompts (x5):**
    *   Role: Financial analyst.
    *   Task: Generate a concise summary for a specific section (Business Overview, Key Findings, etc.).
    *   Input: Relevant extracted data and/or calculated ratios.
    *   Output Format: Text.
    *   Constraints: Be concise and avoid unnecessary jargon.

Example JSON Output Structure (Conceptual):

```json
{
    "business_overview": "...",
    "key_findings": "...",
    "income_statement_overview": "...",
    "balance_sheet_overview": "...",
    "adj_ebitda_overview": "...",
    "adj_working_capital_overview": "...",
    "extracted_data": {
        "company_name": "...",
        "reporting_period": "...",
        "currency": "...",
        "income_statement": { ... },
        "balance_sheet": { ... },
        "notes": { ... }
    },
    "calculated_ratios": {
        "Current Ratio": { "current_assets": , "current_liabilities": , "ratio_value": },
        "Debt to Equity Ratio": { ... },
        ...
    }
}
```

This document provides a complete specification for the AI-Driven Financial Statement Analysis project.  It includes all the requirements, technical details, and step-by-step instructions needed for development.  It is designed to be understandable by both the development team and a new instance of an LLM.
```

This completes the requirements document, including the architecture, development steps, prompt breakdown, and example JSON structure. This should provide a solid foundation for you and your team to build the application and should be detailed enough to give to a new LLM instance for assistance. Remember to continually test and iterate, particularly on your prompts, as you develop. Good luck with the hackathon!
