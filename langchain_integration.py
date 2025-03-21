from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import os
import json
from financial_tools import (
    calculate_current_ratio, calculate_debt_to_equity_ratio,
    calculate_gross_margin_ratio, calculate_operating_margin_ratio,
    calculate_return_on_assets_ratio, calculate_return_on_equity_ratio,
    calculate_asset_turnover_ratio, calculate_inventory_turnover_ratio,
    calculate_receivables_turnover_ratio, calculate_debt_ratio,
    calculate_interest_coverage_ratio
)

load_dotenv()

class LangChainHandler:
    """Handles multi-step LLM operations for financial analysis."""
    
    def __init__(self):
        # Use gemini-2.0-flash for data extraction and gemini-2.0-flash-thinking for reasoning
        self.extraction_llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        self.analysis_llm = GoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.2)
        
        # Define the output schemas and parsers
        self._setup_parsers()
        
        # Create the extraction chain
        self._setup_extraction_chain()
        
    def _setup_parsers(self):
        """Set up output parsers for structured data"""
        # Financial data extraction schema
        self.financial_data_schema = {
            "company_name": "Name of the company",
            "reporting_period": "Reporting period of the financial statement",
            "currency": "Currency used in the financial statement",
            "income_statement": {
                "net_sales": "Total net sales or revenue",
                "cost_of_goods_sold": "Cost of goods sold",
                "gross_profit": "Gross profit",
                "operating_expenses": "Operating expenses",
                "operating_income": "Operating income",
                "interest_expenses": "Interest expenses",
                "net_income": "Net income"
            },
            "balance_sheet": {
                "cash_and_equivalents": "Cash and cash equivalents",
                "current_assets": "Total current assets",
                "total_assets": "Total assets",
                "current_liabilities": "Total current liabilities",
                "total_liabilities": "Total liabilities",
                "shareholders_equity": "Total shareholders' equity",
                "average_inventory": "Average inventory for the period",
                "average_accounts_receivable": "Average accounts receivable for the period"
            },
            "notes": {
                "adj_ebitda_available": "Whether adjusted EBITDA information is available (true/false)",
                "adj_ebitda_details": "Details about adjusted EBITDA if available",
                "adj_working_capital_available": "Whether adjusted working capital information is available (true/false)",
                "adj_working_capital_details": "Details about adjusted working capital if available"
            }
        }
    
    def _setup_extraction_chain(self):
        """Set up the extraction chain for financial data"""
        extraction_template = """You are a financial analyst tasked with extracting key data from financial statements.
Please extract the following information from the provided document and output it in JSON format:

```json
{{
"company_name": "",
"reporting_period": "",
"currency": "",
"income_statement": {{
"net_sales": null,
"cost_of_goods_sold": null,
"gross_profit": null,
"operating_expenses": null,
"operating_income": null,
"interest_expenses": null,
"net_income": null
}},
"balance_sheet": {{
"cash_and_equivalents": null,
"current_assets": null,
"total_assets": null,
"current_liabilities": null,
"total_liabilities": null,
"shareholders_equity": null,
"average_inventory": null,
"average_accounts_receivable": null
}},
"notes": {{
"adj_ebitda_available": false,
"adj_ebitda_details": "",
"adj_working_capital_available": false,
"adj_working_capital_details": ""
}}
}}
```

If any information is not available, use null for that value.
If numbers have units (like thousands or millions), make sure to convert them to actual numbers and not include the units in the JSON values.
If you see values for multiple years, use the most recent year's data.
For average values (like average inventory), calculate them if provided with beginning and ending values, or use the most recent value if only one is available.

Document content:
{document_content}"""
        
        self.extraction_prompt = ChatPromptTemplate.from_template(extraction_template)
        self.extraction_chain = LLMChain(
            llm=self.extraction_llm, 
            prompt=self.extraction_prompt, 
            output_key="extracted_data",
            verbose=True
        )
        
    def calculate_financial_ratios(self, data: Dict) -> Dict:
        """Calculate financial ratios using the extracted data"""
        # Extract relevant financial data
        income_statement = data.get('income_statement', {})
        balance_sheet = data.get('balance_sheet', {})
        
        # Values needed for ratio calculations
        net_sales = income_statement.get('net_sales', 0)
        cost_of_goods_sold = income_statement.get('cost_of_goods_sold', 0)
        gross_profit = income_statement.get('gross_profit', 0)
        operating_income = income_statement.get('operating_income', 0)
        interest_expenses = income_statement.get('interest_expenses', 0)
        net_income = income_statement.get('net_income', 0)
        
        cash_and_equivalents = balance_sheet.get('cash_and_equivalents', 0)
        current_assets = balance_sheet.get('current_assets', 0)
        total_assets = balance_sheet.get('total_assets', 0)
        current_liabilities = balance_sheet.get('current_liabilities', 0)
        total_liabilities = balance_sheet.get('total_liabilities', 0)
        shareholders_equity = balance_sheet.get('shareholders_equity', 0)
        average_inventory = balance_sheet.get('average_inventory', 0)
        average_accounts_receivable = balance_sheet.get('average_accounts_receivable', 0)
        
        # Use net_sales for net_credit_sales if not provided
        net_credit_sales = net_sales
        
        # Calculate average_total_assets as total_assets if not provided
        average_total_assets = total_assets
        
        # Calculate all ratios using our financial_tools.py functions
        ratios = {
            "Current Ratio": {
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "ratio_value": calculate_current_ratio(current_assets, current_liabilities)
            },
            "Cash Ratio": {
                "cash_and_equivalents": cash_and_equivalents,
                "current_liabilities": current_liabilities,
                "ratio_value": calculate_current_ratio(cash_and_equivalents, current_liabilities)
            },
            "Debt to Equity Ratio": {
                "total_liabilities": total_liabilities,
                "shareholders_equity": shareholders_equity,
                "ratio_value": calculate_debt_to_equity_ratio(total_liabilities, shareholders_equity)
            },
            "Gross Margin Ratio": {
                "gross_profit": gross_profit,
                "net_sales": net_sales,
                "ratio_value": calculate_gross_margin_ratio(gross_profit, net_sales)
            },
            "Operating Margin Ratio": {
                "operating_income": operating_income,
                "net_sales": net_sales,
                "ratio_value": calculate_operating_margin_ratio(operating_income, net_sales)
            },
            "Return on Assets Ratio": {
                "net_income": net_income,
                "total_assets": total_assets,
                "ratio_value": calculate_return_on_assets_ratio(net_income, total_assets)
            },
            "Return on Equity Ratio": {
                "net_income": net_income,
                "shareholders_equity": shareholders_equity,
                "ratio_value": calculate_return_on_equity_ratio(net_income, shareholders_equity)
            },
            "Asset Turnover Ratio": {
                "net_sales": net_sales,
                "average_total_assets": average_total_assets,
                "ratio_value": calculate_asset_turnover_ratio(net_sales, average_total_assets)
            },
            "Inventory Turnover Ratio": {
                "cost_of_goods_sold": cost_of_goods_sold,
                "average_inventory": average_inventory,
                "ratio_value": calculate_inventory_turnover_ratio(cost_of_goods_sold, average_inventory)
            },
            "Receivables Turnover Ratio": {
                "net_credit_sales": net_credit_sales,
                "average_accounts_receivable": average_accounts_receivable,
                "ratio_value": calculate_receivables_turnover_ratio(net_credit_sales, average_accounts_receivable)
            },
            "Debt Ratio": {
                "total_liabilities": total_liabilities,
                "total_assets": total_assets,
                "ratio_value": calculate_debt_ratio(total_liabilities, total_assets)
            },
            "Interest Coverage Ratio": {
                "operating_income": operating_income,
                "interest_expenses": interest_expenses,
                "ratio_value": calculate_interest_coverage_ratio(operating_income, interest_expenses)
            }
        }
        
        return ratios
    
    def analyze_financial_document(self, document_content: str) -> Dict[str, Any]:
        """
        Analyze a financial document and return extracted data and calculated ratios
        
        Args:
            document_content: The text content of the financial document
            
        Returns:
            Dict with extracted financial data and calculated ratios
        """
        try:
            # Step 1: Extract financial data
            extraction_result = self.extraction_chain.invoke({"document_content": document_content})
            
            # Convert the string JSON to a Python dict
            try:
                extracted_data = json.loads(extraction_result["extracted_data"])
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response text
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', extraction_result["extracted_data"], re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group(1))
                else:
                    raise ValueError("Failed to extract valid JSON from LLM response")
            
            # Step 2: Calculate financial ratios
            calculated_ratios = self.calculate_financial_ratios(extracted_data)
            
            # Return combined result
            return {
                "extracted_data": extracted_data,
                "calculated_ratios": calculated_ratios
            }
            
        except Exception as e:
            import logging
            logging.exception("Error in analyze_financial_document")
            # Return error information
            return {
                "error": str(e),
                "extracted_data": {},
                "calculated_ratios": {}
            }
            
    def generate_business_overview(self, extracted_data: Dict) -> str:
        """
        Generate a concise business overview based on the extracted data.
        """
        overview_template = """
Based on the extracted financial data, provide a concise business overview:
{extracted_data}

Output only the business overview text.
"""
        prompt = overview_template.format(extracted_data=json.dumps(extracted_data, indent=2))
        response = self.analysis_llm.generate([prompt])
        # Extract generated text from the first generation instead of using response.text
        return response.generations[0].text.strip()
        
    def generate_key_findings(self, extracted_data: Dict, calculated_ratios: Dict) -> str:
        """
        Generate key findings and insights based on the extracted data and calculated ratios.
        """
        findings_template = """
Analyze the following extracted financial data and calculated ratios, and provide key findings 
with focus on profitability, liquidity, solvency, and any notable trends:

Extracted Data:
{extracted_data}

Calculated Ratios:
{calculated_ratios}

Output only the key findings text.
"""
        prompt = findings_template.format(
            extracted_data=json.dumps(extracted_data, indent=2),
            calculated_ratios=json.dumps(calculated_ratios, indent=2)
        )
        response = self.analysis_llm.generate([prompt])
        return response.generations[0].text.strip()
