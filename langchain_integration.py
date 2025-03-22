from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any
from dotenv import load_dotenv
import os
import json
import re
import logging
import time
from financial_tools import (
    calculate_current_ratio, calculate_debt_to_equity_ratio,
    calculate_gross_margin_ratio, calculate_operating_margin_ratio,
    calculate_return_on_assets_ratio, calculate_return_on_equity_ratio,
    calculate_asset_turnover_ratio, calculate_inventory_turnover_ratio,
    calculate_receivables_turnover_ratio, calculate_debt_ratio,
    calculate_interest_coverage_ratio
)
from prompts import EXTRACTION_PROMPT, OVERVIEW_PROMPT, FINDINGS_PROMPT

load_dotenv()

class LangChainHandler:
    """Handles multi-step LLM operations for financial analysis."""
    
    def __init__(self):
        # Use gemini-2.0-flash for data extraction and gemini-2.0-flash-thinking for reasoning
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        self.extraction_llm = GoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.1,
            google_api_key=api_key
        )
        self.analysis_llm = GoogleGenerativeAI(
            model="gemini-2.0-flash-thinking-exp-01-21", 
            temperature=0.2,
            google_api_key=api_key
        )
        
        # Setup the extraction chain
        self._setup_extraction_chain()
        
        # Setup the analysis chains
        self._setup_analysis_chains()
        
    def _setup_extraction_chain(self):
        """Set up the extraction chain for financial data"""
        extraction_template = EXTRACTION_PROMPT
        self.extraction_prompt = ChatPromptTemplate.from_template(extraction_template)
        self.extraction_chain = LLMChain(
            llm=self.extraction_llm, 
            prompt=self.extraction_prompt, 
            output_key="extracted_data",
            verbose=True
        )
    
    def _setup_analysis_chains(self):
        """Set up chains for business overview and key findings"""
        overview_template = OVERVIEW_PROMPT
        self.overview_prompt = ChatPromptTemplate.from_template(overview_template)
        self.overview_chain = LLMChain(
            llm=self.analysis_llm,
            prompt=self.overview_prompt,
            output_key="business_overview",
            verbose=True
        )
        
        findings_template = FINDINGS_PROMPT
        self.findings_prompt = ChatPromptTemplate.from_template(findings_template)
        self.findings_chain = LLMChain(
            llm=self.analysis_llm,
            prompt=self.findings_prompt,
            output_key="key_findings",
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
        
        # Helper function to safely calculate ratios
        def safe_calculate(func, numerator, denominator, default=None):
            try:
                if denominator == 0:
                    return default
                return func(numerator, denominator)
            except Exception:
                return default
        
        # Calculate all ratios using our financial_tools.py functions
        ratios = {
            "Current Ratio": {
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "ratio_value": safe_calculate(calculate_current_ratio, current_assets, current_liabilities, "N/A")
            },
            "Cash Ratio": {
                "cash_and_equivalents": cash_and_equivalents,
                "current_liabilities": current_liabilities,
                "ratio_value": safe_calculate(
                    lambda cash, liab: cash / liab if liab != 0 else "N/A",
                    cash_and_equivalents, current_liabilities, "N/A"
                )
            },
            "Debt to Equity Ratio": {
                "total_liabilities": total_liabilities,
                "shareholders_equity": shareholders_equity,
                "ratio_value": safe_calculate(calculate_debt_to_equity_ratio, total_liabilities, shareholders_equity, "N/A")
            },
            "Gross Margin Ratio": {
                "gross_profit": gross_profit,
                "net_sales": net_sales,
                "ratio_value": safe_calculate(calculate_gross_margin_ratio, gross_profit, net_sales, "N/A")
            },
            "Operating Margin Ratio": {
                "operating_income": operating_income,
                "net_sales": net_sales,
                "ratio_value": safe_calculate(calculate_operating_margin_ratio, operating_income, net_sales, "N/A")
            },
            "Return on Assets Ratio": {
                "net_income": net_income,
                "total_assets": total_assets,
                "ratio_value": safe_calculate(calculate_return_on_assets_ratio, net_income, total_assets, "N/A")
            },
            "Return on Equity Ratio": {
                "net_income": net_income,
                "shareholders_equity": shareholders_equity,
                "ratio_value": safe_calculate(calculate_return_on_equity_ratio, net_income, shareholders_equity, "N/A")
            },
            "Asset Turnover Ratio": {
                "net_sales": net_sales,
                "average_total_assets": average_total_assets,
                "ratio_value": safe_calculate(calculate_asset_turnover_ratio, net_sales, average_total_assets, "N/A")
            },
            "Inventory Turnover Ratio": {
                "cost_of_goods_sold": cost_of_goods_sold,
                "average_inventory": average_inventory,
                "ratio_value": safe_calculate(calculate_inventory_turnover_ratio, cost_of_goods_sold, average_inventory, "N/A")
            },
            "Receivables Turnover Ratio": {
                "net_credit_sales": net_credit_sales,
                "average_accounts_receivable": average_accounts_receivable,
                "ratio_value": safe_calculate(calculate_receivables_turnover_ratio, net_credit_sales, average_accounts_receivable, "N/A")
            },
            "Debt Ratio": {
                "total_liabilities": total_liabilities,
                "total_assets": total_assets,
                "ratio_value": safe_calculate(calculate_debt_ratio, total_liabilities, total_assets, "N/A")
            },
            "Interest Coverage Ratio": {
                "operating_income": operating_income,
                "interest_expenses": interest_expenses,
                "ratio_value": safe_calculate(calculate_interest_coverage_ratio, operating_income, interest_expenses, "N/A")
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
            # Validate input
            if not document_content or len(document_content.strip()) == 0:
                raise ValueError("Document content is empty")
                
            logging.info(f"Starting financial document analysis. Document length: {len(document_content)} characters")
            start_time = time.time()
            
            # Step 1: Extract financial data
            logging.info("Step 1: Extracting financial data...")
            extraction_result = self.extraction_chain.invoke({"document_content": document_content})
            
            logging.info(f"Raw extraction result: {extraction_result['extracted_data'][:200]}...")
            
            # Convert the string JSON to a Python dict
            try:
                # First try direct JSON parsing
                extracted_data = json.loads(extraction_result["extracted_data"])
                logging.info("Successfully parsed JSON directly")
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the response text
                logging.info("Direct JSON parsing failed, trying to extract JSON from text")
                json_match = re.search(r'```json\s*(.*?)\s*```', extraction_result["extracted_data"], re.DOTALL)
                if json_match:
                    try:
                        extracted_data = json.loads(json_match.group(1))
                        logging.info("Successfully extracted and parsed JSON from text")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing error after extraction: {e}")
                        raise ValueError(f"Invalid JSON format after extraction: {e}")
                else:
                    # If no JSON block found, provide a fallback empty structure
                    logging.error("No valid JSON found in extraction result")
                    extracted_data = {
                        "company_name": "",
                        "reporting_period": "",
                        "currency": "",
                        "income_statement": {},
                        "balance_sheet": {},
                        "notes": {
                            "adj_ebitda_available": False,
                            "adj_ebitda_details": "",
                            "adj_working_capital_available": False,
                            "adj_working_capital_details": ""
                        }
                    }
                    
            # Step 2: Calculate financial ratios
            logging.info("Step 2: Calculating financial ratios...")
            calculated_ratios = self.calculate_financial_ratios(extracted_data)
            
            # Step 3: Generate business overview and key findings
            logging.info("Step 3: Generating business overview and key findings...")
            business_overview = self.generate_business_overview(extracted_data)
            key_findings = self.generate_key_findings(extracted_data, calculated_ratios)
            
            end_time = time.time()
            logging.info(f"Financial analysis completed in {end_time - start_time:.2f} seconds")
            
            # Return combined result
            return {
                "extracted_data": extracted_data,
                "calculated_ratios": calculated_ratios,
                "business_overview": business_overview,
                "key_findings": key_findings
            }
            
        except Exception as e:
            logging.exception("Error in analyze_financial_document")
            # Return error information
            return {
                "error": str(e),
                "extracted_data": {},
                "calculated_ratios": {},
                "business_overview": "Error analyzing document: " + str(e),
                "key_findings": "Error analyzing document: " + str(e)
            }
            
    def generate_business_overview(self, extracted_data: Dict) -> str:
        """
        Generate a concise business overview based on the extracted data.
        """
        try:
            result = self.overview_chain.invoke({
                "extracted_data": json.dumps(extracted_data, indent=2)
            })
            return result["business_overview"].strip()
        except Exception as e:
            logging.exception("Error generating business overview")
            return f"Error generating business overview: {str(e)}"
        
    def generate_key_findings(self, extracted_data: Dict, calculated_ratios: Dict) -> str:
        """
        Generate key findings and insights based on the extracted data and calculated ratios.
        """
        try:
            result = self.findings_chain.invoke({
                "extracted_data": json.dumps(extracted_data, indent=2),
                "calculated_ratios": json.dumps(calculated_ratios, indent=2)
            })
            return result["key_findings"].strip()
        except Exception as e:
            logging.exception("Error generating key findings")
            return f"Error generating key findings: {str(e)}"