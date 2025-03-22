EXTRACTION_PROMPT = """You are a financial analyst tasked with extracting key data from financial statements.
Please extract the following information from the provided document and output it in JSON format:

```json
{
"company_name": "",
"reporting_period": "",
"currency": "",
"income_statement": {
"net_sales": null,
"cost_of_goods_sold": null,
"gross_profit": null,
"operating_expenses": null,
"operating_income": null,
"interest_expenses": null,
"net_income": null
},
"balance_sheet": {
"cash_and_equivalents": null,
"current_assets": null,
"total_assets": null,
"current_liabilities": null,
"total_liabilities": null,
"shareholders_equity": null,
"average_inventory": null,
"average_accounts_receivable": null
},
"notes": {
"adj_ebitda_available": false,
"adj_ebitda_details": "",
"adj_working_capital_available": false,
"adj_working_capital_details": ""
}
}
```

If any information is not available, use null for that value.
If numbers have units (like thousands or millions), make sure to convert them to actual numbers and not include the units in the JSON values.
If you see values for multiple years, use the most recent year's data.
For average values (like average inventory), calculate them if provided with beginning and ending values, or use the most recent value if only one is available.

Remember to format your response ONLY as valid JSON within the ```json and ``` tags. Do not add any additional explanation before or after the JSON."""

OVERVIEW_PROMPT = """Based on the extracted financial data, provide a concise business overview:
{extracted_data}

If there is no data available, indicate that the financial information is insufficient to provide an overview.
Output only the business overview text."""

FINDINGS_PROMPT = """Analyze the following extracted financial data and calculated ratios, and provide key findings 
with focus on profitability, liquidity, solvency, and any notable trends:

Extracted Data:
{extracted_data}

Calculated Ratios:
{calculated_ratios}

If data is insufficient, please indicate what specific information is missing that would be needed for a proper analysis.
Output only the key findings text."""
