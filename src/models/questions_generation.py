"""
Question generation for table descriptions
"""

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import JsonOutputParser
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG

# Load environment variables
load_dotenv(CFG.env_file)

# Initialize LLM
llm = ChatOpenAI(model='gpt-4.1')

# Prompt for generating questions for tables
prompt = PromptTemplate(
    template="""
    You are tasked with generating example questions for given tables and formatting the output in JSON. 
    Your input will be a set of tables, and your output should be pairs of tables and corresponding example questions that a user might ask about the data in those tables.
    First, carefully examine the input tables provided in the {{TABLES}} variable. 
    Each table may contain different types of data, so pay attention to the column headers and the information presented.
    For each table:
    1. Generate 3-5 example questions that a user might ask about the tables based on the question examples provided to you. These questions should:
    - Be diverse and cover different aspects of the data
    - Range from simple to more complex queries
    - Not be exact copies of the actual questions, but similar in nature
    Format your output as a JSON array with objects containing two keys: "table" and "questions". 
    The "table" should be the name of the table, and the "questions" value should be an array of example questions.
    This is the desription and questions to the tables: 
    <table_description>     
    {TABLE_DESCRIPTION}
    </table_description>
    Ensure that your JSON is properly formatted and that each table is accurately represented as a string, including the header row and separators.
    Remember to create questions that are relevant to the this table based on examples, and try to showcase different types of queries that users might be interested in asking about the data.
""",
    input_variables=["TABLE_DESCRIPTION"]
)

# Prompt for generating questions for columns
prompt_columns = PromptTemplate(
    template="""
    You are oil and gas specialist from ADNOC.
    You are tasked with generating example questions for given columns and formatting the output in JSON. 
    Your input will be a set of tables, and your output should be pairs of tables and corresponding example questions that a user might ask about the data in those tables.
    First, carefully examine the input tables provided in the {{TABLE_DESCRIPTION}} variable.
    Second, carefully examine the input columns provided in the {{COLUMN_DESCRIPTION}} variable.
    For each column:
    1. Generate 3-5 example questions that a user might ask about the column based on your knowledge. These questions should:
    - Be diverse and cover different aspects of the data
    - Range from simple to more complex queries
    - Not be exact copies of the actual questions, but similar in nature
    - Use different entities from ADNOC in questions like wells, fields, etc. (for example, "What is the total production of BB034 in the BAB field?")
    Format your output as a JSON array with objects containing two keys: "column" and "questions". 
    The "column" should be the name of the column, and the "questions" value should be an array of example questions.
    This is the desription and questions to the tables: 
    <table_description>     
    {TABLE_DESCRIPTION}
    </table_description>
    <column_description>     
    {COLUMN_DESCRIPTION}
    </column_description>
    Ensure that your JSON is properly formatted and that each column is accurately represented as a string, including the header row and separators.
    Remember to create questions that are relevant to the this column based on examples, and try to showcase different types of queries that users might be interested in asking about the data.
""",
    input_variables=["TABLE_DESCRIPTION", "COLUMN_DESCRIPTION"]
)

# Create chains
chain = prompt | llm | JsonOutputParser()
chain_columns = prompt_columns | llm | JsonOutputParser()

# Example result structure from notebook
result_full = {
    'daily_allocation': [
        'Which five wells in SAHIL field had the highest oil output last year?',
        'How did oil production for well UZ0256 change over the past six months?',
        'What is the total oil production for well BU0182 in 2024?',
        'List wells in ASAB field with the largest water injection volumes for the last quarter.',
        'How many producing hours did BU0224 accumulate in 2025?',
        'Which wells achieved the highest oil production in the SAHIL field last year?',
        'Show the daily oil production rates for well BUHASA-100 over the past 6 months.',
        'How many hours did well UZ301 operate in oil production during 2023?',
        'What is the cumulative oil volume produced by well ASB-224 this year?',
        'Compare the oil production trends for the top 3 wells in the ZAKUM field.'
    ],
    'well': [
        'What is the total measured depth of well BA-234?',
        'Can you provide the UTM coordinates for well SH-118?',
        'Which wells have a surface location in the NEB area?',
        'What is the drill completion date of well ZK-38?',
        'List all wells in the offshore category.'
    ],
    'string_event': [
        'What is the current status (open/closed) of the string for well ASB-008?',
        'How did the tubing pressure and temperature for well RA075 vary over the last month?',
        'Show the trend of choke size changes for well SH66 in the past 60 days.',
        'When was the last flowing event recorded for well BU029?',
        'Did well BA251 undergo any significant string events in 2025?'
    ],
    'real_time_corporate_pi': [
        'Display the real-time wellhead temperature for well SH118 over this week.',
        'What are the latest wellhead pressure readings for all wells in BuHasa field?',
        'Show the wellhead pressure and temperature history for well ZK-8 over the last three months.',
        'Are there any wells in Rumaitha showing pressure anomalies in the last 2 weeks?',
        'Plot a moving average of real-time wellhead parameters for well RA0081 for the last 14 days.'
    ],
    'unified_pressure_test': [
        'How many BHP surveys were conducted in Bab field in March 2024?',
        'Which wells in Sahil had pressure tests conducted in May 2025?',
        'What is the historical reservoir pressure for well ZK-08 over the last 10 years?',
        'Which wells in ASAB field did not have a BHCIP in the last year?',
        'List all pressure tests from Umm Al Shaif field in 2025.'
    ],
    'wellbore': [
        'How many wellbores exist in the ZK-8 well?',
        'What is the spud date and final completion date for well BA-215?',
        'List all wells in BUHASA with multiple wellbores.',
        'How many new wellbores were started in 2024 in BAB field?',
        'What is the total number of production wellbores in ASAB field?'
    ],
    'inactive_string': [
        'How many wells in Asab field became inactive during March 2024?',
        'List all wells in Bu Hasa with subsurface integrity issues in the last year.',
        'Which wells were inactivated due to sanding problems in 2025?',
        'What is the average problem duration for inactive strings in Rumaitha over the last six months?',
        'How many inactive oil producers were recorded in April 2025 in BAB field?'
    ],
    'flow_test': [
        'What is the maximum GOR recorded for well SH0082 in the last two years?',
        'Show the water cut progression for each well in Kharaib-2, Rumaitha field.',
        'Plot a histogram of daily flow rate variations for wells in Asab field.',
        'Which wells in BAB field have experienced a water cut increase greater than 15% in the last year?',
        'How do flow rates compare between KHARAIB-1 and KHARAIB-2 reservoirs in the last 5 years?'
    ],
    'well_allowable_limits': [
        'What is the current production allowable rate for well ASB-123?',
        'Show the allowable injection limits for well UZ100 by reservoir over the last 5 years.',
        'Which wells have exceeded their maximum allowable production rate in the last quarter?',
        'List all materials used for maintaining allowable limits in well QW-21.',
        'How does the allowable rate for oil production in BAB field wells vary by reservoir?'
    ],
    'field': [
        'Which fields are categorized as onshore and which as offshore?',
        'List all fields included in the NEB region.',
        'How many fields are currently under operation?',
        'Show the distribution of fields by geographical location.',
        'Which field covers the largest area?'
    ],
    'well_log_index': [
        'What logging activities were performed on well BA-217 in the last 2 years?',
        'List all types of well logs acquired for well ZK-13 during 2023.',
        'How many logging operations occurred in well SH008 this year?',
        'What was the most recent logging activity in well UM-8?',
        'Show the logs conducted on US276 in the past three years.'
    ],
    'well_reservoir': [
        'How many wells are associated with each reservoir in Umm Al Shaif field?',
        'List all wells penetrating the Kharaib-2 reservoir in Bu Hasa field.',
        'Show the count of injector wells by reservoir in Asab field.',
        'Which reservoir in Rumaitha has the most producing wells?',
        'How many wells are completed in the Bab Kharaib-1 reservoir?'
    ],
    'well_completion': [
        'What type of completions (packers, tubing size, etc.) were installed in well BA-159?',
        'List the completion details for all wells with recent workovers in Asab field.',
        'What are the packer and tubing specifications for well ZK-708?',
        'Which wells had sand-control completions installed in 2024?',
        'How does the completion type distribution vary across Bu Hasa field?'
    ]
}

# Function to generate questions for tables
def generate_questions_for_tables(table_descriptions):
    """Generate questions for given table descriptions"""
    return chain.invoke({"TABLE_DESCRIPTION": table_descriptions})

# Function to generate questions for columns
def generate_questions_for_columns(table_descriptions, column_info):
    """Generate questions for specific columns"""
    return chain_columns.invoke({
        "TABLE_DESCRIPTION": table_descriptions,
        "COLUMN_DESCRIPTION": column_info
    })