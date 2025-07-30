# Standard Libraries
import os  # noqa: F401
import pandas as pd  # noqa: F401

# Loaders
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

# Text Splitters
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# Embeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings as OpenAIEmbeddingsV2, AzureOpenAIEmbeddings

# Vector Stores
from langchain_community.vectorstores import Chroma, FAISS

# Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Storage
from langchain.storage import InMemoryStore

# Chains and Models
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

# Prompts and Parsing
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from dotenv import load_dotenv

# Utilities
from langchain_experimental.utilities import PythonREPL
from src.config import CFG
from src.default_descriptions import table_description_list_default

# Load environment variables
load_dotenv(CFG.env_file)

llm = ChatOpenAI(model='gpt-4.1')


df = pd.read_excel("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/table_description 1.xlsx")

ground_truth = pd.read_csv("/Users/eduard_babayan/Documents/personal_projects/geoquery-ai/data/golden_dataset_export_20250716_095952 1.csv")[['NL_QUERY', "TABLES"]]

ground_truth_renamed = ground_truth.rename(columns={
    'NL_QUERY': 'question',
    'TABLES': 'tables'
})

json_data = ground_truth_renamed.to_dict('records')

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


chain = prompt | llm | JsonOutputParser()

result = chain.invoke({"TABLE_DESCRIPTION": json_data})

result_full = {}
for n, i in enumerate(result[:13]):
    print(i)
    result_full[i["table"]] = i["questions"] + result[n]["questions"]

import json

# Save result_full as JSON
with open('result_full.json', 'w') as f:
    json.dump(result_full, f, indent=2)

description_strings = []
for i in df.index:
    description_strings.append(f"Table name is {df.loc[i, "name"]}. Industry terms are {df.loc[i, "industry_terms"]}."
                               f" Data granularity is {df.loc[i, "data_granularity"]}. Main business purpose is "
                               f"{df.loc[i, "main_business_purpose"]}. Alternative business purpose is {df.loc[i, 
                               "alternative_business_purpose"]}. Unique insights are {df.loc[i, "unique_insights"]}.")


description_strings_questions = []
for i in df.index:
    matching_questions = [k["questions"] for k in result if table_description_list_default[i]['table_name'].split('.')[-1] == k["table"]]
    questions_text = ' '.join(matching_questions[0]) if matching_questions else ""
    description_strings_questions.append(f"Table name is {df.loc[i, 'name']}. Industry terms are {df.loc[i, 'industry_terms']}. Data granularity is {df.loc[i, 'data_granularity']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}. Question examples are: {questions_text}")


description_strings_questions_v2 = []
for i in df.index:
    matching_questions = [k["questions"] for k in result if table_description_list_default[i]['table_name'].split('.')[-1] == k["table"]]
    questions_text = ' '.join(matching_questions[0]) if matching_questions else ""
    description_strings_questions_v2.append(f"Table name is {df.loc[i, 'name']}. Main business purpose is {df.loc[i, 
    'main_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}. Question examples are: {questions_text}")


description_strings_questions_v3 = []
for i in df.index:
    matching_questions = [k["questions"] for k in result if table_description_list_default[i]['table_name'].split('.')[-1] == k["table"]]
    description_strings_questions_v3.append(f"Table name is {df.loc[i, 'name']}. Industry terms are {df.loc[i,
    'industry_terms']}. Data granularity is {df.loc[i, 'data_granularity']}. Main business purpose is {df.loc[i, 'main_business_purpose']}. Alternative business purpose is {df.loc[i, 'alternative_business_purpose']}. Unique insights are {df.loc[i, 'unique_insights']}.")

embed = OpenAIEmbeddings(model="text-embedding-3-large")

documents_artem_description = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions)]

documents_artem_v2_description = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions_v2)]

documents_artem_v3_description = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions_v3)]


def create_ensemble_retriever_from_documents(documents, num_docs_retrieved=5, weights = [0.5,0.5], name=""):
  bm25_retriever = BM25Retriever.from_documents(
      documents
  )
  bm25_retriever.k = num_docs_retrieved
  vectorestore = Chroma.from_documents(documents, embed, collection_name=f"test_{name}")
  vector_retriever = vectorestore.as_retriever(search_kwargs={"k": num_docs_retrieved})
  ensemble_retriever = EnsembleRetriever(
      retrievers=[bm25_retriever, vector_retriever], weights=weights
  )
  return ensemble_retriever

retriever_artem_v1 = create_ensemble_retriever_from_documents(documents_artem_description, name="artem_v1")
retriever_artem_v2 = create_ensemble_retriever_from_documents(documents_artem_v2_description, name="artem_v2")
# Create documents from table_description_list_default with proper formatting
default_documents = [Document(page_content=f"Table name is {table_info['table_name'].split('.')[-1]}. Description: {table_info['table_description']}", metadata={"table_name": table_info['table_name'].split('.')[-1]}) for table_info in table_description_list_default]
retriever_default = create_ensemble_retriever_from_documents(default_documents, name="default_v1")
retriever_artem_v3 = create_ensemble_retriever_from_documents(documents_artem_v3_description, name="artem_v3")

query = "Can you plot the wellhead pressure vs. time for the top 5 producing wells over the last 30 days?"

retriever_artem_v1.invoke(query)
retriever_artem_v2.invoke(query)
retriever_default.invoke(query)
retriever_artem_v3.invoke(query)


description_strings_questions_v4 = []
for i in df.index:
    matching_questions = [k["questions"] for k in result if table_description_list_default[i]['table_name'].split('.')[-1] == k["table"]]
    questions_text = ' '.join(matching_questions[0]) if matching_questions else ""
    description_strings_questions_v4.append(f"Table name is {df.loc[i, 'name']}. Main business purpose is {df.loc[i, 'main_business_purpose']} Unique insights are {df.loc[i, 'unique_insights']}")



prompt = PromptTemplate(
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
        input_variables=["TABLE_DESCRIPTION"]
    )


chain_new = prompt | llm | JsonOutputParser()


documents_artem_v4_description = [Document(page_content=desc, metadata={"table_name": (table_description_list_default[i]["table_name"]).split(".")[-1]}) for i, desc in enumerate(description_strings_questions_v4)]

# Read the Excel file
columns_description = pd.read_excel('/Users/artem_nikulchev_1/vscode_rep/cor_42/board-navigator-backend/notebooks/columns_description_new.xlsx')

questions_for_columns = []
columns_list = []
for index, row in columns_description.iterrows():
    table_name = row['table_name']
    final_table_name = table_name.split(".")[-1]
    column_name = row['column_name']
    if column_name in columns_list:
        continue
    columns_list.append(column_name)
    description = row['description']
    print(column_name)
    t = chain_new.invoke({"TABLE_DESCRIPTION": [i.page_content for i in documents_artem_v4_description if
                                                i.metadata['table_name'] == final_table_name],
                          "COLUMN_DESCRIPTION": f"Column name: {column_name}. Description: {description}"})
    questions_for_columns.append(t)


column_questions_dict = {}
for item in questions_for_columns:
    column_info = item[0]
    column_name = column_info['column']
    questions = column_info['questions']
    column_questions_dict[column_name] = ', '.join(questions)

# Create DataFrame with one row per column
df_questions_compact = pd.DataFrame([
    {'column': col, 'questions': questions}
    for col, questions in column_questions_dict.items()
])
df_questions_compact.to_csv('questions_for_columns_compact.csv', index=False)
print(f"Saved {len(df_questions_compact)} columns with comma-separated questions to questions_for_columns_compact.csv")

