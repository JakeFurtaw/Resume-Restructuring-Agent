from abc import ABC
from llama_index.core.agent import AgentRunner, CustomSimpleAgentWorker
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, Document
from llama_parse import LlamaParse
from llama_index.core.tools import FunctionTool
import os
from fpdf import FPDF
from datetime import datetime

# Initialize LLMs
runner_llm = Ollama(model="mistral-nemo:latest")
worker_llm = Ollama(model="gemma2:2b")

# Initialize LlamaParse for parsing resumes
llama_parse = LlamaParse()


def read_files_from_data_directory():
    """Read resume and job description files from the 'data' directory."""
    reader = SimpleDirectoryReader(input_dir="./data")
    documents = reader.load_data()
    resume_doc = next((doc for doc in documents if doc.metadata['file_name'].lower().endswith('.pdf')), None)
    job_description_doc = next((doc for doc in documents if doc.metadata['file_name'].lower().endswith('.txt')), None)
    if not resume_doc or not job_description_doc:
        raise ValueError("Could not find both resume (PDF) and job description (TXT) in the 'data' directory.")
    return resume_doc, job_description_doc


def parse_resume(resume_content: str) -> str:
    """Parse the resume using LlamaParse."""
    # Save the content to a temporary file
    temp_file = "temp_resume.pdf"
    with open(temp_file, "wb") as f:
        f.write(resume_content.encode())
    # Parse the temporary file
    parsed_doc = llama_parse.parse(temp_file)
    # Remove the temporary file
    os.remove(temp_file)
    return parsed_doc.text


def restructure_resume(resume: str, job_description: str) -> str:
    """Restructure the resume based on the job description."""
    prompt = f"""
    You are a professional resume writer. Your task is to restructure the given resume to better match the provided job description.
    Focus on highlighting relevant skills and experiences, and adjusting the language to align with the job requirements.
    Resume:
    {resume}
    Job Description:
    {job_description}
    Please provide the restructured resume:
    """
    response = worker_llm.complete(prompt)
    return response.text


def write_resume_to_file(resume_content: str) -> str:
    """Write the resume content to a text file."""
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"restructured_resume_{timestamp}.txt"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as f:
        f.write(resume_content)
    return file_path


def create_pdf_from_text(text_file_path: str) -> str:
    """Create a PDF file from the given text file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    with open(text_file_path, "r") as f:
        for line in f:
            pdf.cell(0, 10, txt=line.strip(), ln=True)
    pdf_file_path = text_file_path.rsplit(".", 1)[0] + ".pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path


# Define tools for the agent
tools = [
    FunctionTool.from_defaults(fn=read_files_from_data_directory),
    FunctionTool.from_defaults(fn=parse_resume),
    FunctionTool.from_defaults(fn=restructure_resume),
    FunctionTool.from_defaults(fn=write_resume_to_file),
    FunctionTool.from_defaults(fn=create_pdf_from_text),
]


class ResumeWorkerAgent(CustomSimpleAgentWorker, ABC):
    def __init__(self):
        super().__init__(tools=tools, llm=worker_llm)

    def execute_task(self, task_description: str, context: dict):
        # Read files from the data directory
        resume_doc, job_description_doc = self.run_tool("read_files_from_data_directory")
        # Parse the resume
        parsed_resume = self.run_tool("parse_resume", {"resume_content": resume_doc.text})
        # Restructure the resume
        restructured_resume = self.run_tool("restructure_resume",
                                            {"resume": parsed_resume, "job_description": job_description_doc.text})
        # Write the restructured resume to a text file
        text_file_path = self.run_tool("write_resume_to_file", {"resume_content": restructured_resume})
        # Create a PDF from the text file
        pdf_file_path = self.run_tool("create_pdf_from_text", {"text_file_path": text_file_path})
        return f"Restructured resume saved as text file: {text_file_path}\nPDF version saved as: {pdf_file_path}"


# Create the agent runner
agent_runner = AgentRunner.from_llm(
    llm=runner_llm,
    worker=ResumeWorkerAgent()
)

# Example usage
if __name__ == "__main__":
    task_description = "Parse the resume, read the job description, restructure the resume, and save it as both text and PDF files."
    result = agent_runner.chat(task_description)
    print(result.response)