from abc import ABC
from llama_index.core.agent import AgentRunner, CustomSimpleAgentWorker
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from llama_index.core.tools import FunctionTool
import os

# Initialize LLMs
runner_llm = Ollama(model="mistral-nemo:latest", request_timeout=60)
worker_llm = Ollama(model="gemma2:2b", request_timeout=60)

# Initialize LlamaParse for parsing resumes
llama_parse = LlamaParse(api_key="")


def read_files_from_data_directory():
    """Read resume and job description files from the 'data' directory."""
    reader = SimpleDirectoryReader(input_dir="data")
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
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"restructured_resume.txt"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as f:
        f.write(resume_content)
    return file_path

# Define tools for the agent
tools = [
    FunctionTool.from_defaults(fn=read_files_from_data_directory),
    FunctionTool.from_defaults(fn=parse_resume),
    FunctionTool.from_defaults(fn=restructure_resume),
    FunctionTool.from_defaults(fn=write_resume_to_file),
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
        return f"Restructured resume saved as text file: {text_file_path}"

    def _initialize_state(self):
        # Initialize any state needed for the agent
        pass

    def _run_step(self, state):
        # This method is not used in our implementation
        # as we're using execute_task instead
        pass

    def _finalize_task(self, state):
        # Finalize the task if needed
        pass

# Create the agent runner
agent_runner = AgentRunner.from_llm(
    llm=runner_llm,
    worker=ResumeWorkerAgent()
)

# Example usage
if __name__ == "__main__":
    task_description = "Review the resume and read the job description, restructure the supplied resume keeping the same sections and relevant details but remove or adding any details needed or not need for the job, and save it as a text."
    result = agent_runner.chat(task_description)
    print(result.response)