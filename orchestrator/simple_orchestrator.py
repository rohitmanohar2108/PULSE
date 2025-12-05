import asyncio
import os
import sys
import logging
import subprocess
from typing import Tuple
from pathlib import Path
from litellm import completion


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.main import RAGApplication
from speech.SpeechProcessor import SpeechProcessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(SCRIPT_DIR, "vector_store")
PROCESSED_DIR = os.path.join(STORE_DIR, "processed_files")
PROCESSED_DB = os.path.join(STORE_DIR, "processed_db.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleOrchestrator:
    def __init__(self):
        # Initialize RAG
        self.rag_agent = self._setup_rag()

        # Configure LLM settings
        self.api_base = "http://localhost:11434"
        self.bash_model = "ollama/codellama"
        self.query_model = "ollama/mistral"

    def _setup_rag(self) -> RAGApplication:
        rag = RAGApplication(model_name="mistral", temperature=0.2)
        data_dir = os.path.join(SCRIPT_DIR, "data")
        if rag.load_and_process_documents(data_dir):
            logger.info("Documents processed successfully!")
        return rag

    def get_bash_command(self, query: str) -> str:
        """Convert natural language to bash command"""
        try:
            response = completion(
                model=self.bash_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a bash command generator. Return ONLY the command, no explanations. ",
                    },
                    {"role": "user", "content": f"Convert to bash command: {query}"},
                ],
                api_base=self.api_base,
                temperature=0.1,
            )
            # Clean up the response - remove backticks, markdown formatting and extra whitespace
            command = response.choices[0].message.content
            command = command.replace("bash", "").strip()
            command = command.replace("```", "").strip()
            command = command.replace("`", "").strip()
            return command
        except Exception as e:
            logger.error(f"Bash generation error: {str(e)}")
            return f"Error: {str(e)}"

    def classify_query(self, query: str) -> str:
        """Determine if query needs bash command or RAG"""
        try:
            response = completion(
                model=self.query_model,
                messages=[
                    {
                        "role": "user",
                        "content": "Classify as 'bash' or 'rag' (single word only):\n"
                        "- 'rag' for questions related to the documents parsed. These are always questions using words like What, Where, How, When, etc.\n"
                        "- 'bash' for file/system commands. These will always be statements, mentioning 'Execute the following command'.\n"
                        f"Query: {query}",
                    }
                ],
                api_base=self.api_base,
                temperature=0.1,
            )
            # Clean up the response to get just 'bash' or 'rag'
            raw_classification = response.choices[0].message.content.strip().lower()
            classification = "bash" if "bash" in raw_classification else "rag"
            logger.info(f"Query classified as: {classification}")
            return classification
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return "rag"

    def execute_command(self, command: str) -> Tuple[str, bool]:
        """Execute bash command and return output and success status"""
        try:
            # Run command and capture output
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True,
                timeout=30,  # 30 second timeout
            )

            # Get command output (stdout or stderr)
            output = result.stdout if result.stdout else result.stderr
            success = result.returncode == 0

            if success:
                logger.info(f"Command executed successfully: {command}")
            else:
                logger.error(f"Command failed with error: {result.stderr}")

            return output.strip(), success
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds", False
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return str(e), False

    async def process_query(self, query: str) -> str:
        """Main query processing method"""
        try:
            query_type = self.classify_query(query)

            if query_type == "bash":
                logger.info("Generating bash command...")
                command = self.get_bash_command(query)
                logger.info(f"Generated command: {command}")

                # Ask for confirmation before executing
                confirm = (
                    input(f"\nExecute command '{command}'? (y/n): ").lower().strip()
                )
                if confirm == "y":
                    output, success = self.execute_command(command)
                    status = "✓" if success else "✗"
                    return f"Command {status}: {command}\nOutput:\n{output}"
                else:
                    return f"Command not executed: {command}"
            else:
                logger.info("Routing to RAG system...")
                return await self.rag_agent.query_document(query)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}"


async def main():
    use_groq = False
    orchestrator = SimpleOrchestrator()
    talker = SpeechProcessor()
    
    while True:
        print("\nAsk your question! Or say please quit to exit: ")
        talker.text_to_speech(text="Ask your question! Or say please quit to exit.", is_groq=use_groq, output_file="speech.wav")
        talker.play_audio("speech.wav")
        
        recorded_file = talker.record_audio("my_recording.wav")
        transcription = talker.speech_to_text(recorded_file)
        
        # query = input("\nEnter your question (or 'quit' to exit): ").strip()
        transcription = transcription.strip()
        print(f"Transcription: {transcription}")
        if transcription.lower() == 'please quit!' or transcription.lower() == 'please quit' or transcription.lower() == 'please quit.':
            break
            
        print("\nProcessing query...")
        response = await orchestrator.process_query(transcription)
        print(f"\nResponse: {response}")
        talker.text_to_speech(response, is_groq=use_groq, output_file="speech.wav")
        talker.play_audio("speech.wav")


if __name__ == "__main__":
    asyncio.run(main())
