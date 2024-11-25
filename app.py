import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openpyxl
from IPython import get_ipython
import ast
from types import FunctionType
from typing import Dict, Any, Optional, List
import logging
import traceback

class CodeValidator:
    """Validates and cleans generated code for safe execution."""
    
    @staticmethod
    def clean_code(generated_code: str) -> str:
        """
        Cleans generated code by removing non-code narrative lines and validating syntax.
        
        Args:
            generated_code: Raw generated code string
            
        Returns:
            Cleaned and validated code string
        """
        valid_lines = []
        current_block = []
        
        for line in generated_code.split("\n"):
            line = line.rstrip()
            if not line:
                continue
                
            current_block.append(line)
            block_text = "\n".join(current_block)
            
            try:
                ast.parse(block_text)
                if len(current_block) == 1:
                    valid_lines.append(line)
                    current_block = []
            except SyntaxError:
                continue
                
        return "\n".join(valid_lines)
    
    @staticmethod
    def validate_function_definition(code: str) -> Optional[str]:
        """
        Validates if the code contains a proper function definition.
        
        Args:
            code: Code string to validate
            
        Returns:
            Function name if valid, None otherwise
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            return None
        return None

class ModelManager:
    """Manages the ML model and tokenizer."""
    
    def __init__(self, checkpoint: str):
        """
        Initialize model manager with specified checkpoint.
        
        Args:
            checkpoint: HuggingFace model checkpoint
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)
        
    def generate_code(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """
        Generate code from prompt using the loaded model.
        
        Args:
            prompt: Input prompt for code generation
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            
        Returns:
            Generated code string
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class PythonActionExecutor:
    """Executes and manages generated Python code actions."""
    
    def __init__(self):
        """Initialize the executor with IPython kernel and action library."""
        self.ipython_kernel = get_ipython()
        self.action_library: Dict[str, FunctionType] = {}
        self.logger = logging.getLogger(__name__)
        
    def execute(self, action_code: str) -> Dict[str, Any]:
        """
        Execute provided code in a safe environment.
        
        Args:
            action_code: Python code to execute
            
        Returns:
            Dictionary containing execution results or error
        """
        local_env = {}
        try:
            exec(action_code, globals(), local_env)
            return local_env
        except Exception as e:
            self.logger.error(f"Execution error: {str(e)}\n{traceback.format_exc()}")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def add_action(self, func_name: str, func_code: str) -> bool:
        """
        Add a new action to the library.
        
        Args:
            func_name: Name of the function
            func_code: Function code string
            
        Returns:
            Boolean indicating success
        """
        try:
            exec(func_code, globals())
            self.action_library[func_name] = globals()[func_name]
            return True
        except Exception as e:
            self.logger.error(f"Failed to add action {func_name}: {str(e)}")
            return False
    
    def retrieve_action(self, func_name: str) -> Optional[FunctionType]:
        """
        Retrieve an action from the library.
        
        Args:
            func_name: Name of the function to retrieve
            
        Returns:
            Function object if found, None otherwise
        """
        return self.action_library.get(func_name)

class CodeGenerationPipeline:
    """Main pipeline for code generation and execution."""
    
    def __init__(self, checkpoint: str):
        """
        Initialize the pipeline.
        
        Args:
            checkpoint: HuggingFace model checkpoint
        """
        self.model_manager = ModelManager(checkpoint)
        self.executor = PythonActionExecutor()
        self.validator = CodeValidator()
        self.logger = logging.getLogger(__name__)
        
    def generate_and_execute(self, prompt: str) -> Dict[str, Any]:
        """
        Generate code from prompt and execute it.
        
        Args:
            prompt: Input prompt for code generation
            
        Returns:
            Dictionary containing results and/or errors
        """
        try:
            # Generate code
            generated_code = self.model_manager.generate_code(prompt)
            
            # Clean and validate code
            cleaned_code = self.validator.clean_code(generated_code)
            
            # Check for function definition
            func_name = self.validator.validate_function_definition(cleaned_code)
            
            # Execute code
            result = self.executor.execute(cleaned_code)
            
            # Store function if valid
            if func_name:
                self.executor.add_action(func_name, cleaned_code)
            
            return {
                "generated_code": generated_code,
                "cleaned_code": cleaned_code,
                "execution_result": result,
                "function_name": func_name
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

# Example usage
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize pipeline
    checkpoint = "HuggingFaceTB/SmolLM2-360M"
    pipeline = CodeGenerationPipeline(checkpoint)
    
    # Example prompt
    prompt = "Create a Python function to read an Excel file and return its contents as a list of dictionaries."
    
    # Generate and execute code
    result = pipeline.generate_and_execute(prompt)
    
    # Print results
    print("Generation Results:")
    print("------------------")
    print("\nGenerated Code:")
    print(result.get("generated_code", "No code generated"))
    print("\nCleaned Code:")
    print(result.get("cleaned_code", "No cleaned code"))
    print("\nExecution Result:")
    print(result.get("execution_result", "No execution result"))
    print("\nStored Function Name:")
    print(result.get("function_name", "No function stored"))

if __name__ == "__main__":
    main()
