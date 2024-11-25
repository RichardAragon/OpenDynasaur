# OpenDynasaur
An Open Source version Dynasaur

A powerful machine learning framework for generating and executing code using large language models. Open Dynasaur provides a robust pipeline for code generation, validation, and safe execution in Python environments.

## Features

- **Intelligent Code Generation**: Leverages state-of-the-art language models for generating Python code
- **Code Validation**: Advanced syntax checking and code cleaning capabilities
- **Safe Execution Environment**: Secure code execution with comprehensive error handling
- **Action Management**: Store and retrieve generated functions for reuse
- **Comprehensive Logging**: Detailed logging system for debugging and monitoring
- **Type Safety**: Full type hints throughout the codebase
- **Modular Architecture**: Well-organized, maintainable code structure

## Installation

```bash
pip install transformers accelerate openpyxl ipython torch
```

## Quick Start

```python
from open_dynasaur import CodeGenerationPipeline

# Initialize the pipeline
pipeline = CodeGenerationPipeline("HuggingFaceTB/SmolLM2-135M")

# Generate and execute code
result = pipeline.generate_and_execute(
    "Create a function to calculate the factorial of a number"
)

# Access results
print(result["cleaned_code"])
print(result["execution_result"])
```

## Core Components

### CodeValidator
Handles code cleaning and validation:
- Removes non-code narrative
- Validates syntax
- Identifies function definitions

### ModelManager
Manages the machine learning model:
- Handles model loading and device management
- Configurable generation parameters
- Efficient tokenization

### PythonActionExecutor
Executes and manages generated code:
- Safe code execution
- Action library management
- Comprehensive error handling

### CodeGenerationPipeline
Orchestrates the entire process:
- Streamlined code generation workflow
- Result aggregation
- Error management

## Advanced Usage

### Custom Model Configuration

```python
pipeline = CodeGenerationPipeline(
    checkpoint="HuggingFaceTB/SmolLM2-135M",
    max_length=512,
    temperature=0.8
)
```

### Error Handling

```python
try:
    result = pipeline.generate_and_execute(prompt)
    if "error" in result:
        print(f"Error occurred: {result['error']}")
        print(f"Traceback: {result['traceback']}")
except Exception as e:
    print(f"Pipeline error: {str(e)}")
```

### Storing and Retrieving Actions

```python
# Generate and store a function
result = pipeline.generate_and_execute(prompt)
func_name = result["function_name"]

# Retrieve the stored function
stored_func = pipeline.executor.retrieve_action(func_name)
```

## Configuration Options

- `max_length`: Maximum length of generated code (default: 256)
- `temperature`: Sampling temperature for generation (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.95)
- `top_k`: Top-k sampling parameter (default: 50)

## Logging

The framework uses Python's built-in logging module:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace team for their transformers library
- PyTorch team for the underlying ML framework
- The open source community for their valuable contributions

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Roadmap

- [ ] Add support for more model architectures
- [ ] Implement additional code validation rules
- [ ] Create web interface
- [ ] Add unit test suite
- [ ] Improve documentation

## Citation

If you use Open Dynasaur in your research, please cite:

```bibtex
@software{open_dynasaur,
  title = {Open Dynasaur: A Framework for ML-Powered Code Generation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/username/open-dynasaur}
}
```
