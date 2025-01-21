def read_system_prompt(file_name:str) -> str:
    """
    Reads the system prompt from a file
    """
    print(f"reading system prompt from file {file_name}")
    with open(f"{file_name}", "r") as f:
        system_prompt = f.read()
    
    print(f"system_prompt: {system_prompt}")
    return system_prompt