import re

def extract_special_tokens(file_content):
    # Define regular expressions to match token names and their values
    token_pattern = re.compile(r'(\w+):\s*\'([^\']*)\'')
    tokens = {}

    # Search for all matches in the file content
    matches = token_pattern.findall(file_content)

    # Store the matches in the tokens dictionary
    for match in matches:
        token_name, token_value = match
        tokens[token_name] = token_value

    return tokens

if __name__ == "__main__":
    # Read the content of the Solidity lexer grammar file
    with open('SolidityLexer.g4', 'r', encoding='utf-8') as f:
        file_content = f.read()

    special_tokens = extract_special_tokens(file_content)
    
    # Print the extracted special tokens
    for token_name, token_value in special_tokens.items():
        print(f"{token_name}: '{token_value}'")
