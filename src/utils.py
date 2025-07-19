import pickle
import re
import csv

direct_tokens = ["[PAD]", ".", "*", "-", "+", "cos(x)", "sin(x)", "cos(y)", "sin(y)", "cos(z)", "sin(z)", "cos(2*x)", "cos(2*y)", "cos(2*z)", "sin(2*x)", "sin(2*y)", "sin(2*z)", "cos(x)*cos(y)", "cos(x)*sin(y)", "cos(x)*cos(z)", "cos(x)*sin(z)", "cos(y)*cos(z)", "cos(y)*sin(z)", "sin(x)*cos(y)", "sin(x)*sin(y)", "sin(y)*sin(z)", "sin(y)*cos(z)", "sin(x)*cos(z)", "sin(x)*sin(z)", "cos(x)*cos(y)*cos(z)","sin(x)*sin(y)*sin(z)", "cos(x)*cos(x)", "cos(y)*cos(y)", "cos(z)*cos(z)", "sin(x)*sin(x)", "sin(y)*sin(y)", "sin(z)*sin(z)", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
vocab = {token: idx for idx, token in enumerate(direct_tokens)}

first_order_term = ['cos(x)', 'cos(y)', 'cos(z)', 'sin(x)', 'sin(y)', 'sin(z)', 'cos(2*x)', 'cos(2*y)', 'cos(2*z)', 'sin(2*x)', 'sin(2*y)', 'sin(2*z)']
second_order_term = ['cos(x)*cos(y)', 'cos(x)*sin(y)', 'cos(x)*cos(z)', 'cos(x)*sin(z)', 'cos(y)*cos(z)', 'cos(y)*sin(z)', 'sin(x)*cos(y)', 'sin(x)*sin(y)', 'sin(y)*sin(z)', 'sin(y)*cos(z)', 'sin(x)*cos(z)', 'sin(x)*sin(z)',\
                    'cos(x)*cos(x)', 'cos(y)*cos(y)', 'cos(z)*cos(z)', 'sin(x)*sin(x)', 'sin(y)*sin(y)', 'sin(z)*sin(z)']
third_order_term = ['cos(x)*cos(y)*cos(z)','sin(x)*sin(y)*sin(z)']

reverse_vocab = {value: key for key, value in vocab.items()}

# Get the path relative to the project root
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vocab_path = os.path.join(project_root, "data", "eq_vocab.pickle")

# Create the vocab file if it doesn't exist
os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
if not os.path.exists(vocab_path):
    with open(vocab_path, "wb") as file:
        pickle.dump(vocab, file)

def reorder_equations(sequence):
    """
    Reorders the terms in the sequence so that first-order terms
    (e.g., sin(z), cos(x)) appear before higher-order terms.
    """
    import re

    def get_term_order(trig_expr):
        if trig_expr in first_order_term:
            return 1, first_order_term.index(trig_expr)
        elif trig_expr in second_order_term:
            return 2, second_order_term.index(trig_expr)
        elif trig_expr in third_order_term:
            return 3, third_order_term.index(trig_expr)
        else:
            return 4, float('inf')  # Unknown terms go last

    def extract_order(term):
        # Count occurrences of sin or cos in the term
        return term.count('sin') + term.count('cos')

    sequence = sequence.replace(' ', '') 

    if sequence[0] != '-':
        sequence = '+' + sequence
    const = sequence[-4:]
    
    # sequence = "[CLS]" + sequence

    term_pattern = re.compile(
        r"([\+\-]?\d+(\.\d+)?)[\*]?\s*((?:cos|sin)\([^\)]+\)(?:\*(?:cos|sin)\([^\)]+\))*)"
    )

    # Extract terms from the input sequence
    terms = term_pattern.findall(sequence)
    terms = [(match[0], match[2]) for match in terms]

    sorted_terms = sorted(
        terms,
        key=lambda term: get_term_order(term[1])
    )

    # Reconstruct the reordered sequence
    reordered_sequence = ""
    for coeff, trig_expr in sorted_terms:
        sign = "+" if float(coeff) >= 0 else ""
        reordered_sequence += f"{sign}{coeff}*{trig_expr}"

    reordered_sequence = reordered_sequence.replace('++', '+')
    reordered_sequence = reordered_sequence + const

    return reordered_sequence

class EquationTokenizer:
    def __init__(self, vocab_path=None):
        # Use the global vocab_path if none provided
        if vocab_path is None:
            vocab_path = globals()['vocab_path']
        # Load vocabulary
        with open(vocab_path, 'rb') as file:
            self.vocab = pickle.load(file)
        self.reverse_vocab = {value: key for key, value in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.max_num_coefs = 5
        self.pad_token = "[PAD]"
        self.pad_token_id = self.vocab[self.pad_token]

    def tokenize_equation(self, sequence):
        
        if 'cos' in sequence or 'sin' in sequence:
            sequence = reorder_equations(sequence)
        
        sequence = sequence.replace(',', '[SEP]')

        # if sequence[0] != '-':
        #     sequence = '+' + sequence

        token_regex = r"(\[PAD\]|\[SEP\]|\*|\+|\-|\d*\.\d+|sin\(x\)\*sin\(y\)\*sin\(z\)|sin\(x\)\*sin\(y\)\*cos\(z\)|sin\(x\)\*cos\(y\)\*sin\(z\)|sin\(x\)\*cos\(y\)\*cos\(z\)|cos\(x\)\*sin\(y\)\*sin\(z\)|cos\(x\)\*sin\(y\)\*cos\(z\)|cos\(x\)\*cos\(y\)\*sin\(z\)|cos\(x\)\*cos\(y\)\*cos\(z\)|cos\(x\)\*cos\(x\)|cos\(y\)\*cos\(y\)|cos\(z\)\*cos\(z\)|sin\(x\)\*sin\(x\)|sin\(y\)\*sin\(y\)|sin\(z\)\*sin\(z\)|sin\(x\)\*sin\(y\)|sin\(x\)\*cos\(y\)|sin\(x\)\*cos\(z\)|sin\(x\)\*sin\(z\)|cos\(x\)\*cos\(y\)|cos\(x\)\*sin\(y\)|cos\(y\)\*sin\(z\)|cos\(y\)\*cos\(z\)|sin\(y\)\*sin\(z\)|sin\(y\)\*cos\(z\)|cos\(x\)\*cos\(z\)|cos\(x\)\*sin\(z\)|sin\(2\*x\)|sin\(2\*y\)|sin\(2\*z\)|cos\(2\*x\)|cos\(2\*y\)|cos\(2\*z\)|cos\(x\)|sin\(x\)|cos\(y\)|sin\(y\)|cos\(z\)|sin\(z\))"
    
        # Find all tokens in the sequence that match the regex
        tokens = re.findall(token_regex, sequence)
        new_tokens = []
        for token in tokens:
            if re.match(r'\d*\.\d+', token):
                integer_part, decimal_part = token.split('.')
                new_tokens.extend(list(integer_part) + ['.'] + list(decimal_part))
            elif token not in ['(', ')']:
                new_tokens.append(token)
        token_ids = [vocab.get(token) for token in new_tokens]

        return token_ids
    
    def encode(self, equations):
        equations_ids = [self.tokenize_equation(equation.replace(' ', '')) for equation in equations]
        pad_token = '[PAD]'
        max_len = max(len(equation) for equation in equations_ids)
        for equation in equations_ids:
            equation += [self.vocab[pad_token]] * (max_len - len(equation))
        # attention_masks = [[1 if token != self.vocab[pad_token] else 0 for token in equation] for equation in equations_ids]
        return equations_ids
    
    def decode(self, indices):
        tokens = [self.reverse_vocab.get(index, '[UNK]') for index in indices]
        sequence = ''.join(tokens).strip()
        equation = sequence.replace('[PAD]', '').replace('[SEP]', ',')
        return sequence, equation


def convert_equation(equation):
    # Step 1: Replace all k1, k2, k3 with 1
    # k_value = str(2*math.pi/20)
    # k2_value = str(4*math.pi/20)
    # equation = equation.replace("2*k1", k2_value).replace("2*k2", k2_value).replace("2*k3", k2_value)
    k_value = str(1)
    equation = equation.replace("k1", k_value).replace("k2", k_value).replace("k3", k_value)
    
    # Helper function to extract multiplier for a specific variable

    def get_multiplier(term, var):
        pattern = r'(?:(\d+\.\d+|\d+)\*)?' + var  # Match optional number (with or without decimal) followed by the variable
        match = re.search(pattern, term)
        if match:
            return match.group(1) if match.group(1) else '1'
        return '0'


    # Helper function to convert sin and cos terms
    def format_terms(match):
        trig = match.group(1)
        term = match.group(2)
        x_factor = get_multiplier(term, 'x')
        y_factor = get_multiplier(term, 'y')
        z_factor = get_multiplier(term, 'z')
        return f'{trig}({x_factor}*x+{y_factor}*y+{z_factor}*z)'
    
    # Step 2: Format sin and cos terms
    equation = equation.replace('2*1*', "2*")
    equation = re.sub(r'(sin|cos)\(([^)]+)\)', format_terms, equation)
    
    # Step 3: Replace all ** with *
    equation = equation.replace('**', '*')
    equation = equation.replace('cos', 'COS')
    equation = equation.replace('sin', 'SIN')
    equation = equation.replace('.*', '*')
    equation = re.sub(r'(\d+\.\d+)(?=\*?(sin|cos))', r'(\1)', equation)
    
    pattern = r'(?:(\d+\.\d+|\d+)\*)?(SIN|COS)\('
    
    def repl(match):
        coefficient = match.group(1)
        trig_fn = match.group(2)
        
        # If coefficient is found, enclose it in parenthesis, otherwise use 1
        coef_str = '(' + coefficient + ')' if coefficient else '1'
        return coef_str + '*' + trig_fn + '('
    
    # Replace using the defined regex and replacement function
    equation = re.sub(pattern, repl, equation, flags=re.IGNORECASE)
    equation = equation.replace('1*', '')

    return equation

def read_and_transform(input_filename, output_filename):
    # List to store transformed equations
    transformed_equations = []
    
    # Read from the CSV file
    with open(input_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row == []:
                pass
            else:
                # Assuming the equation is the first item in each row
                eq = row[0]
                transformed_eq = convert_equation(eq)
                transformed_equations.append([transformed_eq])
    
    # Write the transformed equations to a new CSV file
    with open(output_filename, 'w', newline='') as csvfile:
        csvfile.write('equation' + '\n')
        writer = csv.writer(csvfile)
        writer.writerows(transformed_equations)


tokenizer = EquationTokenizer()

# Example usage of EquationTokenizer
if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = EquationTokenizer()
    
    # Example equations to tokenize
    equations = [
        '-4.2*cos(x)*cos(y)+2.0*sin(z)*sin(z)+0.2*cos(x)*cos(y)*cos(z)+4.5*sin(x)*sin(z)+1.0',
        '-5.1*cos(x)*sin(z)-1.4*cos(y)-1.5*cos(2*z)+0.5',
        '-0.1*cos(2*z)+3.3*cos(z)*sin(y)'
    ]
    
    print("Original equations:")
    for i, eq in enumerate(equations):
        print(f"  {i+1}: {eq}")
    
    # Encode equations to token IDs
    input_ids = tokenizer.encode(equations)
    print(f"\nEncoded token IDs:")
    for i, ids in enumerate(input_ids):
        print(f"  Equation {i+1}: {ids}")
    
    # Decode back to equations
    print(f"\nDecoded equations:")
    for i, ids in enumerate(input_ids):
        sequence, equation = tokenizer.decode(ids)
        print(f"  Equation {i+1}: {equation}")
    
    # Example of reading and transforming equations from CSV
    print(f"\nExample CSV transformation:")
    print("Use read_and_transform('input.csv', 'output.csv') to process equation files")
    
    # Example of reordering equations
    print(f"\nExample equation reordering:")
    sample_eq = "3.5*sin(x)*cos(y)-1.4*sin(x)-3.1*cos(y)*sin(z)+3.5*cos(x)*cos(z)+1.3"
    reordered = reorder_equations(sample_eq)
    print(f"  Original: {sample_eq}")
    print(f"  Reordered: {reordered}")

