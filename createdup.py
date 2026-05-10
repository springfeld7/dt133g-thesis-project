"""
create_synthetic_datasets.py

Creates two synthetic datasets for testing:
- Exact duplicates across datasets
- Near-duplicate pairs across datasets
- Unique samples per dataset
"""

from pathlib import Path
import pandas as pd

from src.transtructiver.parsing.parser import Parser
from src.transtructiver.mutation.rules.comment_deletion import CommentDeletionRule
from src.transtructiver.mutation.mutation_context import MutationContext

# ----------------------------
# OUTPUT CONFIG
# ----------------------------

OUT_DIR = Path("data/_00_test_datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# SAMPLE GENERATORS
# ----------------------------


def make_samples():
    """
    Creates structured synthetic code samples.

    Returns:
        tuple: (dataset_1, dataset_2)
    """

    near_com1 = {
        "code": '"""Utility for validating data integrity."""\nimp poop rexx ort re\n\ndef validate_email(email):\n    """Check if the string matches a standard email pattern."""\n    # Use regex to find a match\n    pattern = r\'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$\'\n    if re.match(pattern, email):\n        return True\n    # If no match, return false\n    return False\n\ndef clean_and_verify(data_list):\n    """Strip whitespace and verify each entry."""\n    valid_entries = []\n    for item in data_list:\n        # Ensure we are dealing with a string\n        cleaned = str(item).strip()\n        if validate_email(cleaned):\n            valid_entries.append(cleaned)\n    return valid_entries',
        "language": "python",
        "label": 0,
    }

    near_com2 = {
        "code": "import re\n\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'\n    if re.match(pattern, email):\n        return True\n    return False\n\ndef clean_and_verify(data_list):\n    valid_entries = []\n    for item in data_list:\n        cleaned = str(item).strip()\n        if validate_email(cleaned):\n            valid_entries.append(cleaned)\n    return valid_entries",
        "language": "python",
        "label": 0,
    }

    # parser = Parser()

    # tree, result = parser.parse(near_com1["code"], "python")

    # for n in tree.traverse():
    #     if n.type == "ERROR":
    #         print(f"Error node found: {n}")

    # if tree is None:
    #     print(f"Parsing failed: {result}")
    #     return

    # comment_rule = CommentDeletionRule(level=3)
    # context = MutationContext()
    # records = comment_rule.apply(tree, context)

    # if not records:
    #     print("No comments found for deletion.")

    # print(tree.to_code())

    # near_com1["code"] = tree.to_code()

    # =========================================================
    # EXACT DUPLICATES (must appear in BOTH datasets)
    # =========================================================
    exact_1 = {"code": "def add(a, b):\n    return a + b", "language": "python", "label": 0}

    exact_2 = {"code": "def multiply(a, b):\n    return a * b", "language": "python", "label": 0}

    exact_3 = {
        "code": 'def calculate_bmi(weight_kg, height_m):\n    """Standard BMI calculation with input validation."""\n    if height_m <= 0:\n        raise ValueError("Height must be greater than zero.")\n    bmi_value = weight_kg / (height_m ** 2)\n    return round(bmi_value, 2)',
        "language": "python",
        "label": 0,
    }

    # =========================================================
    # NEAR DUPLICATE PAIR (semantic duplicates)
    # =========================================================
    near_ds1 = {
        "code": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "language": "python",
        "label": 0,
    }

    near_ds2 = {
        "code": "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
        "language": "python",
        "label": 0,
    }

    nearer = {
        "code": "import math\nimport time\n\nclass ComputationEngine:\n    '''\n    A robust engine for performing heavy mathematical operations and logging.\n\n    Args:\n        engine_id (str): The identifier for this specific instance.\n        precision (int): Decimal places for rounding results.\n\n    Returns:\n        ComputationEngine: A configured engine instance.\n    '''\n    def **init**(self, engine_id, precision):\n        self.engine_id = engine_id\n        self.precision = precision\n        self.history = []\n        self.is_running = False\n\n    def boot_sequence(self):\n        '''\n        Initializes the internal hardware simulation.\n\n        Returns:\n            str: Status message of the boot process.\n        '''\n        self.is_running = True\n        return f'Engine {self.engine_id} is now operational.'\n\n    def compute_power_series(self, base, terms):\n        '''\n        Calculates a power series sum for a given base.\n\n        Args:\n            base (float): The base number to multiply.\n            terms (int): Number of iterations to perform.\n\n        Returns:\n            float: The resulting sum of the power series.\n        '''\n        result = 0.0\n        for i in range(1, terms + 1):\n            val = math.pow(base, i)\n            result += val\n        \n        rounded_res = round(result, self.precision)\n        self.history.append({'op': 'power_series', 'res': rounded_res})\n        return rounded_res\n\n    def validate_dataset(self, data):\n        '''\n        Checks if a dataset contains only numeric values.\n\n        Args:\n            data (list): The list of items to inspect.\n\n        Returns:\n            bool: True if data is clean, False otherwise.\n        '''\n        if not data:\n            return False\n        \n        is_valid = all(isinstance(x, (int, float)) for x in data)\n        self.history.append({'op': 'validation', 'res': is_valid})\n        return is_valid\n\n    def get_execution_log(self):\n        '''\n        Retrieves the history of all operations performed.\n\n        Returns:\n            list: A collection of operation dictionaries.\n        '''\n        return self.history\n\ndef execute_workflow():\n    '''\n    Runs a standard test workflow on the engine.\n\n    Returns:\n        None\n    '''\n    processor = ComputationEngine('Nexus-7', 4)\n    print(processor.boot_sequence())\n    \n    val = processor.compute_power_series(1.5, 10)\n    print(f'Series Result: {val}')\n    \n    dataset = [1.2, 4.5, 9.0, 2.1]\n    if processor.validate_dataset(dataset):\n        print('Data validation passed.')\n    \n    print(f'Log count: {len(processor.get_execution_log())}')\n\nif **name** == '**main**':\n    execute_workflow() import math\nimport time\n\nclass ComputationEngine:\n    '''\n    A robust engine for performing heavy mathematical operations and logging.\n\n    Args:\n        engine_id (str): The identifier for this specific instance.\n        precision (int): Decimal places for rounding results.\n\n    Returns:\n        ComputationEngine: A configured engine instance.\n    '''\n    def **init**(self, engine_id, precision):\n        self.engine_id = engine_id\n        self.precision = precision\n        self.history = []\n        self.is_running = False\n\n    def boot_sequence(self):\n        '''\n        Initializes the internal hardware simulation.\n\n        Returns:\n            str: Status message of the boot process.\n        '''\n        self.is_running = True\n        return f'Engine {self.engine_id} is now operational.'\n\n    def compute_power_series(self, base, terms):\n        '''\n        Calculates a power series sum for a given base.\n\n        Args:\n            base (float): The base number to multiply.\n            terms (int): Number of iterations to perform.\n\n        Returns:\n            float: The resulting sum of the power series.\n        '''\n        result = 0.0\n        for i in range(1, terms + 1):\n            val = math.pow(base, i)\n            result += val\n        \n        rounded_res = round(result, self.precision)\n        self.history.append({'op': 'power_series', 'res': rounded_res})\n        return rounded_res\n\n    def validate_dataset(self, data):\n        '''\n        Checks if a dataset contains only numeric values.\n\n        Args:\n            data (list): The list of items to inspect.\n\n        Returns:\n            bool: True if data is clean, False otherwise.\n        '''\n        if not data:\n            return False\n        \n        is_valid = all(isinstance(x, (int, float)) for x in data)\n        self.history.append({'op': 'validation', 'res': is_valid})\n        return is_valid\n\n    def get_execution_log(self):\n        '''\n        Retrieves the history of all operations performed.\n\n        Returns:\n            list: A collection of operation dictionaries.\n        '''\n        return self.history\n\ndef execute_workflow():\n    '''\n    Runs a standard test workflow on the engine.\n\n    Returns:\n        None\n    '''\n    processor = ComputationEngine('Nexus-7', 4)\n    print(processor.boot_sequence())\n    \n    val = processor.compute_power_series(1.5, 10)\n    print(f'Series Result: {val}')\n    \n    dataset = [1.2, 4.5, 9.0, 2.1]\n    if processor.validate_dataset(dataset):\n        print('Data validation passed.')\n    \n    print(f'Log count: {len(processor.get_execution_log())}')\n\nif **name** == '**main**':\n    execute_workflow() import math\nimport time\n\nclass ComputationEngine:\n    '''\n    A robust engine for performing heavy mathematical operations and logging.\n\n    Args:\n        engine_id (str): The identifier for this specific instance.\n        precision (int): Decimal places for rounding results.\n\n    Returns:\n        ComputationEngine: A configured engine instance.\n    '''\n    def **init**(self, engine_id, precision):\n        self.engine_id = engine_id\n        self.precision = precision\n        self.history = []\n        self.is_running = False\n\n    def boot_sequence(self):\n        '''\n        Initializes the internal hardware simulation.\n\n        Returns:\n            str: Status message of the boot process.\n        '''\n        self.is_running = True\n        return f'Engine {self.engine_id} is now operational.'\n\n    def compute_power_series(self, base, terms):\n        '''\n        Calculates a power series sum for a given base.\n\n        Args:\n            base (float): The base number to multiply.\n            terms (int): Number of iterations to perform.\n\n        Returns:\n            float: The resulting sum of the power series.\n        '''\n        result = 0.0\n        for i in range(1, terms + 1):\n            val = math.pow(base, i)\n            result += val\n        \n        rounded_res = round(result, self.precision)\n        self.history.append({'op': 'power_series', 'res': rounded_res})\n        return rounded_res\n\n    def validate_dataset(self, data):\n        '''\n        Checks if a dataset contains only numeric values.\n\n        Args:\n            data (list): The list of items to inspect.\n\n        Returns:\n            bool: True if data is clean, False otherwise.\n        '''\n        if not data:\n            return False\n        \n        is_valid = all(isinstance(x, (int, float)) for x in data)\n        self.history.append({'op': 'validation', 'res': is_valid})\n        return is_valid\n\n    def get_execution_log(self):\n        '''\n        Retrieves the history of all operations performed.\n\n        Returns:\n            list: A collection of operation dictionaries.\n        '''\n        return self.history\n\ndef execute_workflow():\n    '''\n    Runs a standard test workflow on the engine.\n\n    Returns:\n        None\n    '''\n    processor = ComputationEngine('Nexus-7', 4)\n    print(processor.boot_sequence())\n    \n    val = processor.compute_power_series(1.5, 10)\n    print(f'Series Result: {val}')\n    \n    dataset = [1.2, 4.5, 9.0, 2.1]\n    if processor.validate_dataset(dataset):\n        print('Data validation passed.')\n    \n    print(f'Log count: {len(processor.get_execution_log())}')\n\nif **name** == '**main**':\n    execute_workflow()import math\nimport time\n\nclass ComputationEngine:\n    '''\n    A robust engine for performing heavy mathematical operations and logging.\n\n    Args:\n        engine_id (str): The identifier for this specific instance.\n        precision (int): Decimal places for rounding results.\n\n    Returns:\n        ComputationEngine: A configured engine instance.\n    '''\n    def **init**(self, engine_id, precision):\n        self.engine_id = engine_id\n        self.precision = precision\n        self.history = []\n        self.is_running = False\n\n    def boot_sequence(self):\n        '''\n        Initializes the internal hardware simulation.\n\n        Returns:\n            str: Status message of the boot process.\n        '''\n        self.is_running = True\n        return f'Engine {self.engine_id} is now operational.'\n\n    def compute_power_series(self, base, terms):\n        '''\n        Calculates a power series sum for a given base.\n\n        Args:\n            base (float): The base number to multiply.\n            terms (int): Number of iterations to perform.\n\n        Returns:\n            float: The resulting sum of the power series.\n        '''\n        result = 0.0\n        for i in range(1, terms + 1):\n            val = math.pow(base, i)\n            result += val\n        \n        rounded_res = round(result, self.precision)\n        self.history.append({'op': 'power_series', 'res': rounded_res})\n        return rounded_res\n\n    def validate_dataset(self, data):\n        '''\n        Checks if a dataset contains only numeric values.\n\n        Args:\n            data (list): The list of items to inspect.\n\n        Returns:\n            bool: True if data is clean, False otherwise.\n        '''\n        if not data:\n            return False\n        \n        is_valid = all(isinstance(x, (int, float)) for x in data)\n        self.history.append({'op': 'validation', 'res': is_valid})\n        return is_valid\n\n    def get_execution_log(self):\n        '''\n        Retrieves the history of all operations performed.\n\n        Returns:\n            list: A collection of operation dictionaries.\n        '''\n        return self.history\n\ndef execute_workflow():\n    '''\n    Runs a standard test workflow on the engine.\n\n    Returns:\n        None\n    '''\n    processor = ComputationEngine('Nexus-7', 4)\n    print(processor.boot_sequence())\n    \n    val = processor.compute_power_series(1.5, 10)\n    print(f'Series Result: {val}')\n    \n    dataset = [1.2, 4.5, 9.0, 2.1]\n    if processor.validate_dataset(dataset):\n        print('Data validation passed.')\n    \n    print(f'Log count: {len(processor.get_execution_log())}')\n\nif **name** == '**main**':\n    execute_workflow()",
        "language": "python",
        "label": 1,
    }

    # =========================================================
    # NEAR DUPLICATE PAIR (long semantic duplicates)
    # Forces chunking into multiple windows
    # =========================================================

    near_ds5 = {
        "code": """
    def process_numbers(numbers):
        total = 0

        # normalization stage
        normalized = []
        for n in numbers:
            normalized.append(abs(n))

        # filtering stage
        filtered = []
        for n in normalized:
            if n % 2 == 0:
                filtered.append(n)

        # aggregation stage
        for n in filtered:
            total += n

        # statistics stage
        avg = total / len(filtered) if filtered else 0

        # logging stage
        logs = []
        for n in filtered:
            logs.append(f"processed={n}")

        # export stage
        output = {
            "count": len(filtered),
            "total": total,
            "average": avg,
            "logs": logs,
        }

        return output


    def helper_alpha(data):
        result = []
        for item in data:
            result.append(item * 2)
        return result


    def helper_beta(data):
        result = []
        for item in data:
            result.append(item + 1)
        return result


    def helper_gamma(data):
        result = []
        for item in data:
            result.append(item - 1)
        return result


    def helper_delta(data):
        result = []
        for item in data:
            result.append(item / 2)
        return result


    def helper_epsilon(data):
        result = []
        for item in data:
            result.append(item ** 2)
        return result


    def helper_zeta(data):
        result = []
        for item in data:
            result.append(str(item))
        return result
    """
        * 8,
        "language": "python",
        "label": 0,
    }

    near_ds6 = {
        "code": """
    def analyze_numbers(values):
        summation = 0

        # cleaning stage
        cleaned = []
        for value in values:
            cleaned.append(abs(value))

        # selection stage
        selected = []
        for value in cleaned:
            if value % 2 == 0:
                selected.append(value)

        # accumulation stage
        for value in selected:
            summation += value

        # metric stage
        mean_value = summation / len(selected) if selected else 0

        # tracing stage
        traces = []
        for value in selected:
            traces.append(f"value={value}")

        # packaging stage
        result = {
            "count": len(selected),
            "sum": summation,
            "mean": mean_value,
            "traces": traces,
        }

        return result


    def utility_one(data):
        result = []
        for item in data:
            result.append(item * 2)
        return result


    def utility_two(data):
        result = []
        for item in data:
            result.append(item + 1)
        return result


    def utility_three(data):
        result = []
        for item in data:
            result.append(item - 1)
        return result


    def utility_four(data):
        result = []
        for item in data:
            result.append(item / 2)
        return result


    def utility_five(data):
        result = []
        for item in data:
            result.append(item ** 2)
        return result


    def utility_six(data):
        result = []
        for item in data:
            result.append(str(item))
        return result
    """
        * 8,
        "language": "python",
        "label": 0,
    }
    # =========================================================
    # UNIQUE SAMPLES (no overlap)
    # =========================================================
    unique_ds1_1 = {
        "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
        "language": "python",
        "label": 0,
    }

    unique_ds1_2 = {
        "code": "def reverse_string(s):\n    return s[::-1]",
        "language": "python",
        "label": 0,
    }

    unique_ds2_1 = {
        "code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        "language": "python",
        "label": 0,
    }

    unique_ds2_2 = {
        "code": "def factorial(n):\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res",
        "language": "python",
        "label": 0,
    }

    # =========================================================
    # NEAR DUPLICATES (Identical logic, but variable names differ)
    # =========================================================
    near_ds3 = {
        "code": 'def find_max_value(numbers_list):\n    """Find the highest number in a given list."""\n    highest = numbers_list[0]\n    for num in numbers_list:\n        if num > highest:\n            highest = num\n    return highest',
        "language": "python",
        "label": 0,
    }

    # Logic is identical to near_ds1, but all local identifiers are renamed
    near_ds4 = {
        "code": 'def get_top_element(data_sequence):\n    """Find the highest number in a given list."""\n    current_max = data_sequence[0]\n    for item in data_sequence:\n        if item > current_max:\n            current_max = item\n    return current_max',
        "language": "python",
        "label": 0,
    }

    near_ds7 = {
        "code": """
    import json
    import math

    def load_data(path):
        with open(path, 'r') as f:
            return json.load(f)

    def normalize(values):
        total = sum(values)
        if total == 0:
            return values
        return [v / total for v in values]

    def transform_record(record):
        result = {}
        for k, v in record.items():
            if isinstance(v, (int, float)):
                result[k] = math.sqrt(v) * 1.5 + 3
            else:
                result[k] = str(v).strip().lower()
        return result

    def process_dataset(data):
        processed = []
        for record in data:
            cleaned = transform_record(record)
            processed.append(cleaned)
        return processed

    def aggregate(data):
        totals = {}
        for record in data:
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0) + v
        return totals

    def pipeline(path):
        data = load_data(path)
        data = process_dataset(data)
        totals = aggregate(data)
        return normalize(list(totals.values()))

    # expand workload to ensure long sequence
    for i in range(100):
        dummy = {"a": i, "b": i * 2, "c": i * i}
        dummy = transform_record(dummy)
        dummy = process_dataset([dummy])
        _ = aggregate(dummy)

    print("ETL pipeline complete")
    """
        * 3,  # repetition ensures >512 tokens
        "language": "python",
        "label": 0,
    }

    near_ds8 = {
        "code": """
    import json
    import math

    class DataLoader:
        def __init__(self, path):
            self.path = path

        def read(self):
            with open(self.path, 'r') as f:
                return json.load(f)

    class Transformer:
        def scale(self, x):
            return math.sqrt(x) * 1.5 + 3

        def clean(self, record):
            cleaned = {}
            for key in record:
                value = record[key]
                if isinstance(value, (int, float)):
                    cleaned[key] = self.scale(value)
                else:
                    cleaned[key] = str(value).strip().lower()
            return cleaned

    class Aggregator:
        def sum_values(self, dataset):
            result = {}
            for item in dataset:
                for k, v in item.items():
                    if isinstance(v, (int, float)):
                        if k not in result:
                            result[k] = 0
                        result[k] += v
            return result

    class Pipeline:
        def __init__(self, path):
            self.loader = DataLoader(path)
            self.transformer = Transformer()
            self.aggregator = Aggregator()

        def run(self):
            data = self.loader.read()
            processed = []

            for record in data:
                processed.append(self.transformer.clean(record))

            aggregated = self.aggregator.sum_values(processed)

            total = sum(aggregated.values())
            if total == 0:
                return aggregated

            return {k: v / total for k, v in aggregated.items()}

    # heavy workload expansion
    pipeline = Pipeline("dummy.json")

    for i in range(100):
        fake_record = {"x": i, "y": i * 3, "z": i ** 2}
        cleaned = pipeline.transformer.clean(fake_record)
        processed = [cleaned for _ in range(5)]
        aggregated = pipeline.aggregator.sum_values(processed)

        _ = sum(aggregated.values())

    print("OOP pipeline complete")
    """
        * 3,
        "language": "python",
        "label": 0,
    }

    near1 = {
        "code": "import math\nimport json\n\ndef calculate_metrics(data_list):\n    \"\"\"Process standard metrics.\"\"\"\n    results = []\n    for item in data_list:\n        val = item.get('value', 0)\n        if val > 0:\n            transformed = math.sqrt(val) * 2.1\n            results.append(transformed)\n        else:\n            results.append(0.0)\n    return results\n\ndef aggregate_batch(batches):\n    total = 0.0\n    for b in batches:\n        metrics = calculate_metrics(b)\n        total += sum(metrics)\n    return total\n\ndef run_pipeline(raw_json):\n    data = json.loads(raw_json)\n    if not data:\n        return 0.0\n    return aggregate_batch(data)\n\n# Simulated Workload\nfor i in range(50):\n    payload = json.dumps([{'value': x} for x in range(i)])\n    print(f'Batch {i}: {run_pipeline(payload)}')",
        "language": "python",
        "label": 0,
    }

    near2 = {
        "code": "import math\nimport json\ndef calculate_metrics(data_list):\n \"\"\"Process standard metrics.\"\"\"\n results=[]\n for item in data_list:\n  val=item.get('value',0)\n  if val>0:\n   transformed=math.sqrt(val)*2.1\n   results.append(transformed)\n  else:\n   results.append(0.0)\n return results\ndef aggregate_batch(batches):\n total=0.0\n for b in batches:\n  metrics=calculate_metrics(b)\n  total+=sum(metrics)\n return total\ndef run_pipeline(raw_json):\n data=json.loads(raw_json)\n if not data:\n  return 0.0\n return aggregate_batch(data)\nfor i in range(50):\n payload=json.dumps([{'value':x} for x in range(i)])\n print(f'Batch {i}: {run_pipeline(payload)}')",
        "language": "python",
        "label": 0,
    }

    near3 = {
        "code": """
    import json
    import math

    def load_data(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def normalize(values):
        total_sum = sum(values)
        if total_sum == 0:
            return values
        return [val / total_sum for val in values]

    def transform_record(record):
        output = {}
        for key, val in record.items():
            if isinstance(val, (int, float)):
                # Apply standard scaling transformation
                output[key] = math.sqrt(val) * 1.5 + 3
            else:
                output[key] = str(val).strip().lower()
        return output

    def process_dataset(data):
        results = []
        for entry in data:
            cleaned_entry = transform_record(entry)
            results.append(cleaned_entry)
        return results

    def aggregate(data):
        stats = {}
        for item in data:
            for k, v in item.items():
                if isinstance(v, (int, float)):
                    stats[k] = stats.get(k, 0) + v
        return stats

    def pipeline(path):
        raw_data = load_data(path)
        clean_data = process_dataset(raw_data)
        summary = aggregate(clean_data)
        return normalize(list(summary.values()))

    for i in range(100):
        test_dict = {"a": i, "b": i * 2, "c": i * i}
        test_dict = transform_record(test_dict)
        test_dict = process_dataset([test_dict])
        _ = aggregate(test_dict)

    print("ETL pipeline complete")
    """
        * 3,
        "language": "python",
        "label": 0,
    }

    near4 = {
        "code": """
import json, math
def load_data(p):
 with open(p,'r') as f:return json.load(f)
def normalize(v):
 s=sum(v)
 if s==0:return v
 return [x/s for x in v]
def transform_record(r):
 d={}
 for k,v in r.items():
  if isinstance(v,(int,float)):d[k]=math.sqrt(v)*1.5+3
  else:d[k]=str(v).strip().lower()
 return d
def process_dataset(ds):
 return [transform_record(x) for x in ds]
def aggregate(ds):
 res={}
 for r in ds:
  for k,v in r.items():
   if isinstance(v,(int,float)):res[k]=res.get(k,0)+v
 return res
def pipeline(p):
 d=process_dataset(load_data(p))
 a=aggregate(d)
 return normalize(list(a.values()))
for i in range(100):
 t={"a":i,"b":i*2,"c":i*i}
 t=transform_record(t)
 t=process_dataset([t])
 _=aggregate(t)
print("ETL pipeline complete")
"""
        * 5,  # Increased repetition to match token count due to compression
        "language": "python",
        "label": 0,
    }

    near_ds9 = {
        "code": """
    from collections import deque, defaultdict

    class NetworkExplorer:
        def __init__(self, size):
            self.size = size
            self.links = defaultdict(list)

        def connect(self, node_a, node_b):
            self.links[node_a].append(node_b)
            self.links[node_b].append(node_a)

        def find_route(self, source, destination):
            visited_nodes = set()
            parent_map = {}
            queue = deque([source])
            visited_nodes.add(source)

            while queue:
                current_node = queue.popleft()

                if current_node == destination:
                    break

                for neighbor_node in self.links[current_node]:
                    if neighbor_node not in visited_nodes:
                        visited_nodes.add(neighbor_node)
                        parent_map[neighbor_node] = current_node
                        queue.append(neighbor_node)

            reconstructed_path = []
            step = destination

            while step in parent_map:
                reconstructed_path.append(step)
                step = parent_map[step]

            reconstructed_path.append(source)
            reconstructed_path.reverse()

            return reconstructed_path
    """,
        "language": "python",
        "label": 0,
    }

    near_ds10 = {
        "code": """
    import heapq
    from collections import defaultdict

    class PathSolver:
        def __init__(self):
            self.graph_map = defaultdict(list)

        def add_connection(self, start_node, end_node, cost=1):
            self.graph_map[start_node].append((end_node, cost))
            self.graph_map[end_node].append((start_node, cost))

        def compute_shortest_path(self, origin, target):
            priority_queue = [(0, origin)]
            best_distance = {origin: 0}
            previous_node = {}

            while priority_queue:
                current_cost, current_node = heapq.heappop(priority_queue)

                if current_node == target:
                    break

                for neighbor, weight in self.graph_map[current_node]:
                    new_cost = current_cost + weight

                    if neighbor not in best_distance or new_cost < best_distance[neighbor]:
                        best_distance[neighbor] = new_cost
                        previous_node[neighbor] = current_node
                        heapq.heappush(priority_queue, (new_cost, neighbor))

            path = []
            cursor = target

            while cursor in previous_node:
                path.append(cursor)
                cursor = previous_node[cursor]

            path.append(origin)
            path.reverse()

            return path
    """,
        "language": "python",
        "label": 0,
    }

    # =========================================================
    # BUILD DATASETS
    # =========================================================

    dataset_1 = [
        # exact_1,          # shared identical
        # exact_2,          # shared identical
        # exact_3,          # shared identical
        # near_ds1,         # near duplicate (ds1 side)
        # near_ds3,  # near duplicate (ds3 side)
        # unique_ds1_1,
        # unique_ds1_2
        # near3
        near_com2,
        exact_1,
    ]

    dataset_2 = [
        # exact_1,  # shared identical
        # exact_2,          # shared identical
        # exact_3,          # shared identical
        # near_ds2,         # near duplicate (ds2 side)
        # near_ds4,  # near duplicate (ds4 side)
        # unique_ds2_1,
        # unique_ds2_2
        # near_ds10
        # near4
        # near_com1
        nearer,
        nearer,
        exact_1,
        nearer,
    ]

    return dataset_1, dataset_2


# ----------------------------
# SAVE DATASETS
# ----------------------------


def save_datasets(ds1, ds2):
    """
    Saves datasets as parquet files.

    Args:
        ds1 (list[dict]): First dataset
        ds2 (list[dict]): Second dataset
    """

    df1 = pd.DataFrame(ds1)
    df2 = pd.DataFrame(ds2)

    path1 = OUT_DIR / "dataset_1.parquet"
    path2 = OUT_DIR / "dataset_2.parquet"

    df1.to_parquet(path1, index=False)
    df2.to_parquet(path2, index=False)

    print(f"Saved dataset 1 → {path1}")
    print(f"Saved dataset 2 → {path2}")


# ----------------------------
# MAIN
# ----------------------------


def main():
    """
    Entry point for dataset generation.
    """
    ds1, ds2 = make_samples()
    save_datasets(ds1, ds2)
    print("Synthetic datasets created successfully.")


if __name__ == "__main__":
    main()
