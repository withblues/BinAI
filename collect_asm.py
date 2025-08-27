import argparse
import json
import subprocess
from datasets import Dataset
import os

def data_generator(data_dir, pattern):
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.startswith(pattern):
                source_file = os.path.join(dirpath, filename)
                source_file = source_file

                # retrieve project
                project = os.path.basename(dirpath)
                
                # retrieve function name
                split = filename.split("_")
                binary_name = split[-1]

                # retrieve compilation data
                compilation_split = split[0].split('-')
                architecture = compilation_split[0]
                compiler = compilation_split[1]
                compiler_version = compilation_split[2]
                compiler_optimization = compilation_split[3]

                yield {
                    "file_path": source_file,
                    "project": project,
                    "binary_name": binary_name,
                    "architecture": architecture,
                    "compiler": compiler,
                    "compiler_version": compiler_version,
                    "compiler_optimization": compiler_optimization,
                }

def disassemble_binary(example, ida_path, ida_script):
    binary_path = os.path.abspath(example['file_path'])
    command = [ida_path, '-c', '-A', f'-S{ida_script}', binary_path]

    try:
        subprocess.run(command, check=True)

        json_path = binary_path + '.json'
        i64_path = binary_path + '.i64'
        with open(json_path, 'r') as f:
            assembly_dict = json.load(f)


        example['functions'] = json.dumps(assembly_dict)

        # clean up possible files
        if os.path.isfile(json_path):
            os.remove(json_path)
        if os.path.isfile(i64_path):
            os.remove(i64_path)

        

    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Failed to process {binary_path}: {e}")
        example['functions'] = {}
    
    
    return example

def flatten_dataset(dataset):
    for row in dataset:
        functions_dict = json.loads(row['functions'])
        for func_name, func_dict in functions_dict.items():
            if not func_dict:
                continue

            keys = list(func_dict.keys())
            instructions = list(func_dict.values())
                
            yield {
                "file_path": row['file_path'],
                "project": row['project'],
                "binary_name": row['binary_name'],
                "architecture": row['architecture'],
                "compiler": row['compiler'],
                "compiler_version": row['compiler_version'],
                "compiler_optimization": row['compiler_optimization'],
                "function_name": func_name,
                "keys": keys,
                "instructions": instructions
            }



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ida_path", required=True)
    parser.add_argument("--ida_script", required=True)

    args = parser.parse_args()

    pattern = "x64"
    
    dataset = Dataset.from_generator(
        data_generator,
        gen_kwargs={
            'data_dir': args.data_dir,
            'pattern': pattern
        }
    )

    ida_path = args.ida_path
    ida_script = os.path.abspath(args.ida_script)

    dataset = dataset.map(
        disassemble_binary,
        fn_kwargs={
            'ida_path': ida_path,
            'ida_script': ida_script,
        },
        num_proc=128,
        writer_batch_size=5
    )
    print(f'done mapping data\nstarting to unflat')
    flat_dataset = Dataset.from_generator(lambda: flatten_dataset(iter(dataset)))
    print(f'unflatten data done\nsaving to disk')
    flat_dataset.save_to_disk(args.output_dir)
