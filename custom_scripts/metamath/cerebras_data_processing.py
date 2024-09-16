from dataclasses import dataclass

from distilabel.distiset import Distiset


@dataclass
class HdF5ProcessingSFT:
    input_path: str
    output_dir: str
    tokenizer_name: str
    n_processes: int = 1
    max_seq_length: int = 1024

    @property
    def yaml_config(self):
        _yaml_config: dict = {
            'setup': {
                'data': {
                    'source': self.input_path,
                    'type': 'local'
                },
                'output_dir': self.output_dir,
                'processes': self.n_processes,
                'mode': 'finetuning'
            },
            'processing': {
                'huggingface_tokenizer': self.tokenizer_name,
                'max_seq_length': self.max_seq_length,
                'short_seq_prob': 0.0,
                'write_in_batch': True,
                'resume_from_checkpoint': False,
                'seed': 0,
                'read_hook': 'cerebras.modelzoo.data_preparation.data_preprocessing.hooks:chat_read_hook',
                'read_hook_kwargs': {
                    'data_keys': {
                        'multi_turn_key': 'messages',
                        'multi_turn_content_key': 'content'
                    }
                }
            },
            'dataset': {
                'use_ftfy': True,
                'ftfy_normalizer': 'NFC',
                'wikitext_detokenize': False
            }
        }
        return _yaml_config

    def process(self, dataset: Distiset):
        from pathlib import Path

        def generate_hdf5():
            import os
            import subprocess
            import tempfile

            import yaml

            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                yaml.dump(self.yaml_config, temp_file, default_flow_style=False)
                temp_file_name = temp_file.name

            try:
                # Set up environment variables
                os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':/cb/home/alexv/ws/monolith/cerebras/models/src/cerebras'
                os.environ['GITTOP'] = '/net/alexv-dev/srv/nfs/alexv-data/ws/monolith/'

                # Enter monolith environment and run the command
                command = f"""
                source /net/alexv-dev/srv/nfs/alexv-data/ws/monolith/flow/devenv.sh && devenv_enter && \
                /cb/home/alexv/ws/monolith/python/python-x86_64/bin/python -m modelzoo.data_preparation.data_preprocessing.preprocess_data --config {temp_file_name}
                """

                # Run the command and stream output in real-time
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

                # Print output in real-time
                for line in process.stdout:
                    print(line, end='')  # end='' to avoid double line breaks

                # Wait for the process to complete and get the return code
                return_code = process.wait()

                if return_code != 0:
                    print(f"Command failed with return code {return_code}")

            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")
            finally:
                os.unlink(temp_file_name)

        def write_json(inputs, data_path):
            import json

            with open(data_path, 'a') as f:
                for input in inputs:
                    f.write(
                        json.dumps(
                            {"messages": [{"role": "user", "content": input['question']}, {"role": "assistant", "content": input['answer']}]}
                        ) + '\n'
                    )

        def generate_jsonl():
            # write each input to json file
            for dataset_name in dataset.keys():
                write_json(dataset[dataset_name]['train'], data_path)

        # Create data.jsonl file
        data_path = Path(self.input_path) / "data.jsonl"
        # clean the file
        open(data_path, 'w').close()

        generate_jsonl()
        generate_hdf5()
