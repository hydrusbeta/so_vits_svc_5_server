import argparse
import base64
import json
import os.path
import shutil
import subprocess
import traceback

import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common import *
from hay_say_common.cache import Stage
from jsonschema import ValidationError

ARCHITECTURE_NAME = 'so_vits_svc_5'
ARCHITECTURE_ROOT = os.path.join(ROOT_DIR, ARCHITECTURE_NAME)
INPUT_COPY_FOLDER = os.path.join(ARCHITECTURE_ROOT, 'input')
OUTPUT_PATH = os.path.join(ARCHITECTURE_ROOT, 'svc_out.wav')
PPG_OUTPUT_PATH = os.path.join(ARCHITECTURE_ROOT, 'ppg', 'svc_tmp.ppg.npy')
PIT_OUTPUT_PATH = os.path.join(ARCHITECTURE_ROOT, 'pit', 'svc_tmp.pit.csv')
PTH_FILENAME = 'sovits5.0.pth'
TEMP_FILE_EXTENSION = '.flac'

PYTHON_EXECUTABLE = os.path.join(ROOT_DIR, '.venvs', 'so_vits_svc_5', 'bin', 'python')
SVC_EXPORT_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, 'svc_export.py')
CONTENT_VECTOR_EXTRACTION_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, 'whisper', 'inference.py')
PITCH_EXTRACTION_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, 'pitch', 'inference.py')
INFERENCE_SCRIPT_PATH = os.path.join(ARCHITECTURE_ROOT, 'svc_inference.py')

app = Flask(__name__)


def register_methods(cache):
    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            input_filename_sans_extension, character, pitch_shift, output_filename_sans_extension, gpu_id, session_id \
                = parse_inputs()
            copy_input_audio(input_filename_sans_extension, session_id)
            execute_program(input_filename_sans_extension, character, pitch_shift, gpu_id)
            copy_output(output_filename_sans_extension, session_id)
            clean_up(get_temp_files())
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = construct_full_error_message(ARCHITECTURE_ROOT, get_temp_files())

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    def parse_inputs():
        schema = {
            'type': 'object',
            'properties': {
                'Inputs': {
                    'type': 'object',
                    'properties': {
                        'User Audio': {'type': 'string'}
                    }
                },
                'Options': {
                    'type': 'object',
                    'properties': {
                        'Pitch Shift': {'type': 'integer'},
                        'Character': {'type': 'string'}
                    }
                },
                'Output File': {'type': 'string'},
                'GPU ID': {'type': ['string', 'integer']},
                'Session ID': {'type': ['string', 'null']}
            }
        }

        try:
            jsonschema.validate(instance=request.json, schema=schema)
        except ValidationError as e:
            raise BadInputException(e.message)

        input_filename_sans_extension = request.json['Inputs']['User Audio']
        character = request.json['Options']['Character']
        pitch_shift = int(request.json['Options']['Pitch Shift'])
        output_filename_sans_extension = request.json['Output File']
        gpu_id = request.json['GPU ID']
        session_id = request.json['Session ID']

        return input_filename_sans_extension, character, pitch_shift, output_filename_sans_extension, gpu_id, session_id

    class BadInputException(Exception):
        pass

    def get_checkpoint_path(character):
        character_dir = get_model_path(ARCHITECTURE_NAME, character)
        checkpoint_filename = get_checkpoint_filename(character_dir)
        return os.path.join(character_dir, checkpoint_filename)

    def get_config_path(character):
        character_dir = get_model_path(ARCHITECTURE_NAME, character)
        config_filename = get_config_filename(character_dir)
        return os.path.join(character_dir, config_filename)

    def get_spk_path(character):
        character_dir = get_model_path(ARCHITECTURE_NAME, character)
        singer_dir = os.path.join(character_dir, 'singer')
        spk_filename = get_spk_filename(singer_dir)
        return os.path.join(singer_dir, spk_filename)

    def get_pth_path(character):
        character_dir = get_model_path(ARCHITECTURE_NAME, character)
        return os.path.join(character_dir, PTH_FILENAME)

    def get_config_filename(character_dir):
        potential_names = [file for file in os.listdir(character_dir) if file.endswith('.yaml')]
        if len(potential_names) == 0:
            # Use the configuration file that comes included with so-vits-svc 5.0
            return '/root/hay_say/so_vits_svc_5/configs/base.yaml'
        if len(potential_names) > 1:
            raise Exception('Too many config files found! Expected only one file with extension ".yaml" in '
                            + character_dir)
        else:
            # Use the configuration file that came with the model
            return potential_names[0]

    def get_checkpoint_filename(character_dir):
        potential_names = [file for file in os.listdir(character_dir) if file.endswith('.pt')]
        if len(potential_names) == 0:
            raise Exception('Checkpoint file was not found! Expected a file with extension ".pt" in ' + character_dir)
        if len(potential_names) > 1:
            raise Exception('Too many checkpoint files found! Expected only one file with extension ".pt" in '
                            + character_dir)
        else:
            return potential_names[0]

    def get_spk_filename(singer_dir):
        potential_names = [file for file in os.listdir(singer_dir) if file.endswith('.spk.npy')]
        if len(potential_names) == 0:
            raise Exception('Speaker file was not found! Expected a file with extension ".spk.npy" in ' + singer_dir)
        if len(potential_names) > 1:
            raise Exception('Too many speaker files found! Expected only one file with extension ".spk.npy" in '
                            + singer_dir)
        else:
            return potential_names[0]

    def copy_input_audio(input_filename_sans_extension, session_id):
        data, sr = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_filename_sans_extension)
        target = os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + TEMP_FILE_EXTENSION)
        try:
            soundfile.write(target, data, sr)
        except Exception as e:
            raise Exception("Unable to copy file from Hay Say's audio cache to so-vits-svc 5's input directory.") from e

    def execute_program(input_filename_sans_extension, character, pitch_shift, gpu_id):
        export_pth_file_if_needed(character, gpu_id)
        extract_content_vector(input_filename_sans_extension, gpu_id)
        extract_f0_data(input_filename_sans_extension, gpu_id)
        infer(character, input_filename_sans_extension, pitch_shift, gpu_id)

    def export_pth_file_if_needed(character, gpu_id):
        character_dir = get_model_path(ARCHITECTURE_NAME, character)
        desired_pth_path = os.path.join(character_dir, PTH_FILENAME)

        if not os.path.isfile(desired_pth_path):
            arguments = [
                '--config', get_config_path(character),
                '--checkpoint_path', get_checkpoint_path(character),
            ]
            env = select_hardware(gpu_id)
            subprocess.run([PYTHON_EXECUTABLE, SVC_EXPORT_SCRIPT_PATH, *arguments], env=env)
            current_pth_path = os.path.join(ARCHITECTURE_ROOT, PTH_FILENAME)
            shutil.copyfile(current_pth_path, desired_pth_path)

    def extract_content_vector(input_filename_sans_extension, gpu_id):
        arguments = [
            '--wav', os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + TEMP_FILE_EXTENSION),
            '--ppg', PPG_OUTPUT_PATH,
        ]
        env = select_hardware(gpu_id)
        env['PYTHONPATH'] = ARCHITECTURE_ROOT
        subprocess.run([PYTHON_EXECUTABLE, CONTENT_VECTOR_EXTRACTION_SCRIPT_PATH, *arguments], env=env)

    def extract_f0_data(input_filename_sans_extension, gpu_id):
        arguments = [
            '--wav', os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + TEMP_FILE_EXTENSION),
            '--pit', PIT_OUTPUT_PATH,
        ]
        env = select_hardware(gpu_id)
        subprocess.run([PYTHON_EXECUTABLE, PITCH_EXTRACTION_SCRIPT_PATH, *arguments], env=env)

    def infer(character, input_filename_sans_extension, pitch_shift, gpu_id):
        arguments = [
            '--config', get_config_path(character),
            '--model', get_pth_path(character),
            '--spk', get_spk_path(character),
            '--wave', os.path.join(INPUT_COPY_FOLDER, input_filename_sans_extension + TEMP_FILE_EXTENSION),
            '--ppg', PPG_OUTPUT_PATH,
            '--pit', PIT_OUTPUT_PATH,
            '--shift', str(pitch_shift)
        ]
        env = select_hardware(gpu_id)
        subprocess.run([PYTHON_EXECUTABLE, INFERENCE_SCRIPT_PATH, *arguments], env=env)

    def copy_output(output_filename_sans_extension, session_id):
        array_output, sr_output = read_audio(OUTPUT_PATH)
        cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array_output, sr_output)

    def get_temp_files():
        files_to_clean = [os.path.join(INPUT_COPY_FOLDER, file) for file in os.listdir(INPUT_COPY_FOLDER)] + \
                         [OUTPUT_PATH] + \
                         [PPG_OUTPUT_PATH] + \
                         [PIT_OUTPUT_PATH]
        return [file for file in files_to_clean if os.path.isfile(file)]  # Removes nonexistent files


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='A webservice interface for voice conversion with so-vits-svc 5.0')
    parser.add_argument('--cache_implementation', default='file', choices=cache_implementation_map.keys(),
                        help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = select_cache_implementation(args.cache_implementation)
    register_methods(cache)
    app.run(host='0.0.0.0', port=6577)
