import argparse
import base64
import json
import os.path
import shutil
import subprocess
import traceback
from tempfile import TemporaryDirectory

import hay_say_common as hsc
import jsonschema
import soundfile
from flask import Flask, request
from hay_say_common.cache import Stage
from jsonschema import ValidationError

# todo: write output file to the temporary directory, not to the architecture root, or add a uuid prefix and search for
#  that file specifically when cleaning up. Otherwise, users might delete each other's outputs if the timing is just
#  wrong. This will require modification to svc_inference.py

app = Flask(__name__)


class Svc5Pipeline:
    """Versions 1 and 2 of so-vits-svc follow a similar pipeline, but their executables are in different directories
    and version 2 additionally uses a HuBERT model. These differences are abstracted away into class
    attributes upon object instantiation so that both versions can use the exact same methods."""

    DIRECTORY_KEY = 'directory'
    EXECUTE_HUBERT_KEY = 'execute_hubert'
    VERSION_CONSTANTS = {
        1: {
            DIRECTORY_KEY: 'so_vits_svc_5_v1',
            EXECUTE_HUBERT_KEY: False
        },
        2: {
            DIRECTORY_KEY: 'so_vits_svc_5_v2',
            EXECUTE_HUBERT_KEY: True
        },
    }

    ARCHITECTURE_NAME = 'so_vits_svc_5'
    DETERMINATOR_PATH = os.path.join(os.path.dirname(__file__), 'version_determinator.py')
    PTH_FILENAME = 'sovits5.0.pth'
    PIT_OUTPUT_FILENAME = 'svc_tmp.pit.csv'
    PPG_OUTPUT_FILENAME = 'svc_tmp.ppg.npy'
    VEC_OUTPUT_FILENAME = 'svc_tmp.vec.npy'
    TEMP_FILE_EXTENSION = '.flac'
    VENV_BIN = os.path.join(hsc.ROOT_DIR, '.venvs', 'so_vits_svc_5', 'bin')
    PYTHON_EXECUTABLE = os.path.join(VENV_BIN, 'python')

    def __init__(self, version_number):
        self.version_number = version_number
        if version_number not in self.VERSION_CONSTANTS.keys():
            raise Exception("Unknown so-vits-svc 5 version: " + version_number)
        self.DIR_NAME = self.VERSION_CONSTANTS[version_number][self.DIRECTORY_KEY]
        self.EXECUTE_HUBERT = self.VERSION_CONSTANTS[version_number][self.EXECUTE_HUBERT_KEY]
        self.ARCHITECTURE_ROOT = os.path.join(hsc.ROOT_DIR, self.DIR_NAME)
        self.OUTPUT_PATH = os.path.join(self.ARCHITECTURE_ROOT, 'svc_out.wav')
        self.BASE_CONFIGURATION_FILE = os.path.join(self.ARCHITECTURE_ROOT, 'configs', 'base.yaml')
        self.SVC_EXPORT_SCRIPT_PATH = os.path.join(self.ARCHITECTURE_ROOT, 'svc_export.py')
        self.CONTENT_VECTOR_EXTRACTION_SCRIPT_PATH = os.path.join(self.ARCHITECTURE_ROOT, 'whisper', 'inference.py')
        self.PITCH_EXTRACTION_SCRIPT = os.path.join(self.ARCHITECTURE_ROOT, 'pitch', 'inference.py')
        self.HIDDEN_UNIT_EXTRACTION_SCRIPT_PATH = os.path.join(self.ARCHITECTURE_ROOT, 'hubert', 'inference.py')
        self.INFERENCE_SCRIPT_PATH = os.path.join(self.ARCHITECTURE_ROOT, 'svc_inference.py')

    @classmethod
    def create_from_character(cls, character):
        version_number = cls.determine_version(character)
        return Svc5Pipeline(version_number)


    @classmethod
    def determine_version(cls, character):
        checkpoint_path = cls.get_checkpoint_path(character)
        version_string = subprocess.check_output([cls.PYTHON_EXECUTABLE, cls.DETERMINATOR_PATH, checkpoint_path])
        return int(version_string.decode('utf-8'))

    def execute_pipeline(self, input_filename_sans_extension, character, pitch_shift, output_filename_sans_extension,
                         gpu_id, session_id):
        with TemporaryDirectory() as temp_dir:
            self.copy_input_audio(input_filename_sans_extension, session_id, temp_dir)
            self.execute_program(input_filename_sans_extension, character, pitch_shift, gpu_id, temp_dir)
            self.copy_output(output_filename_sans_extension, session_id, temp_dir)

    def copy_input_audio(self, input_filename_sans_extension, session_id, temp_dir):
        data, sr = cache.read_audio_from_cache(Stage.PREPROCESSED, session_id, input_filename_sans_extension)
        target = os.path.join(temp_dir, input_filename_sans_extension + self.TEMP_FILE_EXTENSION)
        try:
            soundfile.write(target, data, sr)
        except Exception as e:
            raise Exception("Unable to copy file from Hay Say's audio cache to so-vits-svc 5's input directory.") from e

    def execute_program(self, input_filename_sans_extension, character, pitch_shift, gpu_id, temp_dir):
        self.export_pth_file_if_needed(character, gpu_id)
        self.extract_content_vector(input_filename_sans_extension, gpu_id, temp_dir)
        self.extract_f0_data(input_filename_sans_extension, gpu_id, temp_dir)
        # todo: There's probably a better way to write this without using an if-statement. Consider "composing" a
        #  pipline by specifying a list of methods for each version or something and then just loop through the list,
        #  executing each method.
        if self.EXECUTE_HUBERT:
            self.extract_hidden_units(input_filename_sans_extension, gpu_id, temp_dir)
        self.infer(character, input_filename_sans_extension, pitch_shift, gpu_id, temp_dir)

    def export_pth_file_if_needed(self, character, gpu_id):
        character_dir = hsc.character_dir(self.ARCHITECTURE_NAME, character)
        desired_pth_path = os.path.join(character_dir, self.PTH_FILENAME)

        if not os.path.isfile(desired_pth_path):
            arguments = [
                '--config', self.get_config_path(character),
                '--checkpoint_path', self.get_checkpoint_path(character),
            ]
            env = hsc.select_hardware(gpu_id)
            subprocess.run([self.PYTHON_EXECUTABLE, self.SVC_EXPORT_SCRIPT_PATH, *arguments],
                           env=env,
                           cwd=self.ARCHITECTURE_ROOT)
            current_pth_path = os.path.join(self.ARCHITECTURE_ROOT, self.PTH_FILENAME)
            shutil.copyfile(current_pth_path, desired_pth_path)

    def get_config_path(self, character):
        character_dir = hsc.character_dir(self.ARCHITECTURE_NAME, character)
        config_filename = self.get_config_filename(character_dir)
        return os.path.join(character_dir, config_filename)

    def get_config_filename(self, character_dir):
        potential_names = [file for file in os.listdir(character_dir) if file.endswith('.yaml')]
        if len(potential_names) == 0:
            # Use the configuration file that comes included with so-vits-svc 5.0
            return self.BASE_CONFIGURATION_FILE
        if len(potential_names) > 1:
            raise Exception('Too many config files found! Expected only one file with extension ".yaml" in '
                            + character_dir)
        else:
            # Use the configuration file that came with the model
            return potential_names[0]

    @classmethod
    def get_checkpoint_path(cls, character):
        character_dir = hsc.character_dir(cls.ARCHITECTURE_NAME, character)
        checkpoint_filename = cls.get_checkpoint_filename(character_dir)
        return os.path.join(character_dir, checkpoint_filename)

    @classmethod
    def get_checkpoint_filename(cls, character_dir):
        potential_names = [file for file in os.listdir(character_dir) if file.endswith('.pt')]
        if len(potential_names) == 0:
            raise Exception('Checkpoint file was not found! Expected a file with extension ".pt" in ' + character_dir)
        if len(potential_names) > 1:
            raise Exception('Too many checkpoint files found! Expected only one file with extension ".pt" in '
                            + character_dir)
        else:
            return potential_names[0]

    def extract_content_vector(self, input_filename_sans_extension, gpu_id, temp_dir):
        arguments = [
            '--wav', os.path.join(temp_dir, input_filename_sans_extension + self.TEMP_FILE_EXTENSION),
            '--ppg', os.path.join(temp_dir, self.PPG_OUTPUT_FILENAME)
        ]
        env = hsc.select_hardware(gpu_id)
        env = self.define_python_path(env)
        subprocess.run([self.PYTHON_EXECUTABLE, self.CONTENT_VECTOR_EXTRACTION_SCRIPT_PATH, *arguments],
                       env=env,
                       cwd=self.ARCHITECTURE_ROOT)

    def define_python_path(self, env):
        # so-vits-svc 5 v1 needs "PYTHONPATH" to be defined.
        # This is not required for v2, but including it does not cause any issues.
        env['PYTHONPATH'] = self.ARCHITECTURE_ROOT
        return env

    def extract_f0_data(self, input_filename_sans_extension, gpu_id, temp_dir):
        arguments = [
            '--wav', os.path.join(temp_dir, input_filename_sans_extension + self.TEMP_FILE_EXTENSION),
            '--pit', os.path.join(temp_dir, self.PIT_OUTPUT_FILENAME)
        ]
        env = hsc.select_hardware(gpu_id)
        subprocess.run([self.PYTHON_EXECUTABLE, self.PITCH_EXTRACTION_SCRIPT, *arguments],
                       env=env,
                       cwd=self.ARCHITECTURE_ROOT)

    def extract_hidden_units(self, input_filename_sans_extension, gpu_id, temp_dir):
        arguments = [
            '--wav', os.path.join(temp_dir, input_filename_sans_extension + self.TEMP_FILE_EXTENSION),
            '--vec', os.path.join(temp_dir, self.VEC_OUTPUT_FILENAME)
        ]
        env = hsc.select_hardware(gpu_id)
        subprocess.run([self.PYTHON_EXECUTABLE, self.HIDDEN_UNIT_EXTRACTION_SCRIPT_PATH, *arguments],
                       env=env,
                       cwd=self.ARCHITECTURE_ROOT)

    def infer(self, character, input_filename_sans_extension, pitch_shift, gpu_id, temp_dir):
        arguments = [
            '--config', self.get_config_path(character),
            '--model', self.get_pth_path(character),
            '--spk', self.get_spk_path(character),
            '--wave', os.path.join(temp_dir, input_filename_sans_extension + self.TEMP_FILE_EXTENSION),
            '--ppg', os.path.join(temp_dir, self.PPG_OUTPUT_FILENAME),
            '--pit', os.path.join(temp_dir, self.PIT_OUTPUT_FILENAME),
            '--shift', str(pitch_shift),
            *(['--vec', os.path.join(temp_dir, self.VEC_OUTPUT_FILENAME)] if self.version_number == 2 else [None, None]),
        ]
        arguments = [argument for argument in arguments if argument]  # Removes all "None" objects in the list.
        env = hsc.select_hardware(gpu_id)
        subprocess.run([self.PYTHON_EXECUTABLE, self.INFERENCE_SCRIPT_PATH, *arguments],
                       env=env,
                       cwd=self.ARCHITECTURE_ROOT)

    def get_pth_path(self, character):
        character_dir = hsc.character_dir(self.ARCHITECTURE_NAME, character)
        return os.path.join(character_dir, self.PTH_FILENAME)

    def get_spk_path(self, character):
        character_dir = hsc.character_dir(self.ARCHITECTURE_NAME, character)
        singer_dir = os.path.join(character_dir, 'singer')
        spk_filename = self.get_spk_filename(singer_dir)
        return os.path.join(singer_dir, spk_filename)

    def get_spk_filename(self, singer_dir):
        potential_names = [file for file in os.listdir(singer_dir) if file.endswith('.spk.npy')]
        if len(potential_names) == 0:
            raise Exception('Speaker file was not found! Expected a file with extension ".spk.npy" in ' + singer_dir)
        if len(potential_names) > 1:
            raise Exception('Too many speaker files found! Expected only one file with extension ".spk.npy" in '
                            + singer_dir)
        else:
            return potential_names[0]

    def copy_output(self, output_filename_sans_extension, session_id, temp_dir):
        array_output, sr_output = hsc.read_audio(self.OUTPUT_PATH)
        cache.save_audio_to_cache(Stage.OUTPUT, session_id, output_filename_sans_extension, array_output, sr_output)


def register_methods(cache):
    @app.route('/generate', methods=['POST'])
    def generate() -> (str, int):
        code = 200
        message = ""
        try:
            # todo: I really want to just do this: inputs = parse_inputs() and then pass *inputs to the pipeline. I
            #  still need a reference to just the character, though, to determine the version of so-vits-svc 5 to use.
            #  Maybe I can just pass *inputs to the factory method (create_from_character)? Or just make inputs a
            #  dictionary.
            input_filename_sans_extension, character, pitch_shift, output_filename_sans_extension, gpu_id, session_id \
                = parse_inputs()
            svc5Pipeline = Svc5Pipeline.create_from_character(character)
            svc5Pipeline.execute_pipeline(input_filename_sans_extension, character, pitch_shift,
                                          output_filename_sans_extension, gpu_id, session_id)
            hsc.clean_up(get_temp_files())
        except BadInputException:
            code = 400
            message = traceback.format_exc()
        except Exception:
            code = 500
            message = hsc.construct_full_error_message('', get_temp_files())

        # The message may contain quotes and curly brackets which break JSON syntax, so base64-encode the message.
        message = base64.b64encode(bytes(message, 'utf-8')).decode('utf-8')
        response = {
            "message": message
        }

        return json.dumps(response, sort_keys=True, indent=4), code

    @app.route('/gpu-info', methods=['GET'])
    def get_gpu_info():
        return hsc.get_gpu_info_from_another_venv(Svc5Pipeline.PYTHON_EXECUTABLE)

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

    def get_temp_files():
        files_to_clean = []
        for version_number in Svc5Pipeline.VERSION_CONSTANTS.keys():
            svc5_pipeline = Svc5Pipeline(version_number)
            files_to_clean = files_to_clean + \
                [svc5_pipeline.OUTPUT_PATH]
        return [file for file in files_to_clean if os.path.isfile(file)]  # Removes nonexistent files


def parse_arguments():
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='A webservice interface for voice conversion with so-vits-svc 5.0')
    parser.add_argument('--cache_implementation', default='file', choices=hsc.cache_implementation_map.keys(),
                        help='Selects an implementation for the audio cache, e.g. saving them to files or to a database.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cache = hsc.select_cache_implementation(args.cache_implementation)
    register_methods(cache)
    app.run(host='0.0.0.0', port=6577)
