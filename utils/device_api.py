# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import json
import os
import re
import logging
from io import BytesIO
import subprocess
from subprocess import CompletedProcess
import platform
import time
from typing import (
    Optional,
    Union
)
import uuid
from PIL import Image
from hmdriver2.driver import Driver

from utils import (
    decode_command_output,
    encode_image,
    print_out,
    read_json
)

logging.getLogger("hmdriver2").setLevel(logging.WARNING)
logging.getLogger("hmdriver2.hdc").setLevel(logging.WARNING)
logging.getLogger("hmdriver2._client").setLevel(logging.WARNING)


class Operate(object):
    def __init__(self,
                 bundle_name_dict: dict,
                 hdc_command: str = "hdc.exe",
                 factor: float = 0.5) -> None:
        self.bundle_name_dict = bundle_name_dict
        self.hdc_command = hdc_command
        self.factor = factor
        self.device_id = self._get_device_id()
        self.driver = Driver(serial=self.device_id)
    
    @staticmethod
    def _get_ability_name(all_package_info: dict) -> str:
        """
        提取主入口 ability 名称。
        """
        main_ability_list = list()
        for _ in all_package_info.get('hapModuleInfos'):
            main_ability = _.get('mainAbility')
            for info in _.get('abilityInfos'):
                ability_name = info.get('name')
                if not main_ability or main_ability in ability_name:
                    main_ability = ability_name
                    break
            if main_ability:
                main_ability_list.append(main_ability)
        if 'EntryAbility' in main_ability_list:
            return 'EntryAbility'
        for main_ability in main_ability_list:
            if 'mainAbility' in main_ability:
                return main_ability
            if 'MainAbility' in main_ability:
                return main_ability
        return main_ability_list[0] if main_ability_list else ''
    
    def _get_device_id(self) -> str:
        devices = self.get_connected_devices()
        device_id = ""
        length = len(devices)

        if length > 1:
            print_out(
                "please to choose which devices id:",
                stdout=True
            )
            for idx, sub_device in enumerate(devices):
                print_out(
                    f"{idx + 1}. {sub_device}",
                    stdout=True
                )
            while True:
                choice = input("your choice (input index): ")
                print_out(f"your choice (input index): {choice}")
                if not choice.isdigit():
                    print_out(
                        "Invalid input, please enter a number.",
                        stdout=True
                    )
                    continue
                cur_idx = int(choice) - 1
                if 0 <= cur_idx < length:
                    device_id = devices[cur_idx]
                    print_out(
                        f"Thank you for choice the device id: {device_id}",
                        stdout=True
                    )
                    break
                else:
                    print_out(
                        "Choice out of range, please try again.",
                        stdout=True
                    )
        elif length == 1:
            device_id = devices[0]
            print_out(
                f"current device id: {device_id}",
                stdout=True
            )
        else:
            print_out(
                "No devices found. Exiting.",
                stdout=True,
                log_level="error"
            )
        return device_id

    def _get_command_list(self,
                          command: Union[str, list[str]],
                          device_id: Optional[str] = None) -> list:
        if isinstance(command, str):
            command_list = command.split(' ')
        else:
            command_list = list(command)
        # hdc -> root_dir/hdc/hdc.exe
        command_list[0] = os.path.join(os.environ["ROOT_DIR"], self.hdc_command)
        if device_id:
            command_list.insert(1, '-t')
            command_list.insert(2, device_id)
        return command_list

    def run_hdc_command(self,
                        command: Union[str, list[str]],
                        device_id: Optional[str] = None) -> CompletedProcess:
        kwargs = {}
        platform_name = platform.system().lower()
        command_text = command if isinstance(command, str) else ' '.join(command)
        if platform_name == 'windows':
            kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
        raw_result = subprocess.run(
            self._get_command_list(command, device_id),
            capture_output=True,
            text=False,
            check=False,
            **kwargs
        )
        result = CompletedProcess(
            raw_result.args,
            raw_result.returncode,
            decode_command_output(raw_result.stdout),
            decode_command_output(raw_result.stderr)
        )
        if result.returncode != 0:
            error_detail = result.stderr.strip() or result.stdout.strip()
            print_out(
                f'run command `{command_text}` return {result.returncode}: {error_detail}',
                log_level="error"
            )
            raise Exception(f'run command `{command_text}` return {result.returncode}: {error_detail}')
        if result.stdout and result.stdout.lstrip().startswith('[Fail]'):
            print_out(
                f'run command `{command_text}` error: {result.stdout}',
                log_level="error"
            )
            raise Exception(f'run command `{command_text}` error: {result.stdout}')
        return result

    @staticmethod
    def _extract_dump_layout_path(command_output: str) -> str:
        match = re.search(r'saved to:(.+)', command_output)
        if not match:
            raise ValueError('missing dumpLayout output path')
        return match.group(1).strip()

    @staticmethod
    def _move_recv_file(temp_local_path: str,
                        final_local_path: str) -> str:
        final_dir = os.path.dirname(final_local_path)
        if final_dir:
            os.makedirs(final_dir, exist_ok=True)
        os.replace(temp_local_path, final_local_path)
        return final_local_path

    def _recv_file_to_local(self,
                            device_path: str,
                            final_local_path: str,
                            temp_prefix: str) -> str:
        device_path = device_path.strip()
        platform_name = platform.system().lower()

        if platform_name == 'windows':
            os.makedirs(os.environ['TEMP_DIR'], exist_ok=True)
            suffix = os.path.splitext(final_local_path)[1] or ".tmp"
            temp_local_path = os.path.join(
                os.environ['TEMP_DIR'],
                f'{temp_prefix}_{uuid.uuid4().hex}{suffix}'
            )
            self.run_hdc_command(
                ['hdc', 'file', 'recv', device_path, temp_local_path],
                self.device_id
            )
            return self._move_recv_file(temp_local_path, final_local_path)

        final_dir = os.path.dirname(final_local_path)
        if final_dir:
            os.makedirs(final_dir, exist_ok=True)
        self.run_hdc_command(
            ['hdc', 'file', 'recv', device_path, final_local_path],
            self.device_id
        )
        return final_local_path

    def get_connected_devices(self) -> list:
        try:
            result = self.run_hdc_command('hdc list targets')
            devices = []
            for line in result.stdout.splitlines():
                if line.strip() and 'Empty' not in line:
                    devices.append(line.strip())
            return devices
        except Exception as e:
            print_out(
                f'Error getting devices: {e}',
                log_level="error"
            )
            return []

    def get_foreground_app(self) -> str:
        result = self.run_hdc_command('hdc shell aa dump -a', self.device_id)

        pattern = r'app state #FOREGROUND'
        lines = result.stdout.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if 'ExtensionRecords:' in line:
                print_out(
                    f'ExtensionRecords found in aa dump output, skipping parsing',
                    log_level="warning"
                )
                return ''

            if re.search(pattern, line):
                bundle_name = None

                j = i
                while j >= 0 and not (re.search(r'bundle name', lines[j]) or re.search(r'app name', lines[j])):
                    j -= 1

                if j >= 0:
                    for k in range(j, i + 1):
                        l = lines[k].strip()
                        if re.search(r'bundle name', l):
                            bundle_name = re.search(r'\[(.*?)\]', l).group(1)

                if bundle_name:
                    return bundle_name
            i += 1

        return ''

    def get_installed_apps(self) -> list:
        try:
            result = self.run_hdc_command(f'hdc shell bm dump -a', self.device_id)
            apps = []
            for line in result.stdout.splitlines():
                if line.startswith('ID:'):
                    continue
                package_name = line.strip()
                apps.append(package_name)
            return apps
        except Exception as e:
            print_out(
                f"Error getting apps: {e}",
                log_level="error"
            )
            return []

    def get_package_info(self,
                         package_name: str) -> dict:
        app_name = self.bundle_name_dict.get(package_name)
        if not app_name:
            print_out(
                f'Package name {package_name} not found in white list',
                log_level="error"
            )
            return {}

        result = self.run_hdc_command(f'hdc shell bm dump -n "{package_name}"', self.device_id)
        matches = re.findall(f'{package_name}:' + r'([\s\S]*)', result.stdout)
        all_package_info = json.loads(matches[0])

        application_info = all_package_info.get('applicationInfo')
        package_info = {
            'appName': app_name,
            'packageName': package_name,
            'appVersion': application_info.get('versionName'),
            'isSystemApp': application_info.get('isSystemApp'),
            'mainAbility': self._get_ability_name(all_package_info),
        }
        return package_info

    def start_app(self,
                  package_name: str,
                  ability_name: Optional[str] = None, 
                  restart: bool = True) -> None:
        if ability_name is None:
            package_info = self.get_package_info(package_name)
            ability_name = package_info.get('mainAbility')
            if not package_info or not ability_name:
                print_out(
                    f'Failed to start app {package_name}',
                    log_level="error"
                )
                return

        if restart:
            self.run_hdc_command(f'hdc shell aa force-stop "{package_name}"', self.device_id)
            time.sleep(0.1)
        self.run_hdc_command(f'hdc shell aa start -a "{ability_name}" -b "{package_name}"', self.device_id)
        time.sleep(0.1)

    def get_screenshot_data(self) -> tuple[str, str]:
        uid = uuid.uuid4().hex
        screenshot_path = '/data/local/tmp/' + uid + '.jpeg'
        if not os.path.exists(os.environ['TEMP_DIR']):
            os.makedirs(os.environ['TEMP_DIR'])
        local_screenshot_path = os.path.join(os.environ['TEMP_DIR'], uid + '.jpeg')
        self.run_hdc_command(f'hdc shell snapshot_display -f {screenshot_path}', self.device_id)
        self._recv_file_to_local(screenshot_path, local_screenshot_path, 'screenshot')
        self.run_hdc_command(f'hdc shell rm {screenshot_path}', self.device_id)

        with Image.open(local_screenshot_path) as img:
            fmt = img.format.upper()
            img = img.resize((int(img.width * self.factor), int(img.height * self.factor)), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            encoded_image = encode_image(byte_stream=buffer.getvalue())
        os.remove(local_screenshot_path)
        return encoded_image, fmt

    def perform_back(self) -> None:
        self.run_hdc_command(f'hdc shell uinput -K -d 2 -u 2', self.device_id)
    
    def perform_home(self) -> None:
        self.run_hdc_command(f'hdc shell uinput -K -d 1 -u 1', self.device_id)

    def perform_click(self,
                      x: Union[int, str], 
                      y: Union[int, str]) -> None:
        self.run_hdc_command(f'hdc shell uinput -T -c {x} {y}', self.device_id)

    def perform_longclick(self,
                          x: Union[int, str], 
                          y: Union[int, str]) -> None:
        self.run_hdc_command(f'hdc shell uinput -T -d {x} {y} -i 1000 -u {x} {y}', self.device_id)

    def perform_scroll(self,
                       x1: Union[int, str], 
                       y1: Union[int, str], 
                       x2: Union[int, str], 
                       y2: Union[int, str]) -> None:
        self.run_hdc_command(f'hdc shell uinput -T -m {x1} {y1} {x2} {y2} 500', self.device_id)

    def perform_settext(self,
                        text: str, 
                        enter: bool = False) -> None:
        self.driver.input_text(text)
        # self.run_hdc_command(f'hdc shell uitest uiInput inputText 1 1 "{text}"', self.device_id)
        if enter:
            self.run_hdc_command(f'hdc shell uinput -K -d 2054 -u 2054', self.device_id)

    @staticmethod
    def _extract_snapshot_file_path(command_output: str) -> str:
        match = re.search(r'set filename to\s+(\S+)', command_output)
        return match.group(1) if match else ''

    @staticmethod
    def _parse_snapshot_display_size(command_output: str) -> tuple[int, int]:
        matches = list(re.finditer(r'width:\s*(\d+)\s*,\s*height:\s*(\d+)', command_output))
        if not matches:
            raise ValueError('missing width/height fields')
        match = matches[-1]
        return int(match.group(1)), int(match.group(2))

    def get_screen_scale(self) -> tuple[int, int]:
        command = 'hdc shell snapshot_display /data/local/tmp/'
        snapshot_path = ''
        result = self.run_hdc_command(command, self.device_id)
        command_output = '\n'.join(
            part for part in (result.stdout or '', result.stderr or '') if part
        )
        snapshot_path = self._extract_snapshot_file_path(command_output)

        try:
            return self._parse_snapshot_display_size(command_output)
        except ValueError as e:
            output_excerpt = command_output.strip() or '<empty output>'
            print_out(
                f'Failed to parse screen scale from snapshot_display output: {output_excerpt}',
                log_level='error'
            )
            raise Exception(
                f'Failed to parse screen scale from snapshot_display output: {output_excerpt}'
            ) from e
        finally:
            if snapshot_path:
                try:
                    self.run_hdc_command(f'hdc shell rm {snapshot_path}', self.device_id)
                except Exception as cleanup_error:
                    print_out(
                        f'Failed to remove snapshot_display temp file `{snapshot_path}`: {cleanup_error}',
                        log_level='warning'
                    )

    def dump_ui_tree(self,
                     dump_times: int,
                     is_temp: bool) -> Union[dict, list]:
        try:
            result = self.run_hdc_command(f"hdc shell uitest dumpLayout", self.device_id)
            res = result.stdout
            device_full_path = self._extract_dump_layout_path(res)
        except Exception as e:
            print_out(f"UI automator output vide: {e}", log_level="error")
            return

        if not is_temp:
            local_tree_dir = os.path.join(os.environ['DATA_DIR'], "JsonInfo")
            local_tree_dir = os.path.join(local_tree_dir, f'frame_{dump_times}')
            os.makedirs(local_tree_dir, exist_ok=True)
            local_tree_path = os.path.join(local_tree_dir, f'tree_origin.json')
        else:
            os.makedirs(os.environ['TEMP_DIR'], exist_ok=True)
            local_tree_path = os.path.join(os.environ['TEMP_DIR'], f'tree_origin.json')
        
        self._recv_file_to_local(device_full_path, local_tree_path, 'tree_origin')
        self.run_hdc_command(f"hdc shell rm {device_full_path}", self.device_id)

        ui_tree = read_json(local_tree_path)
        if is_temp:
            os.remove(local_tree_path)
        
        return ui_tree
    
    def get_all_background_app(self) -> list[str]:
        background_activity = []
        result = self.run_hdc_command(f"hdc shell aa dump -l", self.device_id)
        proc_txt = result.stdout or ""

        # 一次性匹配同一个 AbilityRecord 内的 app name / main name / app state
        pattern = re.compile(
            r"AbilityRecord[\s\S]*?"
            r"app name \[(?P<app>[^\]]+)\][\s\S]*?"
            r"main name \[(?P<main>[^\]]+)\][\s\S]*?"
            r"app state #(?P<state>[A-Z_]+)",
            re.MULTILINE
        )

        for m in pattern.finditer(proc_txt):
            state = m.group("state").strip().upper()
            if state == "BACKGROUND":
                background_activity.append(f"{m.group('app')};{m.group('main')}")

        return list(dict.fromkeys(background_activity))
    
    def kill_all_app_process(self) -> None:
        foreground_app_name = self.get_foreground_app()
        self.run_hdc_command(f"hdc shell bm clean -n {foreground_app_name} -d", self.device_id)
        print_out(f"current app {foreground_app_name} foreground is killed", stdout=True)

        background_activity = self.get_all_background_app()
        for current_activity in background_activity:
            app_name = current_activity.split(";")[0]
            self.run_hdc_command(f"hdc shell bm clean -n {app_name} -d", self.device_id)
            print_out(f"current app {app_name} background is killed", stdout=True)
