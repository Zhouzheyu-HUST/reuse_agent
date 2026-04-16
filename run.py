# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import os
import platform
from typing import Optional

from colorama import init, Fore, Style

from utils import (
    build_task_dir_name,
    normalize_console_text,
    setup_logging,
    load_config,
    read_json,
    parse_cli_args_from_init,
    print_out,
    write_json,
    ExcelOperation
)
from tasks.task_manager import TaskManager
from benchmark.single_app_eval import SingleAppEval
from benchmark.multi_app_eval import MultiAppEval


def prepare_data_dir(provider: str,
                     query: str,
                     pass_idx: Optional[int] = None) -> str:
    task_id = build_task_dir_name(query, pass_idx)
    data_dir = os.path.join(os.environ['RESULTS_DIR'], provider, task_id)
    os.makedirs(data_dir, exist_ok=True)
    os.environ['DATA_DIR'] = data_dir
    return data_dir


def main() -> None:
    ########## basic config start ##########
    args = parse_cli_args_from_init()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_config = read_json(os.path.join(root_dir, args.get("file_setting_path")))
    bundle_name_dict = read_json(os.path.join(root_dir, file_path_config["app_package_config_path"]))
    load_config(
        root_dir, 
        file_path_config["results_dir"], 
        file_path_config["temp_dir"]
    )
    hdc_dir = os.path.join(root_dir, "hdc")
    os.environ["PATH"] = hdc_dir + os.pathsep + os.environ.get("PATH", "")
    ########## basic config ended ##########

    ########## welcome interface start ##########
    init()
    print_out(
        'Welcome to the AppAgent Testing CLI!',
        False,
        True,
        Fore.GREEN
    )
    ########## welcome interface ended ##########

    ########## execute start ##########
    if args.get("eval") == "true":
        eval_api_data = read_json(file_path_config["eval_api_path"])

        if args.get("eval_type") == "single":
            eval_dir = os.path.join(file_path_config["eval_dir"], args["provider"], "single_app")
            os.makedirs(eval_dir, exist_ok=True)
            usage_tracking_path = os.path.join(eval_dir, file_path_config["usage_tracking_name"])

            eval_datasets_path = file_path_config["eval_single_datasets_path"]
            eval_ins = SingleAppEval(eval_api_data["ocr_model"], eval_api_data["ocr_endpoint"], eval_api_data["ocr_api_key"], eval_api_data["llm_model"], eval_api_data["llm_endpoints"], eval_api_data["llm_api_key"], eval_api_data["llm_check_way"], usage_tracking_path)
        else:
            eval_dir = os.path.join(file_path_config["eval_dir"], args["provider"], "multi_app")
            os.makedirs(eval_dir, exist_ok=True)
            usage_tracking_path = os.path.join(eval_dir, file_path_config["usage_tracking_name"])

            eval_datasets_path = file_path_config["eval_multi_datasets_path"]
            eval_ins = MultiAppEval(eval_api_data["llm_model"], eval_api_data["llm_endpoints"], eval_api_data["llm_api_key"], eval_api_data["llm_check_way"], usage_tracking_path)
        
        datasets_list = read_json(eval_datasets_path)

        success_list_pass1 = []
        fail_list_pass1 = []

        success_list_pass3 = []
        fail_list_pass3 = []
        
        total_times = 0.0
        total_steps_reference = 0.0
        use_cache_tasks = 0
        datasets_len = len(datasets_list)

        if args["enable_cache"] == "true":
            excel_ins = ExcelOperation(args["provider"], os.path.join(file_path_config["eval_dir"], file_path_config["eval_excel_name_with_cache"]), 3)
        else:
            excel_ins = ExcelOperation(args["provider"], os.path.join(file_path_config["eval_dir"], file_path_config["eval_excel_name_no_cache"]), 3)

        for sub_dataset in datasets_list:
            excel_row = {
                "用例ID": sub_dataset["task_id"],
                "任务描述": sub_dataset["query"],
                "应用名称": sub_dataset["app"],
                "操控步数(GT)": sub_dataset["gt_steps"],
                "智能体的大模型": args["agent_llm"],
                "智能体大模型部署环境": args["agent_deployment_env"],
                "缓存复用的大模型": args["cache_llm"],
                "缓存复用大模型部署环境": args["cache_deployment_env"]
            }
            pass_3_total_step = 0
            pass_3_use_cache_flag = False
            pass_3_success_flag = False
            pass_3_eval_elapsed_time = 0
            pass_3_total_elapsed_time = 0

            for pass_idx in range(args["exector_times"]):
                prepare_data_dir(args["provider"], sub_dataset["query"], pass_idx)
                setup_logging(args.get("log_level"))

                print_out(f'Your command is {sub_dataset["query"]}', stdout=True)

                try:
                    task = TaskManager(
                        sub_dataset["query"], 
                        args.get("provider"), 
                        bundle_name_dict, 
                        args.get("hdc_command"), 
                        args.get("max_retries"), 
                        args.get("factor"),
                        args.get("max_execute_steps")
                    )
                    task.execute()
                except Exception as e:
                    print_out(
                        f'Task {sub_dataset["query"]} execution exception: {e}',
                        log_level="error"
                    )
                
                print_out(f'Here is the evaluation for query: {sub_dataset["query"]}', stdout=True)
                
                try:
                    if task.task_mgr.task_finished:
                        if args.get("eval_type") == "single":
                            evaluation_detail = eval_ins.eval(sub_dataset["query"], os.environ['DATA_DIR'], args["include_text"], args["single_eval_w_crop"], args["single_eval_h_crop"])
                        else:
                            evaluation_detail = eval_ins.eval(sub_dataset["query"], os.environ['DATA_DIR'], args["include_text"], 3, args["multi_eval_w_stage1_crop"], args["multi_eval_h_stage1_crop"], args["multi_eval_w_stage2_crop"], args["multi_eval_h_stage2_crop"])
                    else:
                        evaluation_detail = {
                            "eval_result": "fail",
                            "fine_detect_reason": "This task exceeds the maximum step limit or the maximum number of attempts."
                        }
                except Exception as e:
                    print_out(f"Eval is failed for the reason {e}", log_level="error", stdout=True)
                    evaluation_detail = {
                        "eval_result": "fail",
                        "fine_detect_reason": "This task exceeds the maximum step limit or the maximum number of attempts."
                    }

                evaluation_detail["use_cache_steps_ratio"] = task.task_mgr.use_cache_steps / (task.task_mgr.step_id + 1)

                print_out(f"The eval result is:\n{evaluation_detail}", stdout=True)
                write_json(os.path.join(os.environ['DATA_DIR'], "record.json"), evaluation_detail, "list", "a")

                pass_3_total_step = pass_3_total_step + task.task_mgr.step_id + 1
                if not pass_3_use_cache_flag and task.task_mgr.use_cache_steps != 0:
                    pass_3_use_cache_flag = True
                if not pass_3_success_flag and evaluation_detail["eval_result"] == "success":
                    pass_3_success_flag = True
                
                eval_elapsed_time = task.task_mgr.total_elapsed_time / (task.task_mgr.step_id + 1)
                total_times += eval_elapsed_time
                pass_3_eval_elapsed_time += eval_elapsed_time
                pass_3_total_elapsed_time += task.task_mgr.total_elapsed_time

                steps_reference = (task.task_mgr.step_id + 1) / sub_dataset["gt_steps"]
                total_steps_reference += steps_reference
                
                if pass_idx == 0:
                    excel_row["pass@1 操控步骤数"] = task.task_mgr.step_id + 1

                    if task.task_mgr.use_cache_steps != 0:
                        excel_row["pass@1 是否使用缓存"] = "是"
                    else:
                        excel_row["pass@1 是否使用缓存"] = "否"
                    
                    if evaluation_detail["eval_result"] == "success":
                        success_list_pass1.append({
                            "task_id": sub_dataset["task_id"],
                            "query": sub_dataset["query"]
                        })
                        excel_row["pass@1 是否成功"] = "是"
                    elif evaluation_detail["eval_result"] == "fail":
                        fail_list_pass1.append({
                            "task_id": sub_dataset["task_id"],
                            "query": sub_dataset["query"]
                        })
                        excel_row["pass@1 是否成功"] = "否"
                    elif evaluation_detail["eval_result"] == "error":
                        excel_row["pass@1 是否成功"] = "否"
                    
                    excel_row["pass@1 端到端耗时(s)"] = round(task.task_mgr.total_elapsed_time, 4)
                    excel_row["pass@1 单步操控时延(s)"] = round(eval_elapsed_time, 4)
                
                elif pass_idx == args["exector_times"] - 1:
                    excel_row["pass@3 操控步骤数"] = round(pass_3_total_step / 3, 1)
                    if pass_3_use_cache_flag:
                        excel_row["pass@3 是否使用缓存"] = "是"
                    else:
                        excel_row["pass@3 是否使用缓存"] = "否"
                    
                    if pass_3_success_flag:
                        success_list_pass3.append({
                            "task_id": sub_dataset["task_id"],
                            "query": sub_dataset["query"]
                        })
                        excel_row["pass@3 是否成功"] = "是"
                    else:
                        fail_list_pass3.append({
                            "task_id": sub_dataset["task_id"],
                            "query": sub_dataset["query"]
                        })
                        excel_row["pass@3 是否成功"] = "否"
                    
                    excel_row["pass@3 端到端耗时(s)"] = round(pass_3_total_elapsed_time / 3, 4)
                    excel_row["pass@3 单步操控时延(s)"] = round(pass_3_eval_elapsed_time / 3, 4)
            
            if pass_3_use_cache_flag:
                use_cache_tasks += 1
            
            excel_ins.insert_one_raw(excel_row)
            excel_ins.save()
        
        eval_acc_pass1 = len(success_list_pass1) / datasets_len
        eval_acc_pass3 = len(success_list_pass3) / datasets_len
        gt_steps_ratio = total_steps_reference / (datasets_len * args["exector_times"])
        eval_elapsed_time = total_times / (datasets_len * args["exector_times"])
        use_cache_ratio = use_cache_tasks / datasets_len

        eval_result_data = {
            "use_cache_tasks_ratio": use_cache_ratio,
            "gt_steps_ratio": gt_steps_ratio,
            "eval_elapsed_time": eval_elapsed_time,
            "eval_acc_pass@1": eval_acc_pass1,
            "eval_acc_pass@3": eval_acc_pass3,
            "success_task_pass@1": success_list_pass1,
            "fail_task_pass@1": fail_list_pass1,
            "success_task_pass@3": success_list_pass3,
            "fail_task_pass@3": fail_list_pass3,
        }

        eval_result_path = os.path.join(eval_dir, "eval.json")
        write_json(eval_result_path, eval_result_data)

        if args["enable_cache"] == "true":
            excel_ins.write_overall_excel(args["provider"], os.path.join(file_path_config["eval_dir"], file_path_config["eval_overall_excel_name_with_cache"]))
        else:
            excel_ins.write_overall_excel(args["provider"], os.path.join(file_path_config["eval_dir"], file_path_config["eval_overall_excel_name_no_cache"]))

        print_out(
            'Exiting the Mobile Testing CLI. Goodbye!',
            False,
            True,
            Fore.GREEN
        )
            
    else:
        while True:
            user_query = input(Fore.YELLOW + "Enter your command (or 'exit' to quit): " + Style.RESET_ALL)
            if platform.system().lower() == "windows":
                user_query = normalize_console_text(user_query)

            if user_query.lower() == 'exit':
                print_out(
                    'Exiting the Mobile Testing CLI. Goodbye!',
                    False,
                    True,
                    Fore.GREEN
                )
                break

            prepare_data_dir(args["provider"], user_query)
            setup_logging(args.get("log_level"))

            print_out("Enter your command (or 'exit' to quit): ")
            print_out(f"Your command is {user_query}")

            try:
                task = TaskManager(
                    user_query, 
                    args.get("provider"), 
                    bundle_name_dict, 
                    args.get("hdc_command"), 
                    args.get("max_retries"), 
                    args.get("factor"),
                    args.get("max_execute_steps")
                )
                task.execute()
            except Exception as e:
                print_out(
                    f'Task {user_query} execution exception: {e}',
                    log_level="error"
                )
    ########## execute ended ##########


if __name__ == '__main__':
    main()
