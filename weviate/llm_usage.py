import os
import json
import datetime

class UsageUpdater:

    def update_usage(self, details):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")

        usage_dict = {
            "Total Tokens": details.total_tokens,
            "Prompt Tokens": details.prompt_tokens,
            "Completion Tokens": details.completion_tokens,
            "Total Cost (USD)": details.total_cost,
            "Time Stamp": current_time
        }
        

        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        file_name = f"{current_date}_usage.json"

        if not os.path.exists("output"):
            os.makedirs("output")

        self.json_file_path = os.path.join("output", file_name)

        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as file:
                existing_data = json.load(file)
                existing_data["day_cost"] += usage_dict["Total Cost (USD)"] 
                existing_data["usage_track"].append(usage_dict)
            with open(self.json_file_path, "w") as file:
                json.dump(existing_data, file, indent=4)
            #print(f"Data appended to '{self.json_file_path}'.")
        else:
            with open(self.json_file_path, "w") as file:
                json.dump({ "usage_track": [usage_dict], "day_cost": usage_dict["Total Cost (USD)"] }, file, indent=4)
            #print(f"File '{self.json_file_path}' created and data written.")

    def get_daily_usage(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as file:
                existing_data = json.load(file)
                return existing_data["day_cost"]
        else:
            print("No usage as of todays record.")
        