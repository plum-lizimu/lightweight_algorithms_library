import os

# def logger(save_path, log_str, no_save=False):
#     if no_save == False:
#         with open(os.path.join(save_path, 'record.log'), 'a') as file:
#             file.writelines(log_str + '\n')
#             print(log_str)
#     else:
#         print(log_str)

class logger:

    def __init__(self, save_path: str) -> None:
        self.save_path = save_path
    
    def __call__(self, log_str: str, no_save: bool=False):
        self.forward(log_str, no_save)

    def forward(self, log_str: str, no_save: bool):
        if no_save:
            print(log_str)
        else:
            with open(os.path.join(self.save_path, 'record.log'), 'a') as file:
                file.writelines(log_str + '\n')
                print(log_str)